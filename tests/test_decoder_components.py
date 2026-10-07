"""Replaceable decoder components: registry, contract, kinds, checkpoints, pipeline.

The suite covers the component API end to end on the native encoder and tiny
synthetic clips: an architecture that is *not* the built-in conv upsampler is
registered, trained, evaluated, saved, reloaded, resumed and rendered through the
same path, and the legacy checkpoint format (schema 1, the one that predates
selectable architectures) is read, checked and resumed.

The custom architecture used here is the runnable example in
``examples/custom_decoder.py`` -- importing it registers ``"pixelshuffle"``, which
is exactly what a user does -- so the documented example is exercised, not a test
double that happens to share its name.
"""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from wpm_video.config import (DecoderConfig, DecoderTrainConfig, RunConfig, decoder_architecture,
                              decoder_config_snapshot, validate_decoder_config)
from wpm_video.dataset import TokenDataset, cache_video_tokens
from wpm_video.decoder import (DecoderCompatibilityError, DecoderRegistrationError, RGBDecoder,
                               available_decoders, build_decoder, check_decoder_compatibility,
                               decode_latents, decoder_fingerprint, decoder_identity, identity_gaps,
                               load_decoder, register_decoder, render_latents,
                               run_decoder_training, save_decoder)
from wpm_video.decoder.compat import stored_architecture
from wpm_video.decoder.model import (DECODER_SCHEMA_VERSION, LEGACY_DECODER_CONFIG_FIELDS,
                                     LEGACY_DECODER_SCHEMA_VERSION, RESERVED_CHECKPOINT_FIELDS,
                                     checkpoint_component_kind, checkpoint_schema_version)
from wpm_video.model import VideoWorldModel
from wpm_video.train import build_model, fit_projection, train

from .fixtures import make_clips, native_encoder, tiny_config

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "custom_decoder.py"


def _load_example_module():
    """Import the example by path; importing it registers its component."""
    spec = importlib.util.spec_from_file_location("wpm_example_custom_decoder", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example_module()          # registers example.KIND in this process
CUSTOM_KIND = example.KIND


class StretchDecoder(RGBDecoder):
    """Test-only component: per-cell linear map, then a resize to the output edge.

    It deliberately has no relationship between the patch grid and the output size,
    which is what the shared contract allows and the conv architecture does not: the
    internal resolution is the grid times a fixed block, and the configured output
    edge can be any valid one (smaller than the grid, or not a whole multiple of it).
    """

    BLOCK = 4

    def __init__(self, config: DecoderConfig, patches: int, d_world: int):
        super().__init__(config, patches, d_world)
        self.to_block = nn.Linear(d_world, 3 * self.BLOCK * self.BLOCK)

    def forward(self, latents):
        self.check_input(latents)
        rows, cols = self.grid
        blocks = self.to_block(latents).reshape(latents.shape[0], rows, cols, 3, self.BLOCK,
                                                self.BLOCK)
        image = blocks.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, rows * self.BLOCK,
                                                         cols * self.BLOCK)
        image = nn.functional.interpolate(image, size=(self.output_size, self.output_size),
                                          mode="bilinear", align_corners=False)
        return torch.sigmoid(image)


register_decoder("test_stretch", StretchDecoder)


def custom_decoder_config(train_videos, val_videos, steps: int = 20, image_size: int = 48,
                          hidden: int = 32) -> RunConfig:
    """The tiny native configuration with the custom component instead of conv.

    ``image_size=48`` on a 4x4 patch grid is a factor of 12: not a power of two,
    which the conv architecture refuses and this one does not care about, so every
    run below also proves the pipeline itself imposes no conv-only constraint.
    """
    config = tiny_config(train_videos, val_videos)
    config.decoder = DecoderConfig(kind=CUSTOM_KIND, image_size=image_size,
                                   options={"hidden": hidden})
    config.decoder_train = DecoderTrainConfig(seed=0, batch_windows=2, learning_rate=2e-3,
                                              max_steps=steps, eval_interval=steps, eval_batches=1,
                                              log_interval=1, max_wall_seconds=600.0,
                                              frame_cache_videos=2)
    config.validate()
    return config


def conv_decoder_config(train_videos, val_videos, steps: int = 3) -> RunConfig:
    """The tiny native configuration with the built-in conv component."""
    config = tiny_config(train_videos, val_videos)
    config.decoder = DecoderConfig(image_size=64, base_channels=16, channel_multipliers=[1, 2],
                                   stem_blocks=1, blocks_per_stage=1)
    config.decoder_train = DecoderTrainConfig(seed=0, batch_windows=2, learning_rate=2e-3,
                                              max_steps=steps, eval_interval=steps, eval_batches=1,
                                              log_interval=1, max_wall_seconds=600.0,
                                              frame_cache_videos=2)
    config.validate()
    return config


def downgrade_to_schema1(payload: dict) -> dict:
    """Rewrite a conv checkpoint into the exact shape v0.4.0 wrote (schema 1).

    No ``kind``, no ``options``: schema 1 predates selectable architectures, which is
    what makes reading it unambiguous. Used to check that such a file still loads,
    passes compatibility and can be resumed.
    """
    legacy = dict(payload)
    legacy["schema_version"] = LEGACY_DECODER_SCHEMA_VERSION
    legacy["decoder_config"] = {name: payload["decoder_config"][name]
                                for name in LEGACY_DECODER_CONFIG_FIELDS}
    legacy["architecture"] = {name: payload["architecture"][name]
                              for name in LEGACY_DECODER_CONFIG_FIELDS}
    return legacy


# -- broken components, registered once to test the registry's return checks ----
class _WrongGrid(example.PixelShuffleDecoder):
    def __init__(self, config, patches, d_world):
        super().__init__(config, patches, d_world)
        self.grid = (1, 1)


class _WrongOutput(example.PixelShuffleDecoder):
    @property
    def output_size(self):
        return int(self.config.image_size) * 2


class _WrongConfig(example.PixelShuffleDecoder):
    """Same kind and geometry, but the module claims a configuration it was not built from."""

    def __init__(self, config, patches, d_world):
        super().__init__(config, patches, d_world)
        self.config = DecoderConfig(kind=config.kind, image_size=config.image_size,
                                    options={"hidden": 64})


class _WrongKind(example.PixelShuffleDecoder):
    def __init__(self, config, patches, d_world):
        super().__init__(config, patches, d_world)
        self.config.kind = "conv"


register_decoder("test_broken_return", lambda config, patches, d_world: nn.Linear(1, 1))
register_decoder("test_broken_grid", _WrongGrid)
register_decoder("test_broken_output", _WrongOutput)
register_decoder("test_broken_config", _WrongConfig)
register_decoder("test_broken_kind", _WrongKind)
register_decoder("test_echo_geometry", example.PixelShuffleDecoder)


# -- a shared world model, trained once for the pipeline classes ----------------
_WORKSPACE: dict = {}


def setUpModule():
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    video_dir, cache_dir = root / "videos", root / "cache"
    names = [path.stem for path in make_clips(video_dir, 3)]
    config = tiny_config(names[:2], names[2:])
    config.data.video_dir = str(video_dir)
    config.data.cache_dir = str(cache_dir)
    encoder = native_encoder()
    for name in names:
        cache_video_tokens(video_dir / f"{name}.mp4", config.data, encoder, cache_dir,
                           batch_clips=2)
    train_set = TokenDataset(config.data, cache_dir, config.encoder, "train")
    val_set = TokenDataset(config.data, cache_dir, config.encoder, "val")
    torch.manual_seed(0)
    world = build_model(config, encoder, torch.device("cpu"))
    fit_projection(world, train_set)
    train(config, world, train_set, val_set, root / "world", torch.device("cpu"),
          provenance={"splits": {"train": names[:2], "val": names[2:]}})
    _WORKSPACE.update(temporary=temporary, root=root, video_dir=video_dir, cache_dir=cache_dir,
                      names=names, world_checkpoint=root / "world" / "final.pt")


def tearDownModule():
    _WORKSPACE["temporary"].cleanup()


class RegistryTests(unittest.TestCase):
    """Registration is explicit, process-local and validated."""

    def test_builtin_conv_is_registered(self):
        self.assertIn("conv", available_decoders())
        self.assertEqual(list(available_decoders()), sorted(available_decoders()))

    def test_duplicate_registration_is_refused_including_the_builtin(self):
        for kind in ("conv", CUSTOM_KIND, "test_echo_geometry"):
            with self.assertRaisesRegex(DecoderRegistrationError, "already registered"):
                register_decoder(kind, example.PixelShuffleDecoder)

    def test_invalid_kinds_and_factories_are_refused(self):
        for kind in ("", "Conv", "1kind", "with space", "with-dash", None, 7):
            with self.assertRaises(DecoderRegistrationError, msg=kind):
                register_decoder(kind, example.PixelShuffleDecoder)
        with self.assertRaisesRegex(DecoderRegistrationError, "callable"):
            register_decoder("test_not_callable", "not a factory")

    def test_unknown_kind_fails_in_build_and_load(self):
        config = DecoderConfig(kind="ghost_arch", image_size=64)
        validate_decoder_config(config)   # syntax and shared fields need no registry
        with self.assertRaisesRegex(DecoderRegistrationError, "not registered") as caught:
            build_decoder(config, 16, 32)
        self.assertIn("conv", str(caught.exception))       # the message lists what exists

    def test_a_factory_result_is_checked_against_the_request(self):
        cases = [
            ("test_broken_return", "not an RGBDecoder"),
            ("test_broken_grid", "grid"),
            ("test_broken_output", "output"),
            ("test_broken_config", "configuration"),
            ("test_broken_kind", "kind"),
        ]
        for kind, expected in cases:
            with self.assertRaises(DecoderRegistrationError, msg=kind) as caught:
                build_decoder(DecoderConfig(kind=kind, image_size=48), 16, 32)
            self.assertIn(expected, str(caught.exception))
        # a component that reports what it was asked for is accepted
        decoder = build_decoder(DecoderConfig(kind="test_echo_geometry", image_size=48), 16, 32)
        self.assertEqual(decoder.kind, "test_echo_geometry")

    def test_a_custom_component_reports_the_shared_geometry(self):
        decoder = build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                              options={"hidden": 16}), 16, 32)
        self.assertIsInstance(decoder, RGBDecoder)
        self.assertEqual((decoder.kind, decoder.output_size, decoder.grid), (CUSTOM_KIND, 48,
                                                                             (4, 4)))
        self.assertEqual((decoder.patches, decoder.d_world), (16, 32))
        self.assertGreater(decoder.parameter_count(), 0)
        # genuinely not a convolutional architecture: nothing in it is a conv layer
        self.assertFalse(any(isinstance(module, nn.Conv2d) for module in decoder.modules()))
        images = decoder(torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(0)))
        self.assertEqual(tuple(images.shape), (2, 3, 48, 48))
        self.assertGreaterEqual(float(images.detach().min()), 0.0)
        self.assertLessEqual(float(images.detach().max()), 1.0)

    def test_the_base_class_assumes_no_architecture(self):
        """RGBDecoder itself owns the shared geometry and refuses to invent pixels."""
        decoder = RGBDecoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1), 16, 32)
        self.assertEqual(decoder.parameter_count(), 0)          # no layers of its own
        self.assertEqual((decoder.grid, decoder.output_size, decoder.kind), ((4, 4), 64, "conv"))
        with self.assertRaises(NotImplementedError):
            decoder(torch.randn(2, 16, 32))
        # the shared geometry rules are the *data's*: a square patch grid and positive
        # dimensions. How the output is produced is the architecture's business.
        with self.assertRaisesRegex(ValueError, "perfect square"):
            RGBDecoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1), 17, 32)
        with self.assertRaisesRegex(ValueError, "d_world"):
            RGBDecoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1), 16, 0)
        for size, patches in ((40, 256), (32, 4096)):
            shared = RGBDecoder(DecoderConfig(image_size=size, base_channels=8, stem_blocks=1),
                                patches, 32)
            self.assertEqual((shared.output_size, shared.grid), (size, (16, 16) if patches == 256
                                                                 else (64, 64)))

    def test_output_geometry_is_the_architecture_s_own_constraint(self):
        """A component may resize freely; the conv stack may not."""
        # 40 / 16 is not a whole factor, and 32 is smaller than a 64x64 patch grid:
        # both are valid configurations for an architecture without upsampling stages
        for size, patches in ((40, 256), (32, 4096)):
            decoder = build_decoder(DecoderConfig(kind="test_stretch", image_size=size), patches,
                                    32)
            images = decoder(torch.randn(1, patches, 32, generator=torch.Generator().manual_seed(0)))
            self.assertEqual(tuple(images.shape), (1, 3, size, size))
        # ... and the same configurations are refused by the conv architecture
        with self.assertRaisesRegex(ValueError, "multiple of the patch grid"):
            build_decoder(DecoderConfig(image_size=40, base_channels=8, stem_blocks=1), 256, 32)
        with self.assertRaisesRegex(ValueError, "smaller than the patch grid"):
            build_decoder(DecoderConfig(image_size=32, base_channels=8, stem_blocks=1), 4096, 32)
        # the example architecture needs its own whole factor, and says so itself
        with self.assertRaisesRegex(ValueError, "whole multiple of the patch grid"):
            build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=40), 256, 32)

    def test_the_module_mode_is_allowed_to_matter(self):
        """State-free input is the contract; train/eval behaviour is the architecture's.

        A component with batch normalisation is legitimate: rendering and evaluation
        run it in eval mode, training in train mode.
        """
        class ModeDependent(RGBDecoder):
            def __init__(self, config, patches, d_world):
                super().__init__(config, patches, d_world)
                self.norm = nn.BatchNorm1d(d_world)
                self.head = nn.Linear(d_world, 3 * self.output_size ** 2)

            def forward(self, latents):
                self.check_input(latents)
                pooled = self.norm(latents.mean(dim=1))
                return torch.sigmoid(self.head(pooled)).reshape(
                    -1, 3, self.output_size, self.output_size)

        register_decoder("test_mode_dependent", ModeDependent)
        decoder = build_decoder(DecoderConfig(kind="test_mode_dependent", image_size=32), 16, 32)
        latents = torch.randn(4, 16, 32, generator=torch.Generator().manual_seed(1))
        decoder.eval()
        first = decoder(latents)
        self.assertTrue(torch.equal(first, decoder(latents)))     # eval mode is deterministic
        with tempfile.TemporaryDirectory() as temporary:
            images = decode_latents(decoder, {"h1": latents[0]}, torch.device("cpu"))
            self.assertEqual(tuple(images["h1"].shape), (3, 32, 32))
        self.assertFalse(decoder.training)                        # mode restored by decode_latents

    def test_component_modules_do_not_depend_on_the_dynamics(self):
        """The architecture layer stays a leaf: importing it must not pull in training.

        Checked on the sources, because within one process every module is already
        imported by the package initialiser.
        """
        source_root = Path(__file__).resolve().parents[1] / "src" / "wpm_video"
        forbidden = {
            "config.py": ("decoder",),
            "decoder/base.py": (".compat", ".targets", ".render", ".train", "..model",
                                "..world_state", "..predict", "..train", "..encoder", "..dataset"),
            "decoder/registry.py": (".architectures", ".compat", ".train", ".targets", ".render",
                                    "..model", "..world_state", "..predict", "..train",
                                    "..encoder", "..dataset"),
            "decoder/architectures/conv.py": (".compat", ".train", ".targets", ".render",
                                              "..model", "..world_state", "..predict", "..train",
                                              "..encoder", "..dataset"),
        }
        for relative, patterns in forbidden.items():
            path = source_root / relative
            imports = []
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imports += [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imports.append("." * node.level + (node.module or ""))
            for pattern in patterns:
                offenders = [name for name in imports
                             if name == pattern or (pattern.startswith(("..", "."))
                                                    and name.startswith(pattern))]
                self.assertEqual(offenders, [], f"{relative} imports {offenders}")


class DecoderKindConfigTests(unittest.TestCase):
    def test_kind_and_options_are_appended_after_the_legacy_fields(self):
        """A positional legacy configuration keeps its exact meaning."""
        positional = DecoderConfig(96, 32, [1, 2], 3, 2)
        self.assertEqual(positional, DecoderConfig(image_size=96, base_channels=32,
                                                   channel_multipliers=[1, 2], stem_blocks=3,
                                                   blocks_per_stage=2))
        self.assertEqual((positional.kind, positional.options), ("conv", {}))
        # ... and the legacy JSON of a released configuration still loads unchanged
        payload = {"image_size": 64, "base_channels": 16, "channel_multipliers": [1, 2],
                   "stem_blocks": 1, "blocks_per_stage": 1}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config = tiny_config(["a"], ["b"])
            body = config.to_dict()
            body["decoder"] = payload
            path.write_text(json.dumps(body), encoding="utf-8")
            restored = RunConfig.load(path)
        self.assertEqual(restored.decoder.kind, "conv")
        self.assertEqual(restored.decoder.options, {})
        self.assertEqual(restored.decoder.base_channels, 16)
        self.assertEqual(decoder_architecture(restored.decoder),
                         {"kind": "conv", "image_size": 64, "base_channels": 16,
                          "channel_multipliers": [1, 2], "stem_blocks": 1, "blocks_per_stage": 1})

    def test_a_custom_kind_describes_itself_through_nested_options(self):
        config = DecoderConfig(kind=CUSTOM_KIND, image_size=48, options={"hidden": 16})
        validate_decoder_config(config)
        self.assertEqual(decoder_architecture(config),
                         {"kind": CUSTOM_KIND, "image_size": 48, "options": {"hidden": 16}})
        # the conv widths are refused rather than silently ignored
        with self.assertRaisesRegex(ValueError, "only configure kind 'conv'"):
            validate_decoder_config(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                                 base_channels=8))
        # ... and the conv architecture refuses the options that belong to a custom kind
        with self.assertRaisesRegex(ValueError, "must be empty for kind 'conv'"):
            validate_decoder_config(DecoderConfig(options={"hidden": 16}))

    def test_option_names_are_namespaced_and_cannot_shadow_the_identity(self):
        """An option may be called anything: it can never override the shared fields."""
        config = DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                               options={"kind": "something else", "grid": [1], "image_size": 7,
                                        "patches": 0, "options": {"nested": True}})
        validate_decoder_config(config)
        record = decoder_architecture(config)
        self.assertEqual(record["kind"], CUSTOM_KIND)
        self.assertEqual(record["image_size"], 48)
        self.assertEqual(record["options"]["kind"], "something else")
        self.assertEqual(record["options"]["image_size"], 7)
        self.assertEqual(sorted(record), ["image_size", "kind", "options"])

    def test_options_must_survive_a_json_round_trip_unchanged(self):
        for options, expected in (
            (["hidden"], "must be a dict"),
            ({1: 4}, "keys must be strings"),
            ({"hidden": float("nan")}, "finite"),
            ({"hidden": (1, 2)}, "tuples"),
            ({"hidden": {3, 4}}, "sets"),
            ({"nested": {"width": [1, 2, {"deep": float("inf")}]}}, "finite"),
        ):
            with self.assertRaisesRegex(ValueError, expected, msg=options):
                validate_decoder_config(DecoderConfig(kind=CUSTOM_KIND, options=options))
        # a chain of mappings that is not a dict would come back as one: refused
        from collections import ChainMap
        with self.assertRaisesRegex(ValueError, "dict"):
            validate_decoder_config(DecoderConfig(kind=CUSTOM_KIND,
                                                  options={"nested": ChainMap({"a": 1})}))
        # a cycle is a clear error, not a recursion failure
        cycle = {"hidden": 16}
        cycle["self"] = cycle
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_decoder_config(DecoderConfig(kind=CUSTOM_KIND, options=cycle))
        nested_cycle = {"width": [1, 2]}
        nested_cycle["width"].append(nested_cycle)
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_decoder_config(DecoderConfig(kind=CUSTOM_KIND, options=nested_cycle))

    def test_options_keep_their_types_through_the_record(self):
        options = {"hidden": 16, "ratio": 2.5, "flag": False, "nothing": None,
                   "taps": [1, 2], "nested": {"a": "b", "deep": [{"c": 3}]}}
        config = DecoderConfig(kind=CUSTOM_KIND, image_size=48, options=options)
        validate_decoder_config(config)
        # the record is a JSON copy: same values, and every type survives it
        record = decoder_architecture(config)
        self.assertEqual(record["options"], options)
        self.assertIs(type(record["options"]["hidden"]), int)
        self.assertIs(type(record["options"]["ratio"]), float)
        self.assertIs(type(record["options"]["flag"]), bool)
        self.assertIsNone(record["options"]["nothing"])
        self.assertEqual(json.loads(json.dumps(record["options"])), record["options"])

    def test_the_configuration_is_deep_copied_into_the_module(self):
        """Editing the configuration afterwards cannot change what a module records."""
        options = {"hidden": 16, "nested": {"width": [1, 2]}}
        custom = build_decoder(DecoderConfig(kind="test_stretch", image_size=48, options=options),
                               16, 32)
        record = decoder_architecture(custom.config)
        options["hidden"] = 4096                     # caller keeps editing its own objects
        options["nested"]["width"].append(3)
        self.assertEqual(custom.config.options, {"hidden": 16, "nested": {"width": [1, 2]}})
        self.assertEqual(decoder_architecture(custom.config), record)
        self.assertEqual(decoder_config_snapshot(custom.config)["options"],
                         {"hidden": 16, "nested": {"width": [1, 2]}})
        # the same holds for the conv shape fields, which are lists too
        config = DecoderConfig(image_size=64, base_channels=8, channel_multipliers=[1, 2],
                               stem_blocks=1)
        conv = build_decoder(config, 16, 32)
        conv_record = decoder_architecture(conv.config)
        config.channel_multipliers.append(8)
        config.image_size = 32
        self.assertEqual(conv.config.channel_multipliers, [1, 2])
        self.assertEqual(conv.output_size, 64)
        self.assertEqual(decoder_architecture(conv.config), conv_record)

    def test_a_custom_decoder_validates_its_own_options(self):
        """Options are the architecture's business; the shared layer only checks JSON."""
        decoder = build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                              options={"hidden": 8}), 16, 32)
        self.assertEqual(decoder.config.options["hidden"], 8)
        with self.assertRaisesRegex(ValueError, "hidden"):
            build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                        options={"hidden": 2}), 16, 32)
        # a typo is refused instead of silently falling back to the default
        with self.assertRaisesRegex(ValueError, "unknown \\['hiden'\\]"):
            build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                        options={"hiden": 8}), 16, 32)


class DecoderCheckpointTests(unittest.TestCase):
    """Schema 2 written now, schema 1 read exactly, malformed files refused."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.config = conv_decoder_config(["a"], ["b"])
        cls.decoder = build_decoder(cls.config.decoder, 16, 32)
        cls.world, _ = VideoWorldModel.from_checkpoint(_WORKSPACE["world_checkpoint"])

    def round_trip(self, decoder, directory, extra=None) -> dict:
        path = Path(directory) / "decoder.pt"
        save_decoder(path, decoder, extra)
        _, payload = load_decoder(path)
        return payload

    def test_schema2_round_trip_records_the_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.round_trip(self.decoder, temporary)
            self.assertEqual(payload["schema_version"], DECODER_SCHEMA_VERSION)
            self.assertEqual(payload["kind"], "rgb_decoder")
            self.assertEqual(payload["decoder_config"]["kind"], "conv")
            self.assertEqual(payload["decoder_config"]["options"], {})
            self.assertEqual(payload["architecture"]["kind"], "conv")
            self.assertEqual(decoder_fingerprint(payload)["component_kind"], "conv")

    def test_a_custom_component_round_trips_through_its_own_kind(self):
        config = DecoderConfig(kind=CUSTOM_KIND, image_size=48, options={"hidden": 16})
        decoder = build_decoder(config, 16, 32)
        latents = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.pt"
            save_decoder(path, decoder)
            restored, payload = load_decoder(path)
            self.assertIsInstance(restored, example.PixelShuffleDecoder)
            self.assertEqual(restored.kind, CUSTOM_KIND)
            self.assertEqual(payload["architecture"],
                             {"kind": CUSTOM_KIND, "image_size": 48, "options": {"hidden": 16}})
            self.assertTrue(torch.equal(decoder(latents), restored(latents)))
            # the recorded configuration is what makes the architecture reproducible
            self.assertEqual(payload["decoder_config"]["options"], {"hidden": 16})

    def test_legacy_schema1_checkpoint_loads_and_renders_identically(self):
        latents = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(2))
        with tempfile.TemporaryDirectory() as temporary:
            current = Path(temporary) / "current.pt"
            # saved the way a training run does it: weights plus the world identity
            save_decoder(current, self.decoder,
                         decoder_identity(self.decoder, self.world, self.config))
            payload = torch.load(current, map_location="cpu", weights_only=False)
            legacy_path = Path(temporary) / "legacy.pt"
            torch.save(downgrade_to_schema1(payload), legacy_path)

            decoder, legacy = load_decoder(legacy_path)
            # read as legacy conv, and the loaded payload is *not* rewritten to say so:
            # normalization happens in the comparison, never in the file's metadata
            self.assertEqual(legacy["schema_version"], LEGACY_DECODER_SCHEMA_VERSION)
            self.assertNotIn("kind", legacy["decoder_config"])
            self.assertEqual(checkpoint_component_kind(legacy), "conv")
            self.assertEqual(stored_architecture(legacy)["kind"], "conv")
            self.assertEqual(decoder.kind, "conv")
            self.assertEqual(identity_gaps(legacy), [])
            self.assertEqual(stored_architecture(legacy),
                             decoder_architecture(DecoderConfig(image_size=64, base_channels=16,
                                                                channel_multipliers=[1, 2],
                                                                stem_blocks=1,
                                                                blocks_per_stage=1)))
            # every weight and every pixel is the same as through the current schema
            for key, value in self.decoder.state_dict().items():
                self.assertTrue(torch.equal(value, decoder.state_dict()[key]), key)
            with torch.no_grad():
                self.assertTrue(torch.equal(self.decoder(latents), decoder(latents)))
            identity = check_decoder_compatibility(legacy, self.world, self.config, decoder)
            self.assertEqual(sorted(identity), ["encoder", "frame_target", "sampling",
                                                "world_projection"])
            self.assertEqual(decoder_fingerprint(legacy)["component_kind"], "conv")

    def test_schema1_metadata_is_not_repaired_into_agreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "current.pt"
            save_decoder(path, self.decoder)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            mutations = (
                (lambda p: p["decoder_config"].__setitem__("kind", "conv"), "schema-1"),
                (lambda p: p["decoder_config"].pop("stem_blocks"), "stem_blocks"),
                (lambda p: p["architecture"].__setitem__("kind", "conv"), "schema-1"),
                (lambda p: p["architecture"].pop("base_channels"), "architecture record"),
            )
            for mutate, expected in mutations:
                altered = downgrade_to_schema1(payload)
                mutate(altered)
                broken = Path(temporary) / "broken.pt"
                torch.save(altered, broken)
                with self.assertRaisesRegex(ValueError, expected, msg=expected):
                    load_decoder(broken)

    def test_schema2_metadata_must_be_complete_and_consistent(self):
        """A truncated record is an error, never completed with defaults."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "current.pt"
            save_decoder(path, self.decoder)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for mutate, expected in (
                (lambda p: p["decoder_config"].pop("kind"), "decoder_config.kind"),
                (lambda p: p["decoder_config"].pop("options"), "decoder_config.options"),
                (lambda p: p["decoder_config"].pop("image_size"), "decoder_config.image_size"),
                (lambda p: p["decoder_config"].pop("base_channels"), "decoder_config.base_channels"),
                (lambda p: p["decoder_config"].pop("stem_blocks"), "decoder_config.stem_blocks"),
                (lambda p: p["decoder_config"].pop("channel_multipliers"),
                 "decoder_config.channel_multipliers"),
                (lambda p: p["decoder_config"].pop("blocks_per_stage"),
                 "decoder_config.blocks_per_stage"),
                (lambda p: p["decoder_config"].__setitem__("widths", [1, 2]), "does not know"),
                (lambda p: p["architecture"].__setitem__("image_size", 32), "inconsistent"),
                (lambda p: p["architecture"].pop("kind"), "inconsistent"),
                (lambda p: p.__setitem__("grid", [1, 1]), "grid"),
                (lambda p: p.__setitem__("grid", 4), "grid"),          # malformed, not a TypeError
                (lambda p: p.__setitem__("patches", 17), "perfect square"),
                (lambda p: p.__setitem__("d_world", 0), "positive integers"),
                (lambda p: p.__setitem__("state_dict", {}), "no decoder weights"),
                (lambda p: p["state_dict"].__setitem__("position", "not a tensor"),
                 "other than tensors"),
                (lambda p: p.__setitem__("schema_version", 99), "schema_version"),
                (lambda p: p.__setitem__("schema_version", True), "schema_version"),
                (lambda p: p.__setitem__("schema_version", "2"), "schema_version"),
            ):
                altered = copy.deepcopy(payload)
                mutate(altered)
                broken = Path(temporary) / "broken.pt"
                torch.save(altered, broken)
                with self.assertRaisesRegex(ValueError, expected, msg=expected):
                    load_decoder(broken)

    def test_a_custom_checkpoint_does_not_need_the_conv_fields(self):
        """Only conv describes itself with the conv shape fields."""
        decoder = build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                              options={"hidden": 16}), 16, 32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.pt"
            save_decoder(path, decoder)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for field in ("base_channels", "channel_multipliers", "stem_blocks",
                          "blocks_per_stage"):
                payload["decoder_config"].pop(field)
            trimmed = Path(temporary) / "trimmed.pt"
            torch.save(payload, trimmed)
            restored, _ = load_decoder(trimmed)
            self.assertEqual(restored.kind, CUSTOM_KIND)
            self.assertEqual(restored.config.options, {"hidden": 16})

    def test_an_unregistered_kind_fails_before_any_weight_is_touched(self):
        """A component this installation does not have is an error, never a fallback."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "current.pt"
            save_decoder(path, build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                                           options={"hidden": 16}), 16, 32))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["decoder_config"] = {**payload["decoder_config"], "kind": "ghost_arch"}
            payload["architecture"] = {**payload["architecture"], "kind": "ghost_arch"}
            broken = Path(temporary) / "ghost.pt"
            torch.save(payload, broken)
            with self.assertRaisesRegex(DecoderRegistrationError, "register_decoder"):
                load_decoder(broken)
            # the kind is *reported* as written, never guessed from the shapes
            self.assertEqual(checkpoint_component_kind(payload), "ghost_arch")

    def test_reserved_metadata_cannot_override_the_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            for field in sorted(RESERVED_CHECKPOINT_FIELDS):
                with self.assertRaisesRegex(ValueError, "may not override", msg=field):
                    save_decoder(Path(temporary) / "x.pt", self.decoder, {field: "anything"})
            payload = self.round_trip(self.decoder, temporary, {"provenance": {"run": "local"},
                                                               "step": 3})
            self.assertEqual(payload["provenance"], {"run": "local"})
            self.assertEqual(payload["step"], 3)
            self.assertEqual(payload["schema_version"], DECODER_SCHEMA_VERSION)
            self.assertEqual(payload["patches"], 16)

    def test_weights_that_do_not_fit_the_recorded_architecture_are_refused(self):
        bigger = build_decoder(DecoderConfig(image_size=64, base_channels=32,
                                             channel_multipliers=[1, 2], stem_blocks=1), 16, 32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mismatch.pt"
            save_decoder(path, bigger)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            # describe the small architecture while keeping the big weights
            payload["decoder_config"] = {**payload["decoder_config"], "base_channels": 16}
            payload["architecture"] = {**payload["architecture"], "base_channels": 16}
            broken = Path(temporary) / "broken.pt"
            torch.save(payload, broken)
            with self.assertRaisesRegex(ValueError, "do not fit"):
                load_decoder(broken)

    def test_a_world_checkpoint_is_still_not_a_decoder_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "rgb_decoder"):
            load_decoder(_WORKSPACE["world_checkpoint"])

    def test_compatibility_checks_the_recorded_configuration_without_a_module(self):
        """The checkpoint must agree with itself even when no decoder is compared."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "current.pt"
            save_decoder(path, self.decoder,
                         decoder_identity(self.decoder, self.world, self.config))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            check_decoder_compatibility(payload, self.world, self.config)   # no module needed
            # a record whose architecture was edited to describe another configuration
            altered = copy.deepcopy(payload)
            altered["architecture"] = {**altered["architecture"], "base_channels": 32}
            with self.assertRaisesRegex(DecoderCompatibilityError, "architecture differs"):
                check_decoder_compatibility(altered, self.world, self.config)
            # ... and a truncated configuration is reported the same way
            altered = copy.deepcopy(payload)
            altered["decoder_config"].pop("stem_blocks")
            with self.assertRaisesRegex(DecoderCompatibilityError, "stem_blocks"):
                check_decoder_compatibility(altered, self.world, self.config)


class ConvLegacyPipelineTests(unittest.TestCase):
    """The built-in conv component still trains, resumes and compares as before."""

    def train(self, name: str, steps: int, resume="", config=None):
        config = config or conv_decoder_config(_WORKSPACE["names"][:2], _WORKSPACE["names"][2:])
        config.data.video_dir = str(_WORKSPACE["video_dir"])
        config.data.cache_dir = str(_WORKSPACE["cache_dir"])
        config.decoder_train.max_steps = steps
        out = _WORKSPACE["root"] / name
        run_decoder_training(config, _WORKSPACE["world_checkpoint"], out, torch.device("cpu"),
                             resume=resume)
        return out

    def test_a_schema1_checkpoint_resumes_the_same_trajectory(self):
        short = self.train("legacy_short", 3)
        path = short / "final.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["schema_version"], DECODER_SCHEMA_VERSION)
        torch.save(downgrade_to_schema1(payload), path)       # what v0.4.0 would have written
        live = self.train("legacy_live", 6)
        resumed = self.train("legacy_resumed", 6, resume=str(path))
        live_payload = torch.load(live / "final.pt", map_location="cpu", weights_only=False)
        resumed_payload = torch.load(resumed / "final.pt", map_location="cpu", weights_only=False)
        self.assertEqual(resumed_payload["step"], 6)
        for key, value in live_payload["state_dict"].items():
            self.assertTrue(torch.equal(value, resumed_payload["state_dict"][key]), key)
        self.assertEqual(resumed_payload["best"], live_payload["best"])
        # the resumed run writes the current schema again
        self.assertEqual(resumed_payload["schema_version"], DECODER_SCHEMA_VERSION)

    def test_conv_state_dict_names_and_parameter_count_are_unchanged(self):
        decoder = build_decoder(DecoderConfig(), patches=256, d_world=256)
        self.assertEqual(decoder.parameter_count(), 1_549_699)
        self.assertEqual(build_decoder(DecoderConfig(), patches=256, d_world=512).parameter_count(),
                         1_910_147)
        keys = list(decoder.state_dict())
        self.assertEqual(keys[:2], ["position", "stem.0.weight"])
        self.assertIn("stages.2.1.weight", keys)
        self.assertTrue(keys[-1].startswith("head."))
        self.assertEqual([tuple(decoder.state_dict()["position"].shape)], [(1, 256, 16, 16)])


class CustomPipelineTests(unittest.TestCase):
    """A non-conv component through the shared training/eval/save/load/render path."""

    def setUp(self):
        self.effective: dict = {}

    def train(self, name: str, steps: int, resume="", config=None, image_size=48,
              hidden=32) -> Path:
        config = config or custom_decoder_config(_WORKSPACE["names"][:2], _WORKSPACE["names"][2:],
                                                 image_size=image_size, hidden=hidden)
        config.data.video_dir = str(_WORKSPACE["video_dir"])
        config.data.cache_dir = str(_WORKSPACE["cache_dir"])
        config.decoder_train.max_steps = steps
        out = _WORKSPACE["root"] / name
        run_decoder_training(config, _WORKSPACE["world_checkpoint"], out, torch.device("cpu"),
                             resume=resume)
        self.effective[name] = config          # data splits are filled in by the run itself
        return out

    def test_training_updates_the_custom_component_and_reports_it(self):
        out = self.train("custom", 20)
        summary = json.loads((out / "train_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["decoder"]["kind"], CUSTOM_KIND)
        self.assertEqual(summary["decoder"]["architecture"],
                         {"kind": CUSTOM_KIND, "image_size": 48, "options": {"hidden": 32}})
        self.assertEqual(summary["decoder"]["image_size"], 48)
        self.assertEqual(summary["decoder"]["grid"], [4, 4])
        self.assertGreater(summary["decoder"]["parameters"], 0)
        # the shared summary describes the component, not the conv architecture's fields
        self.assertNotIn("base_channels", summary["decoder"])
        self.assertNotIn("channel_multipliers", summary["decoder"])
        self.assertEqual(summary["decoder_config"]["kind"], CUSTOM_KIND)
        for key in ("l1", "mse", "psnr_db", "per_video"):
            self.assertIn(key, summary["final_val"])
        self.assertTrue(all(summary["final_val"][key] > 0 for key in ("l1", "mse")))

        initial, initial_payload = load_decoder(out / "initial.pt")
        final, final_payload = load_decoder(out / "final.pt")
        self.assertIsInstance(final, example.PixelShuffleDecoder)
        self.assertEqual(final_payload["step"], 20)
        self.assertEqual(final_payload["architecture"], summary["decoder"]["architecture"])
        self.assertEqual(final_payload["decoder_config"]["options"], {"hidden": 32})
        moved = sum(1 for key, value in final.state_dict().items()
                    if not torch.equal(value, initial_payload["state_dict"][key]))
        self.assertGreater(moved, 2)                    # every weight actually moved
        # the frozen world model is untouched and still matches the recorded identity
        world, _ = VideoWorldModel.from_checkpoint(_WORKSPACE["world_checkpoint"])
        check_decoder_compatibility(final_payload, world, self.effective["custom"], final)

    def test_the_custom_component_renders_and_labels_itself(self):
        checkpoint = self.train("custom_render", 3) / "final.pt"
        decoder, payload = load_decoder(checkpoint)
        latents = {"h1": (torch.randn(16, 32, generator=torch.Generator().manual_seed(3)),
                          {"horizon_chunks": 1})}
        with tempfile.TemporaryDirectory() as temporary:
            artifact = render_latents(decoder, latents, Path(temporary), torch.device("cpu"),
                                      payload=payload, prefix="decoded")
            self.assertEqual(tuple(artifact["frames"]["h1"].shape), (3, 48, 48))
            self.assertEqual(artifact["decoder"]["component_kind"], CUSTOM_KIND)
            self.assertEqual(artifact["decoder"]["image_size"], 48)
            import cv2
            png = cv2.imread(str(Path(temporary) / artifact["records"]["h1"]["png"]))
            self.assertEqual(png.shape, (48, 48, 3))

    def test_resume_reproduces_an_uninterrupted_run(self):
        short = self.train("custom_short", 3)
        live = self.train("custom_live", 6)
        resumed = self.train("custom_resumed", 6, resume=str(short / "final.pt"))
        live_payload = torch.load(live / "final.pt", map_location="cpu", weights_only=False)
        resumed_payload = torch.load(resumed / "final.pt", map_location="cpu", weights_only=False)
        self.assertEqual(resumed_payload["step"], 6)
        for key, value in live_payload["state_dict"].items():
            self.assertTrue(torch.equal(value, resumed_payload["state_dict"][key]), key)
        self.assertEqual(resumed_payload["best"], live_payload["best"])

    def test_resume_rejects_a_changed_component_before_training(self):
        short = self.train("custom_guard", 2)
        checkpoint = str(short / "final.pt")
        cases = [
            ("custom_guard_hidden", dict(hidden=64), "does not match the decoder being resumed"),
            ("custom_guard_kind", dict(kind="conv"), "does not match the decoder being resumed"),
            ("custom_guard_size", dict(image_size=64), "does not match the decoder being resumed"),
        ]
        for name, overrides, expected in cases:
            config = custom_decoder_config(_WORKSPACE["names"][:2], _WORKSPACE["names"][2:],
                                           hidden=overrides.get("hidden", 32),
                                           image_size=overrides.get("image_size", 48))
            if "kind" in overrides:
                config.decoder = DecoderConfig(image_size=48, base_channels=16,
                                               channel_multipliers=[1, 2], stem_blocks=1)
            config.data.video_dir = str(_WORKSPACE["video_dir"])
            config.data.cache_dir = str(_WORKSPACE["cache_dir"])
            out = _WORKSPACE["root"] / name
            with self.assertRaises(ValueError, msg=name) as caught:
                run_decoder_training(config, _WORKSPACE["world_checkpoint"], out,
                                     torch.device("cpu"), resume=checkpoint)
            self.assertIn(expected, str(caught.exception))
            self.assertFalse(out.exists())            # nothing was trained or written

    def test_the_power_of_two_factor_is_a_conv_constraint_only(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            build_decoder(DecoderConfig(image_size=48, base_channels=8, stem_blocks=1), 16, 32)
        decoder = build_decoder(DecoderConfig(kind=CUSTOM_KIND, image_size=48,
                                              options={"hidden": 8}), 16, 32)
        images = decoder(torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(4)))
        self.assertEqual(tuple(images.shape), (2, 3, 48, 48))

    def test_unknown_kind_fails_before_training_and_writes_nothing(self):
        config = custom_decoder_config(_WORKSPACE["names"][:2], _WORKSPACE["names"][2:])
        config.data.video_dir = str(_WORKSPACE["video_dir"])
        config.data.cache_dir = str(_WORKSPACE["cache_dir"])
        config.decoder = DecoderConfig(kind="ghost_arch", image_size=48)
        out = _WORKSPACE["root"] / "ghost_run"
        with self.assertRaisesRegex(DecoderRegistrationError, "not registered"):
            run_decoder_training(config, _WORKSPACE["world_checkpoint"], out, torch.device("cpu"))
        self.assertFalse(out.exists())


class ConvArchitectureContractTests(unittest.TestCase):
    """The built-in conv network keeps the v0.4.0 architecture and import surface.

    Parameter counts, module names and tensor shapes are architecture facts and are
    checked here. Exact bit-for-bit equality with the released wheel is verified
    outside the package (against an installed 0.4.0 and its trained checkpoints),
    because seeded digests and float sums depend on the torch version and on the
    thread count.
    """

    def test_seeded_conv_keeps_its_parameter_names_shapes_and_budget(self):
        torch.manual_seed(1234)
        decoder = build_decoder(DecoderConfig(image_size=64, base_channels=8,
                                              channel_multipliers=[1, 2], stem_blocks=1),
                                patches=16, d_world=32)
        self.assertEqual(decoder.parameter_count(), 9779)
        self.assertEqual(decoder.stages_count, 4)              # 64px / 4 grid side = 2^4
        keys = list(decoder.state_dict())
        self.assertEqual(keys[:2], ["position", "stem.0.weight"])
        self.assertEqual([key for key in keys if key.startswith("head")], ["head.weight",
                                                                           "head.bias"])
        shapes = {key: tuple(value.shape) for key, value in decoder.state_dict().items()}
        self.assertEqual(shapes["position"], (1, 32, 4, 4))
        self.assertEqual(shapes["stem.0.weight"], (8, 32, 3, 3))
        self.assertEqual(shapes["stages.0.1.weight"], (8, 8, 3, 3))     # first stage at base width
        self.assertEqual(shapes["stages.1.1.weight"], (16, 8, 3, 3))    # then the multiplier
        self.assertEqual(shapes["head.weight"], (3, 16, 3, 3))
        latents = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            images = decoder(latents)
        self.assertEqual(tuple(images.shape), (2, 3, 64, 64))

    def test_the_conv_class_is_the_same_object_through_every_import_path(self):
        from wpm_video import LatentRGBDecoder as top_level
        from wpm_video.decoder import LatentRGBDecoder as package_level
        from wpm_video.decoder.architectures.conv import LatentRGBDecoder as architecture_level
        from wpm_video.decoder.model import LatentRGBDecoder as facade_level
        self.assertIs(top_level, facade_level)
        self.assertIs(package_level, facade_level)
        self.assertIs(architecture_level, facade_level)
        decoder = build_decoder(DecoderConfig(image_size=32, base_channels=8, stem_blocks=1),
                                patches=16, d_world=32)
        self.assertIsInstance(decoder, facade_level)


if __name__ == "__main__":
    unittest.main()
