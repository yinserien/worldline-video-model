"""Optional RGB decoder: shapes, learning, compatibility, alignment and CLI wiring.

Everything runs on the native encoder and tiny synthetic clips, so the suite needs
no downloads, no GPU and no pretrained weights. Each test builds what it needs
through a memoised helper rather than depending on another test having run first.
"""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from wpm_video.cli import main
from wpm_video.config import DecoderConfig, RunConfig
from wpm_video.data import build_chunks, probe_video, read_frames
from wpm_video.dataset import TokenDataset, cache_video_tokens
from wpm_video.decoder import (ChunkFrameSource, DecoderCompatibilityError, alignment_record,
                               build_decoder, check_decoder_compatibility,
                               check_world_preprocessing, decode_latents, decoder_fingerprint,
                               evaluate_decoder, load_decoder, projection_fingerprint,
                               render_latents, run_decoder_training, train_mean_frame)
from wpm_video.decoder.model import DECODER_SCHEMA_VERSION
from wpm_video.decoder.targets import CacheAlignmentError
from wpm_video.model import VideoWorldModel
from wpm_video.train import build_model, fit_projection, train

from .fixtures import SIZE, make_clips, native_encoder, tiny_config, tiny_decoder_config, write_video


class _DeviceSpy(torch.nn.Module):
    """Decoder wrapper that records the devices it is moved to."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.devices = []

    def to(self, *args, **kwargs):
        self.devices.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)

    def forward(self, latents):
        return self.inner(latents)


def varied_run_config(config: RunConfig, **sections) -> RunConfig:
    """A deep copy of a run config with some section fields changed.

    Round-trips through JSON and ``RunConfig.load``, the same path a real run takes,
    so it exercises the loader instead of constructing dataclasses by hand.
    ``varied_run_config(config, model={"projection_seed": 7})``.
    """
    payload = config.to_dict()
    for section, overrides in sections.items():
        payload[section] = {**payload[section], **overrides}
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return RunConfig.load(path)


class DecoderArchitectureTests(unittest.TestCase):
    def test_default_decoder_matches_the_documented_parameter_budget(self):
        """The default configuration must stay compact: ~1-5M parameters at 128px."""
        decoder = build_decoder(DecoderConfig(), patches=256, d_world=256)
        self.assertEqual(decoder.stages_count, 3)          # 16x16 grid -> 128 pixels: 2^3
        self.assertEqual(decoder.parameter_count(), 1_549_699)
        self.assertGreaterEqual(decoder.parameter_count(), 1_000_000)
        self.assertLessEqual(decoder.parameter_count(), 5_000_000)
        # the same architecture against a wider latent
        wider = build_decoder(DecoderConfig(), patches=256, d_world=512)
        self.assertEqual(wider.parameter_count(), 1_910_147)

    def test_forward_shape_range_and_dtype(self):
        decoder = build_decoder(DecoderConfig(image_size=64, base_channels=8,
                                              channel_multipliers=[1, 2], stem_blocks=1),
                                patches=16, d_world=32)
        latents = torch.randn(3, 16, 32, generator=torch.Generator().manual_seed(0))
        images = decoder(latents)
        self.assertEqual(tuple(images.shape), (3, 3, 64, 64))
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(float(images.detach().min()), 0.0)
        self.assertLessEqual(float(images.detach().max()), 1.0)
        # different latents must give different images, and batch rows must not mix
        other = decoder(torch.randn(3, 16, 32, generator=torch.Generator().manual_seed(1)))
        self.assertFalse(torch.allclose(images, other))
        first_row = decoder(latents[:1])
        self.assertTrue(torch.allclose(first_row[0], images[0], atol=1e-6))

    def test_wrong_input_shape_is_rejected_with_a_readable_message(self):
        decoder = build_decoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1),
                                patches=16, d_world=32)
        for bad in (torch.randn(2, 15, 32), torch.randn(2, 16, 31), torch.randn(16, 32),
                    torch.randn(2, 16, 32, 1)):
            with self.assertRaises(ValueError, msg=tuple(bad.shape)):
                decoder(bad)

    def test_grid_and_size_contract(self):
        """The output edge must be a whole power-of-two multiple of the patch grid."""
        small = DecoderConfig(base_channels=8)
        with self.assertRaisesRegex(ValueError, "perfect square"):
            build_decoder(small, patches=17, d_world=32)
        # 96 / 4 = 24 upsampling steps is not a power of two
        with self.assertRaisesRegex(ValueError, "power of two"):
            build_decoder(DecoderConfig(image_size=96, base_channels=8), patches=16, d_world=32)
        # 32 pixels cannot cover a 64x64 patch grid
        with self.assertRaisesRegex(ValueError, "smaller than the patch grid"):
            build_decoder(DecoderConfig(image_size=32, base_channels=8), patches=4096, d_world=32)
        # 40 pixels is not a whole number of 16-pixel patch cells
        with self.assertRaisesRegex(ValueError, "multiple of the patch grid"):
            build_decoder(DecoderConfig(image_size=40, base_channels=8), patches=256, d_world=32)

    def test_decoder_never_sees_a_world_state_or_a_module_mode(self):
        """The decoder is a pure function of the latent, in train or eval mode."""
        decoder = build_decoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1),
                                patches=16, d_world=32)
        latents = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(4))
        decoder.train()
        training = decoder(latents)
        decoder.eval()
        self.assertTrue(torch.equal(training, decoder(latents)))
        decoded = decode_latents(decoder, {"a": latents[0]}, torch.device("cpu"))
        self.assertEqual(tuple(decoded["a"].shape), (3, 64, 64))
        self.assertFalse(decoder.training)          # decode_latents restores the mode it found

    def test_decode_moves_the_decoder_to_the_requested_device(self):
        """A CPU-loaded decoder must not be paired with latents on another device."""
        inner = build_decoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1),
                              patches=16, d_world=32)
        spy = _DeviceSpy(inner)
        latents = torch.randn(16, 32, generator=torch.Generator().manual_seed(5))
        spy.train()
        images = decode_latents(spy, {"a": latents}, torch.device("cpu"))
        self.assertEqual(spy.devices, [torch.device("cpu")])
        moved = {parameter.device for parameter in spy.parameters()}
        self.assertEqual(moved, {torch.device("cpu")})
        self.assertEqual(tuple(images["a"].shape), (3, 64, 64))
        self.assertTrue(spy.training)               # the mode it was found in is restored


class DecoderConfigTests(unittest.TestCase):
    def test_old_config_without_decoder_sections_still_loads(self):
        """A config written before the decoder existed keeps its exact behaviour."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config = tiny_config(["a"], ["b"])
            payload = config.to_dict()
            for section in ("decoder", "decoder_train"):
                payload.pop(section)
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = RunConfig.load(path)
            self.assertEqual(restored.decoder, DecoderConfig())
            self.assertEqual(restored.to_dict()["data"], config.to_dict()["data"])
            restored.validate()

    def test_invalid_decoder_fields_are_rejected(self):
        cases = [
            {"image_size": 100},                      # not a multiple of 8
            {"image_size": 4},                        # too small
            {"image_size": 128.0},                    # not an int
            {"base_channels": 0},
            {"base_channels": 4096},
            {"channel_multipliers": []},
            {"channel_multipliers": [1, 0]},
            {"channel_multipliers": [1, 128]},
            {"stem_blocks": 0},
            {"blocks_per_stage": 0},
        ]
        for overrides in cases:
            config = RunConfig(decoder=DecoderConfig(**overrides))
            with self.assertRaises(ValueError, msg=overrides):
                config.validate()
            # the module is a public entry point too: it must reject the same configs
            with self.assertRaises(ValueError, msg=overrides):
                build_decoder(DecoderConfig(**overrides), patches=16, d_world=32)

    def test_invalid_decoder_training_fields_are_rejected(self):
        from wpm_video.config import DecoderTrainConfig
        cases = [
            {"max_steps": 0}, {"eval_interval": 0}, {"batch_windows": 0}, {"log_interval": 0},
            {"eval_batches": 0}, {"frame_cache_videos": 0}, {"seed": -1},
            {"learning_rate": 0.0}, {"learning_rate": float("nan")}, {"grad_clip_norm": -1.0},
            {"max_wall_seconds": 0.0}, {"weight_decay": -0.1},
            {"l1_weight": 0.0, "edge_weight": 0.0}, {"edge_weight": -0.5},
        ]
        for overrides in cases:
            config = RunConfig(decoder_train=DecoderTrainConfig(**overrides))
            with self.assertRaises(ValueError, msg=overrides):
                config.validate()
        # a pure edge objective is allowed: only the sum has to be positive
        RunConfig(decoder_train=DecoderTrainConfig(l1_weight=0.0, edge_weight=1.0)).validate()


class DecoderPipelineTests(unittest.TestCase):
    """Train a world model, then a decoder against its frozen latents."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        clips = make_clips(cls.video_dir, 3)
        cls.names = [path.stem for path in clips]
        cls.config = tiny_decoder_config(cls.names[:2], cls.names[2:], steps=80)
        cls.cache_dir = cls.root / "cache"
        cls.config.data.video_dir = str(cls.video_dir)
        cls.config.data.cache_dir = str(cls.cache_dir)
        cls.encoder = native_encoder()
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.config.data, cls.encoder,
                               cls.cache_dir, batch_clips=2)
        cls.train_set = cls.dataset("train")
        cls.val_set = cls.dataset("val")
        torch.manual_seed(0)
        cls.world = build_model(cls.config, cls.encoder, torch.device("cpu"))
        fit_projection(cls.world, cls.train_set)
        cls.world_dir = cls.root / "world"
        train(cls.config, cls.world, cls.train_set, cls.val_set, cls.world_dir, torch.device("cpu"),
              provenance={"splits": {"train": cls.names[:2], "val": cls.names[2:]}})
        cls.world_checkpoint = cls.world_dir / "final.pt"
        cls._trained = {}

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    @classmethod
    def dataset(cls, split: str) -> TokenDataset:
        return TokenDataset(cls.config.data, cls.cache_dir, cls.config.encoder, split)

    @classmethod
    def trained(cls, name: str = "decoder") -> Path:
        """Run decoder training once per directory and return its checkpoint."""
        out = cls.root / name
        if name not in cls._trained:
            cls._trained[name] = run_decoder_training(cls.config, cls.world_checkpoint, out,
                                                      torch.device("cpu"))
        return out / "final.pt"

    def frame_source(self, image_size=None) -> ChunkFrameSource:
        source = ChunkFrameSource(self.video_dir, self.config.data,
                                  image_size or self.config.decoder.image_size)
        for name, cache in {**self.train_set.entries, **self.val_set.entries}.items():
            source.register(name, alignment_record(cache, require_identity=True))
        return source

    def test_heldout_reconstruction_improves_and_checkpoints_are_written(self):
        checkpoint = self.trained("decoder")
        out = checkpoint.parent
        for name in ("initial.pt", "best.pt", "final.pt", "train_summary.json", "train_log.jsonl"):
            self.assertTrue((out / name).is_file(), name)
        summary = json.loads((out / "train_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["steps"], self.config.decoder_train.max_steps)
        metrics = summary["final_val"]
        for key in ("l1", "mse", "psnr_db", "per_video", "constant_frame_reference"):
            self.assertIn(key, metrics)
        self.assertTrue(all(metrics[key] > 0 for key in ("l1", "mse")))
        self.assertEqual(set(metrics["per_video"]), {self.names[2]})
        # the untrained decoder predicts mid grey; a trained one must beat it clearly
        initial_decoder, initial_payload = load_decoder(out / "initial.pt")
        source = self.frame_source()
        world, _ = VideoWorldModel.from_checkpoint(self.world_checkpoint)
        before = evaluate_decoder(world, initial_decoder, self.val_set, source, self.config,
                                  torch.device("cpu"), 2,
                                  reference=train_mean_frame(source, self.train_set,
                                                             torch.device("cpu")))
        self.assertLess(metrics["l1"], before["l1"] * 0.9)
        self.assertLess(metrics["l1"], metrics["constant_frame_reference"]["l1"])
        self.assertGreater(metrics["psnr_db"], before["psnr_db"])
        final_decoder, final_payload = load_decoder(checkpoint)
        moved = sum(1 for key, value in final_decoder.state_dict().items()
                    if not torch.equal(value, initial_payload["state_dict"][key]))
        self.assertGreater(moved, 5)
        # the frozen world model and its projection are untouched, and recorded
        self.assertEqual(final_payload["step"], self.config.decoder_train.max_steps)
        self.assertEqual(final_payload["metrics_step"], self.config.decoder_train.max_steps)
        self.assertEqual(final_payload["world_projection"], projection_fingerprint(world))
        self.assertEqual(final_payload["frame_target"]["semantics"],
                         initial_payload["frame_target"]["semantics"])
        self.assertEqual(final_payload["schema_version"], DECODER_SCHEMA_VERSION)
        self.assertEqual(final_payload["d_world"], self.config.model.d_world)
        self.assertEqual(final_payload["architecture"]["image_size"], SIZE)

    def test_wall_budget_stop_before_the_first_evaluation_still_writes_best(self):
        """A run stopped before its scheduled evaluation must still score and keep best.

        The wall budget is tiny and ``eval_interval`` never fires on schedule, so the
        only evaluation is the one after the stop. How many steps fit in the budget is
        not asserted (the clock's resolution decides), only that the stop happened
        early and that ``best``/``best.pt`` describe those exact final weights.
        """
        config = tiny_decoder_config(self.names[:2], self.names[2:], steps=50)
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.cache_dir)
        config.decoder_train.eval_interval = 10_000        # never fires on schedule
        config.decoder_train.max_wall_seconds = 1e-9       # stops almost immediately
        out = self.root / "wall_budget"
        summary = run_decoder_training(config, self.world_checkpoint, out, torch.device("cpu"))
        steps = summary["steps"]
        self.assertLess(steps, config.decoder_train.max_steps)
        final = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
        self.assertTrue((out / "best.pt").is_file())
        payload = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["step"], steps)              # best is the final weights
        self.assertEqual(payload["metrics_step"], steps)
        self.assertEqual(payload["best"]["step"], steps)
        self.assertEqual(payload["best"]["val_l1"], summary["best"]["val_l1"])
        self.assertEqual(summary["best"]["val_l1"], summary["final_val"]["l1"])
        self.assertLess(summary["best"]["val_l1"], float("inf"))
        # the metrics belong to the weights in final.pt, not to an earlier checkpoint
        self.assertEqual(final["metrics"], payload["metrics"])
        evaluations = [entry for entry in final["history"] if "val" in entry]
        self.assertEqual(len(evaluations), 1)                 # no scheduled evaluation ran
        self.assertEqual(evaluations[0]["reason"], "final checkpoint evaluation")
        self.assertEqual(evaluations[0]["step"], steps)
        self.assertEqual([entry for entry in final["history"] if entry.get("stopped")],
                         [{"step": steps, "stopped": "wall_budget"}])

    def test_checkpoints_are_rejected_for_a_different_world_projection(self):
        decoder, payload = load_decoder(self.trained("decoder"))
        world, _ = VideoWorldModel.from_checkpoint(self.world_checkpoint)
        check_decoder_compatibility(payload, world, self.config, decoder)   # the matching case
        other = build_model(varied_run_config(self.config, model={"projection_seed": 7}),
                            self.encoder, torch.device("cpu"))
        other.projection.fit_standardisation(torch.randn(64, 48))
        with self.assertRaisesRegex(DecoderCompatibilityError, "projection"):
            check_decoder_compatibility(payload, other, self.config, decoder)
        # a module whose output size disagrees with the checkpoint is refused too
        smaller = build_decoder(DecoderConfig(image_size=32, base_channels=8, stem_blocks=1), 16, 32)
        with self.assertRaisesRegex(DecoderCompatibilityError, "architecture differs"):
            check_decoder_compatibility(payload, world, self.config, smaller)

    def test_loaded_checkpoint_records_its_absolute_path(self):
        """Artifacts must name the decoder that produced them, not null."""
        checkpoint = self.trained("decoder")
        decoder, payload = load_decoder(checkpoint)
        self.assertTrue(Path(payload["path"]).is_absolute())
        self.assertEqual(Path(payload["path"]), checkpoint.resolve())
        self.assertEqual(decoder_fingerprint(payload)["checkpoint"], checkpoint.name)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = render_latents(
                decoder, {"h1": (torch.randn(16, 32, generator=torch.Generator().manual_seed(9)),
                                 {"horizon_chunks": 1})},
                Path(temporary), torch.device("cpu"), payload=payload, prefix="decoded")
            self.assertEqual(artifact["decoder_checkpoint"], str(checkpoint.resolve()))
            self.assertTrue(Path(payload["path"]).is_file())

    def test_context_chunks_is_not_part_of_the_token_space(self):
        """A different observation window is not a different decoder input space."""
        decoder, payload = load_decoder(self.trained("decoder"))
        world, _ = VideoWorldModel.from_checkpoint(self.world_checkpoint)
        check_decoder_compatibility(payload, world, self.config, decoder)
        wider = varied_run_config(self.config, data={"context_chunks": 5})
        self.assertNotEqual(wider.data.context_chunks, self.config.data.context_chunks)
        check_decoder_compatibility(payload, world, wider, decoder)     # still accepted
        # ... but the fields that really define the token space stay enforced
        for overrides, expected in (({"fps": 12.0}, "fps"),
                                    ({"chunk_frames": 2}, "chunk_frames"),
                                    ({"image_size": 128}, "image_size")):
            altered = varied_run_config(self.config, data=overrides)
            with self.assertRaises(DecoderCompatibilityError, msg=expected) as caught:
                check_decoder_compatibility(payload, world, altered, decoder)
            self.assertIn(expected, str(caught.exception))
        # context_chunks is still recorded as provenance
        self.assertEqual(payload["sampling"]["context_chunks"], self.config.data.context_chunks)

    def test_native_encoder_ignores_inert_pretrained_metadata(self):
        """The native backend neutralises model_id/revision, so edits there are inert."""
        payload = torch.load(self.world_checkpoint, map_location="cpu", weights_only=False)
        altered = varied_run_config(self.config, encoder={"model_id": "some/other-model",
                                                          "revision": "deadbeef"})
        check_world_preprocessing(payload, altered, self.world_checkpoint)
        self.assertEqual(self.config.encoder.identity, ("native", "native"))
        # a genuinely different backend is still refused
        swapped = varied_run_config(self.config, encoder={"kind": "vjepa2"})
        with self.assertRaisesRegex(DecoderCompatibilityError, "encoder.kind"):
            check_world_preprocessing(payload, swapped, self.world_checkpoint)

    def test_identity_fields_are_each_checked(self):
        decoder, payload = load_decoder(self.trained("decoder"))
        world, _ = VideoWorldModel.from_checkpoint(self.world_checkpoint)
        mutations = (
            (lambda p: p["encoder"].__setitem__("revision", "deadbeef"), "encoder revision"),
            (lambda p: p["encoder"].__setitem__("d_model", 64), "encoder d_model"),
            (lambda p: p["sampling"].__setitem__("fps", 12.0), "sampling fps"),
            (lambda p: p["sampling"].__setitem__("chunk_frames", 2), "sampling chunk_frames"),
            (lambda p: p["sampling"].__setitem__("timeline_schema", "t1"), "sampling timeline_schema"),
            (lambda p: p["frame_target"].__setitem__("semantics", "anything else"), "semantics"),
            (lambda p: p["frame_target"].__setitem__("resolution", 32), "resolution"),
            (lambda p: p["world_projection"].__setitem__("mean", "0" * 64), "projection mean"),
            (lambda p: p.__setitem__("patches", 9), "patch layout"),
            (lambda p: p.__setitem__("d_world", 64), "latent width"),
            (lambda p: p.__setitem__("schema_version", 99), "schema_version"),
        )
        for mutate, expected in mutations:
            altered = json.loads(json.dumps(payload, default=str))
            mutate(altered)
            with self.assertRaises(DecoderCompatibilityError, msg=expected) as caught:
                check_decoder_compatibility(altered, world, self.config, decoder)
            self.assertIn(expected.split()[-1], str(caught.exception))
        with self.assertRaisesRegex(DecoderCompatibilityError, "rgb_decoder"):
            check_decoder_compatibility({"kind": "world_model"}, world, self.config, decoder)

    def test_incomplete_identity_is_refused_even_when_shapes_match(self):
        """An empty identity must not pass just because d_world and patches are right."""
        decoder, payload = load_decoder(self.trained("decoder"))
        world, _ = VideoWorldModel.from_checkpoint(self.world_checkpoint)
        altered = json.loads(json.dumps(payload, default=str))
        altered["encoder"] = {}
        altered["sampling"] = {}
        altered["frame_target"] = {}
        with self.assertRaisesRegex(DecoderCompatibilityError, "complete identity"):
            check_decoder_compatibility(altered, world, self.config, decoder)
        for section in ("world_projection", "encoder", "sampling", "frame_target"):
            altered = json.loads(json.dumps(payload, default=str))
            altered.pop(section)
            with self.assertRaises(DecoderCompatibilityError, msg=section):
                check_decoder_compatibility(altered, world, self.config, decoder)

    def test_world_checkpoint_config_must_match_the_caller_config(self):
        """Training a decoder requires the same token space as the world checkpoint."""
        payload = torch.load(self.world_checkpoint, map_location="cpu", weights_only=False)
        check_world_preprocessing(payload, self.config, self.world_checkpoint)
        for overrides, expected in (
            ({"encoder": {"kind": "vjepa2", "model_id": "other/model", "revision": "abc"}},
             "encoder.kind"),
            ({"data": {"fps": 12.0}}, "data.fps"),
            ({"data": {"chunk_frames": 2}}, "data.chunk_frames"),
            ({"data": {"image_size": 128}}, "data.image_size"),
        ):
            altered = varied_run_config(self.config, **overrides)
            with self.assertRaises(DecoderCompatibilityError, msg=expected) as caught:
                check_world_preprocessing(payload, altered, self.world_checkpoint)
            self.assertIn(expected, str(caught.exception))
        # a checkpoint that records no configuration cannot be trained against
        stripped = {key: value for key, value in payload.items() if key != "config"}
        stripped["provenance"] = {key: value for key, value in payload["provenance"].items()
                                  if key != "config"}
        with self.assertRaisesRegex(DecoderCompatibilityError, "does not record the configuration"):
            check_world_preprocessing(stripped, self.config, self.world_checkpoint)
        # ... and the CLI says so instead of training
        broken = self.root / "no_config_world.pt"
        torch.save(stripped, broken)
        with self.assertRaisesRegex(DecoderCompatibilityError, "does not record"):
            run_decoder_training(self.config, broken, self.root / "rejected_world",
                                 torch.device("cpu"))


class DecoderTargetAlignmentTests(unittest.TestCase):
    """The RGB target must be the chunk's last sampled frame, on the recorded timeline."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.video_dir = self.root / "videos"
        self.clips = make_clips(self.video_dir, 2)
        self.video = self.clips[0]
        self.config = tiny_decoder_config(["clip0"], ["clip1"])
        self.encoder = native_encoder()
        self.cache_dir = self.root / "cache"
        for clip in self.clips:                      # both splits, so a training path can run
            cache_video_tokens(clip, self.config.data, self.encoder, self.cache_dir)
        self.dataset = TokenDataset(self.config.data, self.cache_dir, self.config.encoder, "train")
        self.cache = self.dataset.entries["clip0"]
        self.source = ChunkFrameSource(self.video_dir, self.config.data,
                                       self.config.decoder.image_size)
        self.source.register("clip0", alignment_record(self.cache))

    def tearDown(self):
        self._temporary.cleanup()

    def test_target_is_the_last_sampled_frame_with_its_exact_timestamp(self):
        frames, timestamps = read_frames(self.video, self.config.data.fps,
                                        self.config.decoder.image_size)
        info = probe_video(self.video)
        for chunk in self.cache.chunks:
            frame, timestamp = self.source.target_frame("clip0", chunk)
            self.assertTrue(torch.equal(frame, frames[chunk["end_frame"] - 1]))
            self.assertAlmostEqual(timestamp, float(timestamps[chunk["end_frame"] - 1]), places=9)
            # exactly one source frame period before the chunk end ...
            self.assertAlmostEqual(timestamp, chunk["end_seconds"] - info.frame_period, places=9)
            # ... so the previous sampled frame is measurably earlier
            previous = float(timestamps[chunk["end_frame"] - 2])
            self.assertGreater(abs(previous - (chunk["end_seconds"] - info.frame_period)), 0.05)
        self.assertEqual(self.source.decoded_videos, 1)
        self.assertEqual(self.source.verified_hashes, 1)

    def test_chunks_come_from_the_cache_timeline_not_a_recomputation(self):
        info = probe_video(self.video)
        frames, timestamps = read_frames(self.video, self.config.data.fps, SIZE)
        rebuilt = build_chunks(info, self.config.data, timestamps)
        self.assertEqual([chunk.start_frame for chunk in rebuilt],
                         self.cache.meta["sampled_frame_indices"])

    def test_cache_payload_and_metadata_must_agree(self):
        """TokenDataset reads the payload chunk list; it must equal the recorded one."""
        record = alignment_record(self.cache)
        self.assertEqual(len(record["chunks"]), len(self.cache.chunks))
        original = self.cache.chunks
        for mutation, expected in (
            (lambda chunks: chunks.__setitem__(0, {**chunks[0], "end_frame": chunks[0]["end_frame"] + 1}),
             "end_frame"),
            (lambda chunks: chunks.__setitem__(1, {**chunks[1], "start_seconds": 99.0}),
             "start_seconds"),
            # dropping a chunk leaves more token rows than chunks, which is caught first
            (lambda chunks: chunks.pop(), "rows but the cache lists"),
        ):
            altered = [dict(chunk) for chunk in original]
            mutation(altered)
            self.cache.chunks = altered
            try:
                with self.assertRaises(CacheAlignmentError, msg=expected) as caught:
                    alignment_record(self.cache)
                self.assertIn(expected, str(caught.exception))
            finally:
                self.cache.chunks = original
        # a NaN boundary must fail the same way instead of passing every tolerance
        self.cache.meta["chunks"][0]["end_seconds"] = float("nan")
        with self.assertRaisesRegex(CacheAlignmentError, "not finite"):
            alignment_record(self.cache)

    def test_token_row_count_must_match_the_chunk_list(self):
        """A truncated token payload must fail here, not in some later sampled batch."""
        original = self.cache.tokens
        for rows in (len(original) - 1, len(original) + 1):
            self.cache.tokens = torch.zeros(rows, original.shape[1], original.shape[2])
            try:
                with self.assertRaisesRegex(CacheAlignmentError, "rows but the cache lists"):
                    alignment_record(self.cache, require_identity=True)
            finally:
                self.cache.tokens = original
        alignment_record(self.cache, require_identity=True)      # the intact cache still passes

    def test_missing_identity_is_refused_for_training(self):
        """Without a hash/chunk list there is nothing to verify pixels against."""
        self.assertTrue(alignment_record(self.cache)["sha256"])
        original_meta = self.cache.meta["sha256"]
        self.cache.meta["sha256"] = ""
        try:
            with self.assertRaisesRegex(CacheAlignmentError, "missing"):
                alignment_record(self.cache, require_identity=True)
            # ad hoc API use still works: the requirement belongs to training
            self.assertEqual(alignment_record(self.cache)["sha256"], "")
        finally:
            self.cache.meta["sha256"] = original_meta
        original_chunks = self.cache.chunks
        self.cache.chunks = []
        try:
            with self.assertRaisesRegex(CacheAlignmentError, "no chunk records"):
                alignment_record(self.cache, require_identity=True)
        finally:
            self.cache.chunks = original_chunks

    def test_truncated_cache_fails_during_training_preparation(self):
        """The same check must fire from the training path, before any gradient step."""
        import shutil
        from wpm_video.decoder import prepare_decoder_data

        cache_root = self.root / "truncated_cache"
        shutil.copytree(self.cache_dir, cache_root)
        path = next(cache_root.rglob("clip0.pt"))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["tokens"] = payload["tokens"][:-1]
        torch.save(payload, path)
        config = tiny_decoder_config(["clip0"], ["clip1"])
        config.data.cache_dir = str(cache_root)
        dataset = TokenDataset(config.data, cache_root, config.encoder, "train")
        world_payload = {"config": config.to_dict(),
                         "provenance": {"splits": {"train": ["clip0"], "val": ["clip1"]}}}
        with self.assertRaisesRegex(CacheAlignmentError, "rows but the cache lists"):
            prepare_decoder_data(config, world_payload, config.decoder.image_size,
                                 "world.pt", None)
        # the intact cache loads fine, so the failure is about the truncation only
        self.assertEqual(int(dataset.tokens("clip0", 0).shape[0]), 16)

    def test_replaced_video_is_refused(self):
        write_video(self.video, seed=99)                      # same name, different bytes
        with self.assertRaisesRegex(CacheAlignmentError, "no longer matches the token cache"):
            self.source.target_frame("clip0", self.cache.chunks[0])

    def test_cache_written_with_other_sampling_is_refused(self):
        record = alignment_record(self.cache)
        record["chunk_frames"] = 2
        with self.assertRaisesRegex(CacheAlignmentError, "chunk_frames"):
            ChunkFrameSource(self.video_dir, self.config.data, SIZE).register("clip0", record)
        record = alignment_record(self.cache)
        record["timeline_schema"] = "t1-sample-indices"
        with self.assertRaisesRegex(CacheAlignmentError, "timeline_schema"):
            ChunkFrameSource(self.video_dir, self.config.data, SIZE).register("clip0", record)
        record = alignment_record(self.cache)
        record["fps"] = float("nan")
        with self.assertRaisesRegex(CacheAlignmentError, "not finite"):
            ChunkFrameSource(self.video_dir, self.config.data, SIZE).register("clip0", record)

    def test_bounded_frame_cache_evicts_and_releases(self):
        source = ChunkFrameSource(self.video_dir, self.config.data, SIZE, max_videos=1)
        source.register("clip0", alignment_record(self.cache))
        source.frames("clip0")
        self.assertEqual(list(source._frames), ["clip0"])
        source.release("clip0")
        self.assertEqual(source.decoded_videos, 1)            # released, not re-decoded
        self.assertEqual(source.frames("clip0")[0].shape[0],
                         read_frames(self.video, self.config.data.fps, SIZE)[0].shape[0])


class DecoderResumeTests(unittest.TestCase):
    """A resumed decoder run must continue the same trajectory, like the world model."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.cache_dir = cls.root / "cache"
        cls.encoder = native_encoder()
        cls.base = tiny_decoder_config(cls.names[:2], cls.names[2:], steps=3)
        cls.base.data.video_dir = str(cls.video_dir)
        cls.base.data.cache_dir = str(cls.cache_dir)
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.base.data, cls.encoder,
                               cls.cache_dir, batch_clips=2)
        train_set = TokenDataset(cls.base.data, cls.cache_dir, cls.base.encoder, "train")
        val_set = TokenDataset(cls.base.data, cls.cache_dir, cls.base.encoder, "val")
        torch.manual_seed(0)
        world = build_model(cls.base, cls.encoder, torch.device("cpu"))
        fit_projection(world, train_set)
        train(cls.base, world, train_set, val_set, cls.root / "world", torch.device("cpu"),
              provenance={"splits": {"train": cls.names[:2], "val": cls.names[2:]}})
        cls.world_checkpoint = cls.root / "world" / "final.pt"

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def train_run(self, name: str, steps: int, resume: str = ""):
        config = tiny_decoder_config(self.names[:2], self.names[2:], steps=steps)
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.cache_dir)
        out = self.root / name
        run_decoder_training(config, self.world_checkpoint, out, torch.device("cpu"), resume=resume)
        return out

    def test_resume_reproduces_an_uninterrupted_run(self):
        short = self.train_run("short", 3)
        live = self.train_run("live", 6)
        resumed = self.train_run("resumed", 6, resume=str(short / "final.pt"))
        live_payload = torch.load(live / "final.pt", map_location="cpu", weights_only=False)
        resumed_payload = torch.load(resumed / "final.pt", map_location="cpu", weights_only=False)
        self.assertEqual(live_payload["step"], resumed_payload["step"])
        self.assertEqual(live_payload["step"], 6)
        for key, value in live_payload["state_dict"].items():
            self.assertTrue(torch.equal(value, resumed_payload["state_dict"][key]), key)
        self.assertEqual(live_payload["best"], resumed_payload["best"])

    def test_resume_rejects_a_different_architecture_or_world(self):
        short = self.train_run("short_arch", 2)
        other_world = VideoWorldModel(
            type(self.base.model)(**{**vars(self.base.model), "projection_seed": 3}),
            d_encoder=48, patches=16, chunk_seconds=2.0)
        other_path = self.root / "other_world.pt"
        other_world.save(other_path)
        with self.assertRaises(DecoderCompatibilityError):
            run_decoder_training(self.base, other_path, self.root / "rejected", torch.device("cpu"),
                                 resume=str(short / "final.pt"))
        # the checkpoint's architecture is authoritative: a config that describes a
        # different output size is refused instead of producing shape errors later
        changed = tiny_decoder_config(self.names[:2], self.names[2:], steps=4, image_size=32)
        changed.data.video_dir = str(self.video_dir)
        changed.data.cache_dir = str(self.cache_dir)
        with self.assertRaisesRegex(DecoderCompatibilityError, "architecture"):
            run_decoder_training(changed, self.world_checkpoint, self.root / "rejected_arch",
                                 torch.device("cpu"), resume=str(short / "final.pt"))


class DecoderCliTests(unittest.TestCase):
    """cache -> train -> train-decoder -> predict/query, through the real CLI."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.config_path = cls.root / "config.json"
        cls.config = tiny_decoder_config(cls.names[:2], cls.names[2:], steps=6)
        cls.config.decoder_train.max_steps = 20
        cls.config.decoder_train.eval_interval = 10
        cls.config.decoder_train.learning_rate = 2e-3
        cls.config.data.video_dir = str(cls.video_dir)
        cls.config.data.cache_dir = str(cls.root / "cache")
        cls.config.save(cls.config_path)
        cls.world_dir = cls.root / "world"
        cls.decoder_dir = cls.root / "decoder"
        cls._predictions = {}

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    # -- memoised fixtures ---------------------------------------------------
    @classmethod
    def world(cls) -> Path:
        checkpoint = cls.world_dir / "best.pt"
        if not checkpoint.is_file():
            for command in (["cache"], ["train", "--out", str(cls.world_dir)]):
                code = main([command[0], "--config", str(cls.config_path), *command[1:],
                             "--allow-native"])
                assert code == 0, command
        return checkpoint

    @classmethod
    def decoder(cls) -> Path:
        checkpoint = cls.decoder_dir / "final.pt"
        if not checkpoint.is_file():
            code = main(["train-decoder", "--config", str(cls.config_path), "--checkpoint",
                         str(cls.world()), "--out", str(cls.decoder_dir)])
            assert code == 0
        return checkpoint

    def predict(self, name: str, decoder: bool, allow_native: bool = True) -> Path:
        out = self.root / name
        if name in self._predictions:
            return self._predictions[name]
        command = ["predict", "--config", str(self.config_path), "--checkpoint", str(self.world()),
                   "--video", str(self.video_dir / f"{self.names[2]}.mp4"), "--out", str(out),
                   "--prefix-chunks", "2"]
        if allow_native:
            command.append("--allow-native")
        if decoder:
            command += ["--decoder-checkpoint", str(self.decoder())]
        self.assertEqual(main(command), 0)
        self._predictions[name] = out
        return out

    def test_train_decoder_command_writes_checkpoints_and_split_provenance(self):
        checkpoint = self.decoder()
        for name in ("initial.pt", "best.pt", "final.pt", "train_summary.json", "train_log.jsonl",
                     "config.json", "splits.json"):
            self.assertTrue((self.decoder_dir / name).is_file(), name)
        splits = json.loads((self.decoder_dir / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(splits["train"], self.names[:2])
        self.assertEqual(splits["val"], self.names[2:])
        self.assertIn("world checkpoint provenance", splits["source"])
        # the written config records what was really trained with
        written = RunConfig.load(self.decoder_dir / "config.json")
        self.assertEqual(written.data.train_videos, self.names[:2])
        self.assertEqual(written.data.val_videos, self.names[2:])
        self.assertEqual(written.decoder.image_size, SIZE)
        summary = json.loads((self.decoder_dir / "train_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["kind"], "rgb_decoder")
        self.assertEqual(summary["decoder"]["image_size"], SIZE)
        self.assertEqual(summary["decoder"]["grid"], [4, 4])
        self.assertGreater(summary["decoder"]["parameters"], 0)
        self.assertIn("constant_frame_reference", summary["final_val"])
        self.assertEqual(summary["final_val"], torch.load(checkpoint, map_location="cpu",
                                                          weights_only=False)["metrics"])
        self.assertIn("not a continuous clip", " ".join(summary["notes"]))

    def test_train_decoder_needs_split_provenance_and_a_recorded_config(self):
        # a bare `model.save` checkpoint records neither: refused with a clear reason
        bare = self.root / "bare_world.pt"
        torch.manual_seed(0)
        model = build_model(self.config, native_encoder(), torch.device("cpu"))
        model.save(bare)
        with self.assertRaisesRegex(ValueError, "does not record the configuration"):
            main(["train-decoder", "--config", str(self.config_path), "--checkpoint", str(bare),
                  "--out", str(self.root / "no_config")])
        # a checkpoint with a config but no split provenance cannot define the held-out set
        payload = torch.load(self.world(), map_location="cpu", weights_only=False)
        payload["provenance"] = {key: value for key, value in payload["provenance"].items()
                                 if key != "splits"}
        no_splits = self.root / "no_splits_world.pt"
        torch.save(payload, no_splits)
        with self.assertRaisesRegex(ValueError, "split provenance"):
            main(["train-decoder", "--config", str(self.config_path), "--checkpoint", str(no_splits),
                  "--out", str(self.root / "no_split")])

    def test_predict_with_a_decoder_writes_labelled_keyframes(self):
        out = self.predict("predict_decoded", decoder=True)
        for name in ("decoded_h1.png", "target_reconstruction_h1.png", "decoded_predictions.pt",
                     "decoded_targets.pt", "decoded_summary.json", "decoded_frames.png",
                     "future_latents.pt", "world_state.pt"):
            self.assertTrue((out / name).is_file(), name)
        summary = json.loads((out / "predict_summary.json").read_text(encoding="utf-8"))
        decoded = summary["decoded_rgb"]
        self.assertTrue(decoded["enabled"])
        self.assertEqual(decoded["image_size"], SIZE)
        entry = decoded["predicted_keyframes"]["h1"]
        self.assertEqual(entry["horizon_chunks"], 1)
        self.assertGreater(entry["target_time_seconds"], 0.0)
        self.assertEqual(entry["latent_source"], "predicted mu")
        # the pixel time is the chunk end minus one source frame, not the chunk end
        self.assertLess(entry["target_frame_timestamp_seconds"], entry["chunk_end_seconds"])
        self.assertGreater(entry["source_fps"], 0)
        reconstruction = decoded["true_target_reconstructions"]["h1"]
        self.assertEqual(reconstruction["latent_source"], "true future latent")
        import cv2
        png = cv2.imread(str(out / decoded["predicted_keyframes"]["h1"]["png"]))
        self.assertEqual(png.shape, (SIZE, SIZE, 3))
        artifact = torch.load(out / "decoded_predictions.pt", map_location="cpu", weights_only=False)
        self.assertEqual(tuple(artifact["frames"]["h1"].shape), (3, SIZE, SIZE))
        self.assertEqual(artifact["kind"], "rgb_decoder_frames")
        self.assertTrue(artifact["decoder"]["projection_sha256"])

    def test_query_keeps_float64_deltas_and_distinct_keyframes(self):
        """Two near-identical query times must stay two entries with two file names."""
        state = self.predict("predict_plain", decoder=False) / "world_state.pt"
        world, _ = VideoWorldModel.from_checkpoint(self.world())
        decoder, payload = load_decoder(self.decoder())
        from wpm_video.predict import predict_at, query_saved_state
        loaded = torch.load(state, map_location="cpu", weights_only=False)
        from wpm_video import WorldState
        world_state = WorldState(**loaded).to(device=torch.device("cpu"), dtype=torch.float32)
        close = [1.0, 1.000000001]
        outputs = predict_at(world, world_state, close)
        self.assertEqual(sorted(outputs), close)                    # not collapsed to one key
        self.assertEqual(outputs[1.000000001]["delta_seconds"], 1.000000001)
        self.assertNotEqual(outputs[1.000000001]["target_time_seconds"],
                            outputs[1.0]["target_time_seconds"])
        out = self.root / "query_close"
        query_saved_state(world, state, close, torch.device("cpu"), out, decoder=decoder,
                          decoder_payload=payload, config=self.config)
        summary = json.loads((out / "state_query.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(summary["decoded_rgb"]["keyframes"]),
                         ["d1.000000001s", "d1.0s"])           # lexicographic, both present
        for name in ("decoded_d1.0s.png", "decoded_d1.000000001s.png"):
            self.assertTrue((out / name).is_file(), name)
        entry = summary["decoded_rgb"]["keyframes"]["d1.000000001s"]
        self.assertEqual(entry["delta_seconds"], 1.000000001)

    def test_query_with_a_decoder_stays_video_and_encoder_free(self):
        """No --allow-native: the state-only query path must not build an encoder."""
        state = self.predict("predict_plain", decoder=False) / "world_state.pt"
        out = self.root / "query_decoded"
        code = main(["query", "--config", str(self.config_path), "--checkpoint", str(self.world()),
                     "--state", str(state), "--deltas", "1", "2", "--out", str(out),
                     "--decoder-checkpoint", str(self.decoder())])
        self.assertEqual(code, 0)
        for name in ("decoded_d1.0s.png", "decoded_d2.0s.png", "state_query_decoded.pt",
                     "decoded_summary.json", "state_query.json", "state_query_latents.pt"):
            self.assertTrue((out / name).is_file(), name)
        summary = json.loads((out / "state_query.json").read_text(encoding="utf-8"))
        decoded = summary["decoded_rgb"]
        self.assertTrue(decoded["enabled"])
        self.assertEqual(sorted(decoded["keyframes"]), ["d1.0s", "d2.0s"])
        entry = decoded["keyframes"]["d2.0s"]
        self.assertEqual(entry["delta_seconds"], 2.0)
        self.assertAlmostEqual(entry["target_time_seconds"],
                               summary["state_time_seconds"] + 2.0, places=4)
        # no video was read, so the frame's capture time is unknown, not invented
        self.assertIsNone(entry["target_frame_timestamp_seconds"])
        self.assertIsNone(entry["source_fps"])
        self.assertIn("unknown", entry["timestamp_basis"])
        self.assertIn("not a continuous video", decoded["note"])

    def test_predict_without_a_decoder_is_unchanged(self):
        """No decoder checkpoint: the artifacts are exactly the ones that existed before."""
        out = self.predict("predict_plain", decoder=False)
        produced = sorted(path.name for path in out.iterdir())
        self.assertEqual(produced, ["frames.png", "future_latents.pt", "latent_pca.png",
                                    "pca_meta.json", "predict_summary.json", "uncertainty.png",
                                    "world_state.pt"])
        summary = json.loads((out / "predict_summary.json").read_text(encoding="utf-8"))
        self.assertIsNone(summary["decoded_rgb"])
        pca = json.loads((out / "pca_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(pca["status"], "written")     # pre-existing behaviour: the cache is here

    def test_incompatible_decoder_fails_before_writing_any_image(self):
        """Same d_world and patch count, different projection: rejected, nothing written."""
        other = build_model(varied_run_config(self.config, model={"projection_seed": 4242}),
                            native_encoder(), torch.device("cpu"))
        other_path = self.root / "other_projection.pt"
        other.save(other_path)
        out = self.root / "predict_mismatch"
        with self.assertRaises(DecoderCompatibilityError) as caught:
            main(["predict", "--config", str(self.config_path), "--checkpoint", str(other_path),
                  "--video", str(self.video_dir / f"{self.names[2]}.mp4"), "--out", str(out),
                  "--prefix-chunks", "2", "--allow-native", "--decoder-checkpoint",
                  str(self.decoder())])
        self.assertIn("projection", str(caught.exception))
        self.assertFalse(out.exists() and any(out.iterdir()))
        with self.assertRaisesRegex(ValueError, "rgb_decoder"):
            main(["predict", "--config", str(self.config_path), "--checkpoint", str(self.world()),
                  "--video", str(self.video_dir / f"{self.names[2]}.mp4"),
                  "--out", str(self.root / "predict_wrong_kind"), "--allow-native",
                  "--decoder-checkpoint", str(self.world())])   # a world checkpoint, not a decoder

    def test_query_refuses_to_silently_ignore_a_decoder(self):
        from wpm_video.predict import query_saved_state
        state = self.predict("predict_plain", decoder=False) / "world_state.pt"
        world, _ = VideoWorldModel.from_checkpoint(self.world())
        decoder, payload = load_decoder(self.decoder())
        with self.assertRaisesRegex(ValueError, "needs out_dir"):
            query_saved_state(world, state, [1.0], torch.device("cpu"), None, decoder=decoder,
                              decoder_payload=payload, config=self.config)


class DecoderRenderingTests(unittest.TestCase):
    def test_pngs_and_tensor_artifacts_are_within_range_and_labelled(self):
        decoder = build_decoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1),
                                patches=16, d_world=32)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            artifact = render_latents(
                decoder,
                {1: (torch.randn(16, 32, generator=torch.Generator().manual_seed(2)),
                     {"horizon_chunks": 1, "delta_seconds": 1.0, "target_time_seconds": 3.0})},
                out, torch.device("cpu"), payload={"kind": "rgb_decoder"}, prefix="decoded")
            record = artifact["records"]["1"]
            self.assertEqual(record["target_time_seconds"], 3.0)
            self.assertTrue((out / record["png"]).is_file())
            image = artifact["frames"]["1"]
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
            self.assertIn("not a continuous video", artifact["note"])
            import cv2
            png = cv2.imread(str(out / record["png"]))
            self.assertEqual(png.shape, (64, 64, 3))

    def test_two_keys_cannot_overwrite_each_other(self):
        decoder = build_decoder(DecoderConfig(image_size=64, base_channels=8, stem_blocks=1),
                                patches=16, d_world=32)
        latent = torch.randn(16, 32, generator=torch.Generator().manual_seed(3))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "both map to"):
                render_latents(decoder, {"h 1": (latent, {}), "h_1": (latent, {})},
                               Path(temporary), torch.device("cpu"), payload={})


if __name__ == "__main__":
    unittest.main()
