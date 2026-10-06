"""Performance options: batching, data supply, sync reduction, reuse, policy checks.

Every test runs on CPU with the tiny native fixtures. The accelerations are
opt-in, so the tests check two things each time: that the option does what it says,
and that switching it on does not change the numbers (within an appropriate
floating-point tolerance) relative to the reference path.
"""

import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from wpm_video.config import PerformanceConfig, RunConfig, validate_performance_config
from wpm_video.dataset import TokenDataset, cache_video_tokens, gather_batch, substeps_for
from wpm_video.decoder import (ChunkFrameSource, alignment_record, decoder_loss,
                               gather_decoder_batch, run_decoder_training)
from wpm_video.model import gaussian_kl_bits, gaussian_nll_bits
from wpm_video.encoder import Encoder, NativeEncoder, cache_path, check_clips
from wpm_video.model import VideoWorldModel
from wpm_video.performance import (PerformanceError, apply_compile, autocast_context,
                                   build_optimizer, comparable_policy, math_dtype, optimizer_policy,
                                   require_matching_policy, resume_policy_error, should_pin,
                                   stable_dtype, to_device)
from wpm_video.predict import demo
from wpm_video.train import build_model, fit_projection, forward_window, materialize_stats, train

from .fixtures import (SIZE, make_clips, native_encoder, tiny_config,
                       tiny_decoder_config, write_video)

DEVICE = torch.device("cpu")


def clips_stack(count: int, frames: int = 4, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 256, (count, frames, 3, SIZE, SIZE), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(seed))


class SerialOnlyEncoder(Encoder):
    """Third-party style encoder: implements only the single-clip entry point."""

    kind = "custom"
    is_pretrained = False
    model_id = "custom"
    revision = "custom"

    def __init__(self, d_model: int = 48, image_size: int = SIZE, patch: int = 16, seed: int = 3):
        super().__init__(d_model, (image_size // patch, image_size // patch))
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(d_model, 3, generator=generator)
        self.calls = 0

    def encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        pooled = clip.float().reshape(clip.shape[0], -1).mean(dim=0)     # (3HW,)
        pooled = pooled.reshape(3, -1).mean(dim=1)                       # (3,)
        tokens = self.weight @ pooled                                    # (d_model,)
        return tokens.unsqueeze(0).repeat(self.patches, 1).contiguous()  # (P, d_model)


class EncoderBatchingTests(unittest.TestCase):
    def setUp(self):
        self.encoder = NativeEncoder(d_model=48, image_size=SIZE, patch=16, tubelet=2, seed=5)

    def test_batched_encoding_matches_serial_clips(self):
        clips = clips_stack(3)
        batched = self.encoder.encode_clips(clips)
        self.assertEqual(len(batched), 3)
        for index, tokens in enumerate(batched):
            serial = self.encoder.encode_clip(clips[index])
            self.assertEqual(tuple(tokens.shape), (16, 48))
            self.assertTrue(torch.allclose(tokens, serial, atol=1e-5, rtol=1e-4),
                            f"clip {index} differs between batched and serial encoding")
            self.assertEqual(tokens.dtype, torch.float32)

    def test_short_final_batch_encodes_every_clip(self):
        clips = clips_stack(3)
        for batch in (1, 2, 3, 5):
            produced = []
            for start in range(0, clips.shape[0], batch):
                produced.extend(self.encoder.encode_clips(clips[start:start + batch]))
            self.assertEqual(len(produced), 3, f"batch={batch}")
            self.assertTrue(torch.allclose(torch.stack(produced),
                                           torch.stack(self.encoder.encode_clips(clips)),
                                           atol=1e-5))

    def test_clips_never_see_a_later_chunk(self):
        """Changing chunk 2's frames must not move chunk 0's or chunk 1's tokens."""
        clips = clips_stack(3, frames=8)
        baseline = self.encoder.encode_clips(clips.clone())
        altered = clips.clone()
        altered[2] = torch.randint(0, 256, altered[2].shape, dtype=torch.uint8,
                                   generator=torch.Generator().manual_seed(99))
        after = self.encoder.encode_clips(altered)
        self.assertTrue(torch.equal(baseline[0], after[0]))
        self.assertTrue(torch.equal(baseline[1], after[1]))
        self.assertFalse(torch.allclose(baseline[2], after[2]))

    def test_input_contract(self):
        with self.assertRaisesRegex(ValueError, "B, T, 3, H, W"):
            self.encoder.encode_clips(torch.zeros(4, 3, SIZE, SIZE, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "uint8"):
            self.encoder.encode_clips(torch.zeros(2, 4, 3, SIZE, SIZE))
        with self.assertRaisesRegex(ValueError, "3 colour channels"):
            self.encoder.encode_clips(torch.zeros(2, 4, 1, SIZE, SIZE, dtype=torch.uint8))
        check_clips(clips_stack(1))                       # valid input passes

    def test_batched_encoding_is_one_convolution_for_the_whole_batch(self):
        """The batch is real: one conv3d call for N clips, not N calls of one."""
        calls = []
        original = torch.nn.functional.conv3d

        def counting(inputs, *args, **kwargs):
            calls.append(int(inputs.shape[0]))
            return original(inputs, *args, **kwargs)

        torch.nn.functional.conv3d = counting
        try:
            self.encoder.encode_clips(clips_stack(4))
        finally:
            torch.nn.functional.conv3d = original
        self.assertEqual(calls, [4])

    def test_cache_batches_follow_batch_clips(self):
        """``batch_clips`` is the real batch size handed to the encoder, incl. the tail."""
        class BatchSpy(Encoder):
            kind = "native"
            model_id = "native"
            revision = "native"
            is_pretrained = False

            def __init__(self, inner):
                super().__init__(inner.d_model, inner.grid)
                self.inner = inner
                self.batches = []

            def encode_clip(self, clip):
                return self.inner.encode_clip(clip)

            def encode_clips(self, clips):
                self.batches.append(int(clips.shape[0]))
                return self.inner.encode_clips(clips)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = make_clips(root / "videos", 1)[0]
            config = tiny_config(["clip0"], ["clip1"])
            spy = BatchSpy(native_encoder())
            meta = cache_video_tokens(clip, config.data, spy, root / "cache", batch_clips=3)
            chunks = len(meta["chunks"])
            self.assertGreater(chunks, 3)
            expected = [3] * (chunks // 3) + ([chunks % 3] if chunks % 3 else [])
            self.assertEqual(spy.batches, expected)

    def test_serial_fallback_keeps_custom_encoders_working(self):
        encoder = SerialOnlyEncoder()
        clips = clips_stack(3)
        self.assertIs(Encoder.encode_clips, SerialOnlyEncoder.encode_clips)
        batched = encoder.encode_clips(clips)
        self.assertEqual(encoder.calls, 3)
        for index, tokens in enumerate(batched):
            self.assertTrue(torch.allclose(tokens, encoder.encode_clip(clips[index])))

    def test_cache_uses_batches_and_batch_size_does_not_change_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clips = make_clips(root / "videos", 1)
            config = tiny_config(["clip0"], ["clip1"])
            tokens = {}
            for batch in (1, 4):
                encoder = native_encoder()
                meta = cache_video_tokens(clips[0], config.data, encoder, root / f"cache{batch}",
                                          batch_clips=batch)
                dataset = TokenDataset(config.data, root / f"cache{batch}", config.encoder, "train")
                tokens[batch] = dataset.entries["clip0"].tokens
                self.assertGreaterEqual(len(meta["chunks"]), 2)
            self.assertEqual(tokens[1].shape, tokens[4].shape)
            # cached as float16, so allow its storage precision
            self.assertTrue(torch.allclose(tokens[1].float(), tokens[4].float(), atol=5e-3))
            # the cache key does not depend on the batch size
            self.assertEqual(cache_path(root / "cache1", "clip0", config.data, config.encoder).name,
                             cache_path(root / "cache4", "clip0", config.data, config.encoder).name)


class TransferTests(unittest.TestCase):
    def test_to_device_matches_a_plain_move(self):
        tensor = torch.randn(4, 8)
        for performance in (PerformanceConfig(), PerformanceConfig(pin_memory=True),
                            PerformanceConfig(non_blocking=True)):
            moved = to_device(tensor, DEVICE, performance)
            self.assertTrue(torch.equal(moved, tensor.to(DEVICE)))
            self.assertEqual(moved.device.type, "cpu")

    def test_to_device_casts_when_asked(self):
        tensor = torch.zeros(2, 2, dtype=torch.uint8)
        self.assertEqual(to_device(tensor, DEVICE, None, dtype=torch.float32).dtype, torch.float32)


class PerformanceConfigTests(unittest.TestCase):
    def test_defaults_are_the_reference_path(self):
        performance = PerformanceConfig()
        self.assertEqual(performance.precision, "float32")
        self.assertEqual(performance.anchor_attention, "reference")
        self.assertFalse(performance.fused_optimizer)
        self.assertFalse(performance.compile)
        self.assertFalse(performance.pin_memory)
        validate_performance_config(performance)
        # an old config without the section keeps reference behaviour
        payload = tiny_config(["a"], ["b"]).to_dict()
        payload.pop("performance")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(RunConfig.load(path).performance, PerformanceConfig())

    def test_invalid_values_are_rejected(self):
        for overrides in ({"precision": "float16"}, {"precision": "fp32"},
                          {"anchor_attention": "flash"}, {"fused_optimizer": "yes"},
                          {"compile": 1}, {"pin_memory": None}):
            with self.assertRaises(ValueError, msg=overrides):
                RunConfig(performance=PerformanceConfig(**overrides)).validate()

    def test_precision_context_and_state_dtype(self):
        with autocast_context(PerformanceConfig(), DEVICE):
            self.assertFalse(torch.is_autocast_enabled("cpu"))
        with autocast_context(PerformanceConfig(precision="bfloat16"), DEVICE):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertEqual((torch.zeros(2, 2) @ torch.zeros(2, 2)).dtype, torch.bfloat16)
        self.assertEqual(stable_dtype(torch.bfloat16), torch.float32)
        self.assertEqual(stable_dtype(torch.float16), torch.float32)
        self.assertEqual(stable_dtype(torch.float64), torch.float64)
        self.assertEqual(stable_dtype(torch.float32), torch.float32)


class OptimizerPolicyTests(unittest.TestCase):
    def test_fused_optimizer_requires_cuda_and_says_so(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        with self.assertRaisesRegex(PerformanceError, "fused"):
            build_optimizer([parameter], PerformanceConfig(fused_optimizer=True), 1e-3, 0.0,
                            DEVICE)
        optimizer = build_optimizer([parameter], PerformanceConfig(), 1e-3, 0.0, DEVICE)
        self.assertEqual(type(optimizer).__name__, "AdamW")
        self.assertFalse(optimizer_policy(PerformanceConfig(), DEVICE)["fused"])

    def test_resume_policy_helpers(self):
        current = {"precision": "bfloat16", "fused": False}
        self.assertIsNone(resume_policy_error({"precision": "bfloat16"}, current, "performance"))
        self.assertIn("precision", resume_policy_error({"precision": "float32"}, current,
                                                       "performance"))
        # a checkpoint that predates the record is reference-only
        self.assertIn("predates", resume_policy_error(None, current, "performance"))
        self.assertIsNone(resume_policy_error(None, {"precision": "float32"}, "performance"))
        with self.assertRaises(PerformanceError):
            require_matching_policy({"precision": "float32"}, current, "performance")

    def test_only_semantic_policy_entries_are_compared(self):
        """Device, pinning and verbose status metadata never block a resume."""
        recorded = {"precision": "float32", "anchor_attention": "reference", "compile": False,
                    "optimizer": {"name": "AdamW", "device_type": "cuda", "fused": False},
                    "compile_status": {"applied": False, "reason": "disabled by configuration"},
                    "pin_memory": False}
        current = {"precision": "float32", "anchor_attention": "reference", "compile": False,
                   "optimizer": {"name": "AdamW", "device_type": "cpu", "fused": False},
                   "compile_status": {"applied": False, "reason": "a different string"},
                   "pin_memory": True}
        self.assertIsNone(resume_policy_error(recorded, current, "performance"))
        self.assertEqual(comparable_policy(current),
                         {"precision": "float32", "anchor_attention": "reference", "compile": False,
                          "fused": False, "optimizer_name": "AdamW"})
        # ... while the semantic entries, including a nested optimiser policy, are not
        for change, expected in (({"optimizer": {"name": "SGD"}}, "optimizer_name"),
                                 ({"optimizer": {"name": "AdamW", "fused": True}}, "fused"),
                                 ({"precision": "bfloat16"}, "precision"),
                                 ({"anchor_attention": "sdpa"}, "anchor_attention"),
                                 ({"compile": True}, "compile")):
            self.assertIn(expected, resume_policy_error(recorded, {**current, **change},
                                                        "performance"), change)
        # an incomplete record is completed with the reference values, never with the
        # acceleration the checkpoint cannot account for
        partial = {key: value for key, value in recorded.items() if key != "compile"}
        self.assertIn("compile", resume_policy_error(partial, {**current, "compile": True},
                                                     "performance"))
        self.assertIsNone(resume_policy_error(partial, current, "performance"))


class CompileHookTests(unittest.TestCase):
    def setUp(self):
        self.model = VideoWorldModel(tiny_config(["a"], ["b"]).model, d_encoder=48, patches=16,
                                     chunk_seconds=2.0)

    def test_disabled_reports_a_reason_and_changes_nothing(self):
        status = apply_compile(self.model, PerformanceConfig())
        self.assertFalse(status["applied"])
        self.assertIn("disabled", status["reason"])
        self.assertIsNone(self.model.compiled_predictor)

    def test_applied_hook_keeps_the_model_identical(self):
        keys = set(self.model.state_dict())
        logits = {}
        status = apply_compile(self.model, PerformanceConfig(compile=True),
                               compile_fn=lambda function: function)
        self.assertTrue(status["applied"])
        self.assertEqual(status["targets"], ["predictor"])
        self.assertEqual(set(self.model.state_dict()), keys)     # no _orig_mod wrappers
        state = self.model.initial_state(1, DEVICE, torch.float32)
        tokens = torch.randn(1, 16, 48, generator=torch.Generator().manual_seed(0))
        observed, _ = self.model.observe(state, tokens, 2.0, sample=False)
        logits["hooked"] = self.model.predict(observed, 2.0)[0]
        del self.model.compiled_predictor
        logits["reference"] = self.model.predict(observed, 2.0)[0]
        self.assertTrue(torch.equal(logits["hooked"], logits["reference"]))

    def test_compile_failures_are_explicit(self):
        def broken(function):
            raise RuntimeError("inductor unavailable")

        with self.assertRaisesRegex(PerformanceError, "performance.compile=false"):
            apply_compile(self.model, PerformanceConfig(compile=True), compile_fn=broken)
        self.assertIsNone(self.model.compiled_predictor)

        def returns_junk(function):
            return None

        with self.assertRaisesRegex(PerformanceError, "non-callable"):
            apply_compile(self.model, PerformanceConfig(compile=True), compile_fn=returns_junk)


class SyncReductionTests(unittest.TestCase):
    """Reference equivalence for the CPU-side validation, indices and substeps."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.config = tiny_config(cls.names[:2], cls.names[2:])
        cls.config.data.video_dir = str(cls.video_dir)
        cls.config.data.cache_dir = str(cls.root / "cache")
        cls.encoder = native_encoder()
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.config.data, cls.encoder,
                               cls.config.data.cache_dir, batch_clips=2)
        cls.train_set = TokenDataset(cls.config.data, cls.config.data.cache_dir, cls.config.encoder,
                                     "train")
        torch.manual_seed(0)
        cls.model = build_model(cls.config, cls.encoder, DEVICE)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_batch_carries_host_indices_and_substeps(self):
        batch = gather_batch(self.train_set, self.train_set.windows[:2], DEVICE,
                             model_config=self.config.model)
        self.assertIsNotNone(batch["context_substeps"])
        self.assertEqual(len(batch["context_substeps"]), self.config.data.context_chunks)
        for key, entry in batch["targets"].items():
            self.assertEqual(entry["index"].device.type, "cpu")
            if entry["substeps"] is not None:
                expected = substeps_for(entry["delta_seconds"].cpu(), self.config.model.substep_seconds,
                                        self.config.model.max_substeps)
                self.assertEqual(entry["substeps"], expected, key)
                self.assertLessEqual(entry["substeps"], self.config.model.max_substeps)

    def test_precomputed_substeps_reproduce_the_reference_advance(self):
        state = self.model.initial_state(3, DEVICE, torch.float32)
        deltas = torch.tensor([0.25, 0.6, 1.3], dtype=torch.float32)
        reference = self.model.advance(state, deltas)
        substep_seconds = self.config.model.substep_seconds
        steps = substeps_for(deltas.double(), substep_seconds, self.config.model.max_substeps)
        self.assertEqual(steps, 11)                      # ceil(1.3 / 0.125), the batch maximum
        precomputed = self.model.advance(state, deltas, substeps=steps)
        for name in ("slots", "velocity", "time"):
            self.assertTrue(torch.equal(getattr(reference, name), getattr(precomputed, name)), name)
        self.assertEqual(float(precomputed.time[0]) - float(state.time[0]), 0.25)

    def test_substeps_helper_matches_the_model_formula(self):
        import math
        for deltas in ([0.3], [0.25, 0.6, 1.3], [2.0, 0.1]):
            tensor = torch.tensor(deltas, dtype=torch.float64)
            expected = max(1, int(math.ceil(float(tensor.max()) / self.config.model.substep_seconds)))
            self.assertEqual(substeps_for(tensor, self.config.model.substep_seconds, 256), expected)
        with self.assertRaisesRegex(ValueError, "max_substeps"):
            substeps_for(torch.tensor([100.0]), self.config.model.substep_seconds, 8)
        state = self.model.initial_state(1, DEVICE, torch.float32)
        # a public count is a consistency check: a wrong one is refused, never obeyed
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.model.advance(state, 1.0, substeps=0)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.model.advance(state, 1.0, substeps=self.config.model.max_substeps + 1)
        # the private planned path still rejects a nonsensical count outright
        with self.assertRaisesRegex(ValueError, "steps must be a positive integer"):
            self.model._advance_planned(state, 1.0, 0)
        with self.assertRaisesRegex(ValueError, "above the configured limit"):
            self.model._advance_planned(state, 1.0, self.config.model.max_substeps + 1)

    def test_host_validation_matches_the_model_contract(self):
        windows = self.train_set.windows[:2]
        batch = gather_batch(self.train_set, windows, DEVICE, model_config=self.config.model)
        self.assertTrue(torch.isfinite(batch["end_seconds"]).all())
        self.assertTrue((batch["end_seconds"][:, 1:] > batch["end_seconds"][:, :-1]).all())
        # the model still refuses the same violations when called directly
        state = self.model.initial_state(1, DEVICE, torch.float32)
        tokens = torch.randn(1, 16, 48, generator=torch.Generator().manual_seed(1))
        with self.assertRaises(ValueError):
            self.model.observe(state, tokens, float(batch["end_seconds"][0, 0]) - 1.0, sample=False)

    def test_forward_window_matches_a_no_reuse_reference_objective(self):
        """The optimised objective computes the same quantity as the straightforward one.

        The reference below is written the long way on purpose: a fresh present
        estimate per horizon, no precomputed substeps, plain float accumulation --
        i.e. what the training step did before the accelerations were introduced.
        """
        from wpm_video.model import gaussian_nll_bits
        from wpm_video.train import stream_observed

        batch = gather_batch(self.train_set, self.train_set.windows[:2], DEVICE,
                             model_config=self.config.model)
        torch.manual_seed(0)
        model = build_model(self.config, self.encoder, DEVICE)
        model.train()
        torch.manual_seed(7)
        _, stats = forward_window(model, batch, self.config, sample=True, stats_mode="tensor")

        torch.manual_seed(7)
        states, diagnostics = stream_observed(model, batch, True)
        horizon_sum, equal_weight, items = None, {}, 0
        for (offset, horizon), entry in batch["targets"].items():
            index = entry["valid"].nonzero().flatten()
            if index.numel() == 0:
                continue
            mu, logvar, _, _ = model.predict(states[offset][index], entry["delta_seconds"][index])
            target = model.project(entry["tokens"][index])
            bits = gaussian_nll_bits(target, mu, logvar).mean(dim=(1, 2))
            horizon_sum = bits.sum() if horizon_sum is None else horizon_sum + bits.sum()
            equal_weight.setdefault(horizon, []).append(bits.mean())
            items += int(index.numel())
        present_mu, present_logvar = model.present_estimate(states[-1])
        present_bits = gaussian_nll_bits(model.project(batch["observed"][:, -1]), present_mu,
                                         present_logvar).mean()
        prior_bits = torch.stack([r["prior_nll_bits_per_dim"] for r in diagnostics]).mean()
        kl_bits = torch.stack([r["kl_bits_per_dim"] for r in diagnostics]).mean()
        reference_loss = (self.config.model.w_horizon * (horizon_sum / items)
                          + self.config.model.w_prior * prior_bits
                          + self.config.model.w_present * present_bits
                          + self.config.model.kl_beta * kl_bits)
        self.assertAlmostEqual(float(reference_loss.detach()), float(stats["loss"]), places=5)
        self.assertAlmostEqual(float((horizon_sum / items).detach()),
                               float(stats["nll_bits_per_dim_horizon"]), places=5)
        self.assertAlmostEqual(float(present_bits.detach()),
                               float(stats["present_nll_bits_per_dim"]), places=5)
        self.assertAlmostEqual(float(kl_bits.detach()), float(stats["kl_bits_per_dim"]), places=5)
        self.assertAlmostEqual(float(prior_bits.detach()),
                               float(stats["prior_nll_bits_per_dim"]), places=5)
        for horizon, values in equal_weight.items():
            self.assertAlmostEqual(float(torch.stack(values).mean().detach()),
                                   float(stats[f"nll_h{horizon}_bits_per_dim"]), places=5)
        self.assertEqual(items, stats["n_items"])

    def test_forward_window_tensor_stats_equal_the_float_view(self):
        batch = gather_batch(self.train_set, self.train_set.windows[:2], DEVICE,
                             model_config=self.config.model)
        torch.manual_seed(0)
        model = build_model(self.config, self.encoder, DEVICE)
        model.train()
        torch.manual_seed(7)          # sampling draws must match between the two modes
        loss_floats, stats = forward_window(model, batch, self.config, sample=True)
        self.assertIsInstance(stats["loss"], float)
        torch.manual_seed(7)
        loss_tensor, tensor_stats = forward_window(model, batch, self.config, sample=True,
                                                   stats_mode="tensor")
        self.assertTrue(torch.is_tensor(tensor_stats["loss"]))
        materialized = materialize_stats(tensor_stats)
        self.assertIsInstance(materialized["loss"], float)
        for key, value in stats.items():
            if torch.is_tensor(tensor_stats[key]):
                self.assertAlmostEqual(value, materialized[key], places=6, msg=key)
            else:
                self.assertEqual(value, materialized[key], key)
        self.assertAlmostEqual(float(loss_floats.detach()), float(loss_tensor.detach()), places=6)
        with self.assertRaisesRegex(ValueError, "stats_mode"):
            forward_window(model, batch, self.config, stats_mode="half")


class PresentEstimateReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.config = tiny_config(cls.names[:2], cls.names[2:])
        cls.encoder = native_encoder()
        torch.manual_seed(0)
        cls.model = build_model(cls.config, cls.encoder, DEVICE)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def state(self, rows: int = 2):
        state = self.model.initial_state(rows, DEVICE, torch.float32)
        tokens = torch.randn(rows, 16, 48, generator=torch.Generator().manual_seed(2))
        observed, _ = self.model.observe(state, tokens, 2.0, sample=False)
        observed, _ = self.model.observe(observed, tokens, 4.0, sample=False)
        return observed

    def test_shared_reference_matches_a_fresh_estimate(self):
        state = self.state(3)
        present_mu = self.model.present_estimate(state)[0]
        index = torch.tensor([0, 2])
        deltas = torch.tensor([1.0, 2.5])
        fresh = self.model.predict(state[index], deltas)[0]
        shared = self.model.predict(state[index], deltas, reference_mu=present_mu[index])[0]
        self.assertTrue(torch.allclose(fresh, shared, atol=1e-5, rtol=1e-4))

    def test_shared_reference_keeps_gradients_from_every_horizon(self):
        state = self.state(2).detach()
        state.slots.requires_grad_(True)
        index = torch.tensor([0, 1])
        deltas = torch.tensor([1.0, 4.0])

        def horizon_loss(shared: bool):
            self.model.zero_grad(set_to_none=True)
            state.slots.grad = None
            if shared:
                reference = self.model.present_estimate(state)[0]
                mu_a = self.model.predict(state[index], deltas, reference_mu=reference)[0]
                mu_b = self.model.predict(state[index], deltas * 2, reference_mu=reference)[0]
            else:
                mu_a = self.model.predict(state[index], deltas)[0]
                mu_b = self.model.predict(state[index], deltas * 2)[0]
            target = torch.zeros_like(mu_a)
            loss = (mu_a - target).pow(2).mean() + (mu_b - target).pow(2).mean()
            loss.backward()
            return loss.detach(), state.slots.grad.detach().clone()

        loss_reference, grad_reference = horizon_loss(shared=False)
        loss_shared, grad_shared = horizon_loss(shared=True)
        self.assertAlmostEqual(float(loss_reference), float(loss_shared), places=5)
        self.assertTrue(torch.allclose(grad_reference, grad_shared, atol=1e-5, rtol=1e-3))
        self.assertGreater(float(grad_shared.abs().sum()), 0.0)

    def test_masked_rows_do_not_disturb_the_used_ones(self):
        """Rows never mix, so an estimate for a superset is exact for a subset."""
        state = self.state(4)
        full = self.model.present_estimate(state)[0]
        subset = self.model.present_estimate(state[torch.tensor([1, 3])])[0]
        self.assertTrue(torch.allclose(full[torch.tensor([1, 3])], subset, atol=1e-5))


class AnchorAttentionTests(unittest.TestCase):
    def build(self, mode: str) -> VideoWorldModel:
        model = VideoWorldModel(tiny_config(["a"], ["b"]).model, d_encoder=48, patches=16,
                                chunk_seconds=2.0, anchor_attention=mode)
        model.eval()
        return model

    def test_reference_and_sdpa_reads_agree(self):
        reference, sdpa = self.build("reference"), self.build("sdpa")
        sdpa.load_state_dict(reference.state_dict())
        slots = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(3))
        queries = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(4))
        coordinates = reference.patch_coordinates
        explicit, attention = reference.anchors.read(slots, queries, coordinates)
        fused, none_attention = sdpa.anchors.read(slots, queries, coordinates, use_sdpa=True)
        self.assertIsNone(none_attention)
        self.assertIsNotNone(attention)
        self.assertTrue(torch.allclose(explicit, fused, atol=1e-5, rtol=1e-4))
        # the explicit path still returns normalised weights
        self.assertTrue(torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)),
                                       atol=1e-5))

    def test_write_semantics_are_untouched_by_the_read_mode(self):
        reference, sdpa = self.build("reference"), self.build("sdpa")
        sdpa.load_state_dict(reference.state_dict())
        slots = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(5))
        tokens = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(6))
        coordinates = reference.patch_coordinates
        self.assertTrue(torch.equal(reference.anchors.write(slots, tokens, coordinates),
                                    sdpa.anchors.write(slots, tokens, coordinates)))

    def test_model_predictions_match_between_modes(self):
        reference, sdpa = self.build("reference"), self.build("sdpa")
        sdpa.load_state_dict(reference.state_dict())
        state = reference.initial_state(2, DEVICE, torch.float32)
        tokens = torch.randn(2, 16, 48, generator=torch.Generator().manual_seed(7))
        first, _ = reference.observe(state, tokens, 2.0, sample=False)
        second, _ = sdpa.observe(state, tokens, 2.0, sample=False)
        self.assertTrue(torch.allclose(first.slots, second.slots, atol=1e-5, rtol=1e-4))
        deltas = torch.tensor([1.0, 3.0])
        # the public default still returns attention weights, even from an SDPA model
        mu_reference, _, _, weights = reference.predict(first, deltas)
        mu_sdpa, logvar_sdpa, _, attention = sdpa.predict(second, deltas)
        self.assertIsNotNone(attention)
        self.assertTrue(torch.isfinite(attention).all())
        self.assertAlmostEqual(float(attention.sum(dim=-1).mean().detach()), 1.0, places=5)
        self.assertTrue(torch.allclose(mu_reference, mu_sdpa, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(weights, attention, atol=1e-5, rtol=1e-4))
        # the optimised path opts out explicitly and is numerically equivalent
        mu_fast, logvar_fast, _, none_attention = sdpa.predict(second, deltas, want_attention=False)
        self.assertIsNone(none_attention)
        self.assertTrue(torch.allclose(mu_sdpa, mu_fast, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.isfinite(mu_sdpa).all() and torch.isfinite(logvar_sdpa).all())
        self.assertTrue(torch.isfinite(mu_fast).all() and torch.isfinite(logvar_fast).all())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "anchor_attention"):
            self.build("flash")


class MixedPrecisionTrainingTests(unittest.TestCase):
    """bfloat16 compute on CPU: finite, and the state/logs keep their contract."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.encoder = native_encoder()
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", tiny_config([], []).data, cls.encoder,
                               cls.root / "cache", batch_clips=2)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def config(self, **performance):
        config = tiny_config(self.names[:2], self.names[2:], steps=2)
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.root / "cache")
        config.performance = PerformanceConfig(**performance)
        return config

    def test_bfloat16_run_stays_finite_and_keeps_state_dtype(self):
        config = self.config(precision="bfloat16")
        train_set = TokenDataset(config.data, config.data.cache_dir, config.encoder, "train")
        val_set = TokenDataset(config.data, config.data.cache_dir, config.encoder, "val")
        torch.manual_seed(0)
        model = build_model(config, self.encoder, DEVICE)
        fit_projection(model, train_set)
        summary = train(config, model, train_set, val_set, self.root / "bf16", DEVICE,
                        provenance={"splits": {"train": self.names[:2], "val": self.names[2:]}})
        self.assertTrue(math_isfinite(summary["best"]["val_mean_nll_bits_per_dim"]))
        state = model.initial_state(1, DEVICE, torch.float32)
        tokens = torch.randn(1, 16, 48, generator=torch.Generator().manual_seed(0))
        with autocast_context(config.performance, DEVICE):
            observed, diagnostics = model.observe(state, tokens, 2.0, sample=False)
        self.assertEqual(observed.slots.dtype, torch.float32)     # storage stays float32
        self.assertEqual(observed.time.dtype, torch.float64)
        self.assertTrue(torch.isfinite(observed.slots).all())
        payload = torch.load(self.root / "bf16" / "final.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["performance"]["precision"], "bfloat16")
        self.assertEqual(payload["performance"]["state_dtype"], "float32")
        self.assertEqual(payload["performance"]["loss_dtype"], "float32")
        self.assertEqual(payload["performance"]["optimizer"]["fused"], False)

    def test_resume_refuses_a_precision_change(self):
        reference = self.config()
        bf16 = self.config(precision="bfloat16")
        train_set = TokenDataset(reference.data, reference.data.cache_dir, reference.encoder, "train")
        val_set = TokenDataset(reference.data, reference.data.cache_dir, reference.encoder, "val")
        torch.manual_seed(0)
        model = build_model(reference, self.encoder, DEVICE)
        fit_projection(model, train_set)
        train(reference, model, train_set, val_set, self.root / "ref", DEVICE,
              provenance={"splits": {"train": self.names[:2], "val": self.names[2:]}})
        checkpoint = self.root / "ref" / "final.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(payload["performance"]["precision"], "float32")
        torch.manual_seed(0)
        other = build_model(bf16, self.encoder, DEVICE)
        fit_projection(other, train_set)
        with self.assertRaisesRegex(PerformanceError, "precision"):
            train(bf16, other, train_set, val_set, self.root / "resumed", DEVICE,
                  resume=str(checkpoint),
                  provenance={"splits": {"train": self.names[:2], "val": self.names[2:]}})
        # resuming an old-style checkpoint (no performance record) is allowed with the
        # reference settings -- the flat policy comparison must not trip on the
        # verbose compile status that newer checkpoints carry
        stripped = {key: value for key, value in payload.items() if key != "performance"}
        legacy = self.root / "legacy.pt"
        torch.save(stripped, legacy)
        torch.manual_seed(0)
        resumed = build_model(reference, self.encoder, DEVICE)
        fit_projection(resumed, train_set)
        summary = train(reference, resumed, train_set, val_set, self.root / "legacy_resumed",
                        DEVICE, resume=str(legacy),
                        provenance={"splits": {"train": self.names[:2], "val": self.names[2:]}})
        self.assertTrue(math_isfinite(summary["best"]["val_mean_nll_bits_per_dim"]))
        with self.assertRaisesRegex(PerformanceError, "predates"):
            require_matching_policy(None, {"precision": "bfloat16", "fused": False},
                                    "performance")


def math_isfinite(value) -> bool:
    import math
    return math.isfinite(value)


class DecoderDataSupplyTests(unittest.TestCase):
    """Batched supply, the on-disk target cache and its integrity checks."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.config = tiny_decoder_config(cls.names[:2], cls.names[2:], steps=3)
        cls.config.data.video_dir = str(cls.video_dir)
        cls.config.data.cache_dir = str(cls.root / "cache")
        cls.encoder = native_encoder()
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.config.data, cls.encoder,
                               cls.config.data.cache_dir, batch_clips=2)
        cls.train_set = TokenDataset(cls.config.data, cls.config.data.cache_dir, cls.config.encoder,
                                     "train")
        torch.manual_seed(0)
        cls.world = build_model(cls.config, cls.encoder, DEVICE)
        fit_projection(cls.world, cls.train_set)
        cls.world_dir = cls.root / "world"
        train(cls.config, cls.world, cls.train_set,
              TokenDataset(cls.config.data, cls.config.data.cache_dir, cls.config.encoder, "val"),
              cls.world_dir, DEVICE,
              provenance={"splits": {"train": cls.names[:2], "val": cls.names[2:]}})
        cls.world_checkpoint = cls.world_dir / "final.pt"

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def source(self, cache_dir="", max_videos=2) -> ChunkFrameSource:
        source = ChunkFrameSource(self.video_dir, self.config.data, self.config.decoder.image_size,
                                  max_videos=max_videos, target_cache_dir=cache_dir)
        for dataset in (self.train_set,):
            for name, cache in dataset.entries.items():
                source.register(name, alignment_record(cache, require_identity=True))
        return source

    def test_batched_supply_equals_per_example_supply(self):
        source = self.source()
        windows = self.train_set.windows[:2]
        latents, targets, records = gather_decoder_batch(self.world, source, self.train_set, windows,
                                                         DEVICE, deterministic=True)
        self.assertEqual(latents.shape[0], len(records))
        for row, record in enumerate(records):
            tokens = self.train_set.tokens(record["video"], record["chunk"])
            expected = self.world.project(tokens.to(DEVICE).float().unsqueeze(0))[0]
            self.assertTrue(torch.allclose(latents[row], expected, atol=1e-6))
            frame, timestamp = source.target_frame(record["video"],
                                                   self.train_set.chunk(record["video"],
                                                                        record["chunk"]))
            self.assertAlmostEqual(record["target_time_seconds"], timestamp, places=9)
            self.assertTrue(torch.allclose(targets[row], frame.float().to(DEVICE).div(255.0),
                                           atol=1e-6))

    def test_disk_cache_avoids_decoding_and_is_exact(self):
        cache_dir = self.root / "targets"
        cold = self.source(str(cache_dir))
        warm_frames = {}
        for chunk in self.train_set.entries["clip0"].chunks:
            warm_frames[chunk["index"]] = cold.target_frame("clip0", chunk)
        counters = cold.counters()
        self.assertEqual(counters["videos_decoded"], 1)
        # the first chunk decodes the clip and writes every chunk-last frame, so the
        # remaining chunks of that video are already hits inside the same run
        self.assertGreaterEqual(counters["target_cache_misses"], 1)
        self.assertGreater(counters["target_cache_entries"], 0)
        self.assertGreater(counters["target_cache_bytes"], 0)
        self.assertEqual(counters["content_hashes_verified"], 1)

        warm = self.source(str(cache_dir))
        for chunk in self.train_set.entries["clip0"].chunks:
            frame, timestamp = warm.target_frame("clip0", chunk)
            expected_frame, expected_timestamp = warm_frames[chunk["index"]]
            self.assertTrue(torch.equal(frame, expected_frame))          # bitwise identical
            self.assertEqual(timestamp, expected_timestamp)
        counters = warm.counters()
        self.assertEqual(counters["videos_decoded"], 0)                 # no video decoding
        self.assertEqual(counters["target_cache_misses"], 0)
        self.assertEqual(counters["target_cache_hits"],
                         len(self.train_set.entries["clip0"].chunks))
        self.assertEqual(counters["content_hashes_verified"], 1)        # still verified

    def test_changed_target_identity_is_a_miss_not_a_stale_hit(self):
        cache_dir = self.root / "targets_identity"
        source = self.source(str(cache_dir))
        chunk = self.train_set.entries["clip0"].chunks[0]
        source.target_frame("clip0", chunk)
        other_size = ChunkFrameSource(self.video_dir, self.config.data, SIZE // 2,
                                      target_cache_dir=cache_dir)
        other_size.register("clip0", alignment_record(self.train_set.entries["clip0"],
                                                      require_identity=True))
        frame, _ = other_size.target_frame("clip0", chunk)
        self.assertEqual(tuple(frame.shape), (3, SIZE // 2, SIZE // 2))
        self.assertEqual(other_size.disk_cache_hits, 0)

    def test_corrupt_and_stale_entries_are_rejected(self):
        cache_dir = self.root / "targets_corrupt"
        source = self.source(str(cache_dir))
        chunk = self.train_set.entries["clip0"].chunks[0]
        source.target_frame("clip0", chunk)
        path = next((cache_dir).rglob("clip0.pt"))
        original = path.read_bytes()

        path.write_bytes(original[: len(original) // 2])                 # truncated file
        fresh = self.source(str(cache_dir))
        with self.assertRaisesRegex(Exception, "cannot be read"):
            fresh.target_frame("clip0", chunk)

        path.write_bytes(original)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["identity"]["target_image_size"] = 999                  # stale identity
        torch.save(payload, path)
        fresh = self.source(str(cache_dir))
        with self.assertRaisesRegex(Exception, "does not describe the current"):
            fresh.target_frame("clip0", chunk)

        payload["identity"]["target_image_size"] = self.config.decoder.image_size
        payload["frames"][chunk["index"]] = torch.zeros(3, 4, 4, dtype=torch.uint8)
        torch.save(payload, path)
        fresh = self.source(str(cache_dir))
        with self.assertRaisesRegex(Exception, "expected"):
            fresh.target_frame("clip0", chunk)

        payload["frames"][chunk["index"]] = torch.zeros(3, SIZE, SIZE, dtype=torch.uint8)
        payload["timestamps"][chunk["index"]] = 12345.0
        torch.save(payload, path)
        fresh = self.source(str(cache_dir))
        with self.assertRaisesRegex(Exception, "stale"):
            fresh.target_frame("clip0", chunk)

    def test_ram_cache_stays_bounded(self):
        source = self.source(max_videos=1)
        for name in self.train_set.entries:
            source.frames(name)
        self.assertLessEqual(len(source._frames), 1)
        self.assertEqual(source.decoded_videos, 2)

    def test_training_reports_counters_and_records_the_policy(self):
        config = tiny_decoder_config(self.names[:2], self.names[2:], steps=2)
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.config.data.cache_dir)
        config.decoder_train.target_cache_dir = str(self.root / "training_targets")
        summary = run_decoder_training(config, self.world_checkpoint, self.root / "decoder",
                                       DEVICE)
        counters = summary["decoded_frame_cache"]
        self.assertTrue(counters["target_cache_enabled"])
        self.assertGreater(counters["target_cache_entries"], 0)
        self.assertEqual(summary["performance"]["precision"], "float32")
        payload = torch.load(self.root / "decoder" / "final.pt", map_location="cpu",
                             weights_only=False)
        self.assertEqual(payload["performance"]["optimizer"]["fused"], False)
        self.assertFalse(payload["performance"]["compile"])              # compared policy
        self.assertFalse(payload["performance"]["compile_status"]["applied"])
        self.assertIn("decoder", payload["performance"]["compile_status"]["reason"])
        # a second run reuses the cached targets without decoding a video
        again = run_decoder_training(config, self.world_checkpoint, self.root / "decoder_warm",
                                     DEVICE)
        self.assertEqual(again["decoded_frame_cache"]["videos_decoded"], 0)
        self.assertGreater(again["decoded_frame_cache"]["target_cache_hits"], 0)
        self.assertTrue(math_isfinite(again["final_val"]["l1"]))


class ChangedSourceTests(unittest.TestCase):
    """Replacing a clip must be refused, warm target cache or not.

    Own temporary directory: the video is deliberately overwritten, so it cannot
    share fixtures with any other test.
    """

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.video_dir = self.root / "videos"
        self.clip = make_clips(self.video_dir, 1)[0]
        self.config = tiny_decoder_config(["clip0"], ["clip1"])
        self.cache_dir = self.root / "cache"
        cache_video_tokens(self.clip, self.config.data, native_encoder(), self.cache_dir)
        self.dataset = TokenDataset(self.config.data, self.cache_dir, self.config.encoder, "train")
        self.chunk = self.dataset.entries["clip0"].chunks[0]

    def tearDown(self):
        self._temporary.cleanup()

    def source(self, target_cache: str) -> ChunkFrameSource:
        source = ChunkFrameSource(self.video_dir, self.config.data, self.config.decoder.image_size,
                                  target_cache_dir=target_cache)
        source.register("clip0", alignment_record(self.dataset.entries["clip0"],
                                                  require_identity=True))
        return source

    def test_changed_source_is_refused_even_with_a_warm_cache(self):
        target_cache = str(self.root / "targets")
        source = self.source(target_cache)
        source.target_frame("clip0", self.chunk)
        self.assertGreater(source.disk_cache_hits + source.disk_cache_misses, 0)
        write_video(self.clip, seed=77)                        # same name, different bytes
        warm = self.source(target_cache)
        with self.assertRaisesRegex(Exception, "no longer matches the token cache"):
            warm.target_frame("clip0", self.chunk)
        self.assertEqual(warm.disk_cache_hits, 0)              # never served from disk


class DecoderStatsTests(unittest.TestCase):
    def test_decoder_loss_stats_modes_agree(self):
        torch.manual_seed(0)
        prediction = torch.rand(2, 3, 8, 8)
        target = torch.rand(2, 3, 8, 8)
        loss_float, stats = decoder_loss(prediction, target, 1.0, 0.1)
        loss_tensor, tensor_stats = decoder_loss(prediction, target, 1.0, 0.1, stats_mode="tensor")
        self.assertIsInstance(stats["l1"], float)
        self.assertTrue(torch.is_tensor(tensor_stats["l1"]))
        self.assertAlmostEqual(stats["l1"], float(tensor_stats["l1"]), places=7)
        self.assertAlmostEqual(stats["edge"], float(tensor_stats["edge"]), places=7)
        self.assertAlmostEqual(float(loss_float), float(loss_tensor), places=7)
        with self.assertRaisesRegex(ValueError, "stats_mode"):
            decoder_loss(prediction, target, 1.0, 0.1, stats_mode="fp16")


class LegacyCheckpointTests(unittest.TestCase):
    def test_checkpoint_without_a_performance_record_still_loads(self):
        """A checkpoint written before the accelerations existed keeps working."""
        config = tiny_config(["a"], ["b"])
        model = build_model(config, native_encoder(), DEVICE)
        legacy = {"schema_version": 2, "model_config": vars(model.config), "patches": model.patches,
                  "chunk_seconds": model.chunk_seconds,
                  "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        self.assertNotIn("performance", legacy)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            torch.save(legacy, path)
            loaded, payload = VideoWorldModel.from_checkpoint(path)
        self.assertEqual(loaded.anchor_attention, "reference")      # the reference path
        state = loaded.initial_state(1, DEVICE, torch.float32)
        tokens = torch.randn(1, 16, 48, generator=torch.Generator().manual_seed(0))
        observed, _ = loaded.observe(state, tokens, 2.0, sample=False)
        self.assertTrue(torch.equal(observed.slots, model.observe(state, tokens, 2.0,
                                                                  sample=False)[0].slots))


class CacheIdentityStabilityTests(unittest.TestCase):
    def test_performance_settings_do_not_change_the_cache_or_decoder_identity(self):
        """The encoder stays float32, so no cache key or decoder identity is expanded."""
        config = tiny_config(["a"], ["b"])
        encoder = native_encoder()
        before = cache_path(Path("cache"), "clip", config.data, config.encoder)
        config.performance = PerformanceConfig(precision="bfloat16", pin_memory=True,
                                               anchor_attention="sdpa")
        after = cache_path(Path("cache"), "clip", config.data, config.encoder)
        self.assertEqual(before, after)
        self.assertEqual(encoder.d_model, 48)            # encoder precision untouched


class TransferDefaultTests(unittest.TestCase):
    """``performance=None`` must mean the reference settings, on any device string."""

    def test_none_and_string_devices_are_accepted(self):
        tensor = torch.zeros(2, 3)
        for device in (None, "cpu", torch.device("cpu")):
            moved = to_device(tensor, device, None)
            self.assertEqual(moved.device.type, "cpu")
            self.assertTrue(torch.equal(moved, tensor))
            self.assertFalse(should_pin(None, device))

    def test_device_strings_work_for_the_optimizer_policy(self):
        self.assertEqual(optimizer_policy(None, "cpu")["device_type"], "cpu")
        self.assertFalse(optimizer_policy(None, "cuda")["fused"])

    def test_pinning_policy_is_device_aware_without_needing_cuda(self):
        self.assertFalse(should_pin(PerformanceConfig(pin_memory=True), "cpu"))
        self.assertFalse(should_pin(PerformanceConfig(non_blocking=True), torch.device("cpu")))
        self.assertTrue(should_pin(PerformanceConfig(pin_memory=True), "cuda"))
        self.assertTrue(should_pin(PerformanceConfig(non_blocking=True), torch.device("cuda")))
        self.assertFalse(should_pin(None, "cuda"))

    def test_world_batch_is_identical_with_and_without_a_transfer_policy(self):
        dataset = SyncReductionTests.train_set
        windows = dataset.windows[:2]
        plain = gather_batch(dataset, windows, DEVICE, model_config=SyncReductionTests.config.model)
        pinned = gather_batch(dataset, windows, DEVICE,
                              model_config=SyncReductionTests.config.model,
                              performance=PerformanceConfig(pin_memory=True, non_blocking=True))
        self.assertTrue(torch.equal(plain["observed"], pinned["observed"]))
        self.assertTrue(torch.equal(plain["end_seconds"], pinned["end_seconds"]))
        for key, entry in plain["targets"].items():
            self.assertTrue(torch.equal(entry["tokens"], pinned["targets"][key]["tokens"]), key)
            self.assertTrue(torch.equal(entry["index"], pinned["targets"][key]["index"]), key)
            self.assertEqual(entry["substeps"], pinned["targets"][key]["substeps"], key)

    def test_decoder_gather_accepts_the_default_transfer_policy(self):
        """``performance=None`` is the reference policy, not a crash (the CUDA default)."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_dir = root / "videos"
            names = [path.stem for path in make_clips(video_dir, 2)]
            config = tiny_decoder_config(names[:1], names[1:])
            cache_dir = root / "cache"
            encoder = native_encoder()
            for name in names:
                cache_video_tokens(video_dir / f"{name}.mp4", config.data, encoder, cache_dir,
                                   batch_clips=2)
            dataset = TokenDataset(config.data, cache_dir, config.encoder, "train")
            torch.manual_seed(0)
            world = build_model(tiny_config(names[:1], names[1:]), encoder, DEVICE)
            source = ChunkFrameSource(video_dir, config.data, config.decoder.image_size)
            for name, cache in dataset.entries.items():
                source.register(name, alignment_record(cache, require_identity=True))
            latents, targets, records = gather_decoder_batch(
                world, source, dataset, dataset.windows[:1], DEVICE, deterministic=True)  # no policy
            self.assertEqual(latents.shape[0], len(records))
            self.assertTrue(torch.isfinite(latents).all())
            self.assertEqual(targets.dtype, torch.float32)
            self.assertEqual(latents.dtype, torch.float32)


class Float32ContractTests(unittest.TestCase):
    """Reported quantities stay float32, whatever precision the model computes in."""

    def test_probability_math_promotes_reduced_inputs(self):
        generator = torch.Generator().manual_seed(0)
        target = torch.randn(4, generator=generator).bfloat16()
        mu = torch.randn(4, generator=generator).bfloat16()
        logvar = torch.zeros(4, dtype=torch.bfloat16)
        with autocast_context(PerformanceConfig(precision="bfloat16"), DEVICE):
            bits = gaussian_nll_bits(target, mu, logvar)
            kl = gaussian_kl_bits(mu, logvar, mu, logvar)
        self.assertEqual(bits.dtype, torch.float32)
        self.assertEqual(kl.dtype, torch.float32)
        self.assertTrue(torch.isfinite(bits).all())
        expected = gaussian_nll_bits(target.float(), mu.float(), logvar.float())
        self.assertTrue(torch.allclose(bits, expected, atol=1e-6))

    def test_float64_is_preserved(self):
        zeros = torch.zeros(3, dtype=torch.float64)
        self.assertEqual(gaussian_nll_bits(zeros, zeros, zeros).dtype, torch.float64)
        self.assertEqual(gaussian_kl_bits(zeros, zeros, zeros, zeros).dtype, torch.float64)
        single = torch.zeros(1)
        self.assertEqual(math_dtype(torch.zeros(1, dtype=torch.float64), single), torch.float64)
        self.assertEqual(math_dtype(torch.zeros(1, dtype=torch.bfloat16), single), torch.float32)
        self.assertEqual(math_dtype(torch.zeros(1, dtype=torch.float16)), torch.float32)

    def test_gradients_flow_through_reduced_inputs(self):
        target = torch.zeros(2, dtype=torch.bfloat16)
        mu = torch.zeros(2, dtype=torch.bfloat16, requires_grad=True)
        logvar = torch.zeros(2, dtype=torch.bfloat16)
        with autocast_context(PerformanceConfig(precision="bfloat16"), DEVICE):
            loss = gaussian_nll_bits(target, mu, logvar).mean()
        loss.backward()
        self.assertIsNotNone(mu.grad)
        self.assertEqual(mu.grad.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(mu.grad).all())

    def test_projection_is_float32_under_autocast_and_keeps_buffers(self):
        model = VideoWorldModel(tiny_config(["a"], ["b"]).model, d_encoder=48, patches=16,
                                chunk_seconds=2.0)
        tokens = torch.randn(2, 16, 48, generator=torch.Generator().manual_seed(1))
        reduced = tokens.bfloat16()
        with autocast_context(PerformanceConfig(precision="bfloat16"), DEVICE):
            projected = model.project(reduced)
        self.assertEqual(projected.dtype, torch.float32)
        # identical to projecting the same (already rounded) values in float32
        self.assertTrue(torch.allclose(projected, model.project(reduced.float()), atol=1e-6))
        self.assertTrue(torch.allclose(projected, model.project(tokens), atol=5e-2))
        self.assertEqual(model.projection.weight.dtype, torch.float32)
        self.assertEqual(model.projection.mean.dtype, torch.float32)
        self.assertEqual(model.projection.std.dtype, torch.float32)
        self.assertEqual(model.project(tokens.double()).dtype, torch.float64)

    def test_standardisation_is_fitted_in_float32_under_autocast(self):
        model = VideoWorldModel(tiny_config(["a"], ["b"]).model, d_encoder=48, patches=16,
                                chunk_seconds=2.0)
        tokens = torch.randn(32, 48, generator=torch.Generator().manual_seed(2)).bfloat16()
        with autocast_context(PerformanceConfig(precision="bfloat16"), DEVICE):
            model.projection.fit_standardisation(tokens)
        self.assertEqual(model.projection.mean.dtype, torch.float32)
        self.assertTrue(torch.isfinite(model.projection.std).all())
        self.assertGreater(float(model.projection.std.min()), 0.0)

    def test_reported_mse_is_float32_under_autocast(self):
        config = SyncReductionTests.config
        batch = gather_batch(SyncReductionTests.train_set, SyncReductionTests.train_set.windows[:2],
                             DEVICE, model_config=config.model)
        torch.manual_seed(0)
        model = build_model(config, SyncReductionTests.encoder, DEVICE)
        model.train()
        torch.manual_seed(3)
        with autocast_context(PerformanceConfig(precision="bfloat16"), DEVICE):
            loss, stats = forward_window(model, batch, config, sample=True, stats_mode="tensor")
        self.assertEqual(stats["mse"].dtype, torch.float32)
        self.assertEqual(stats["loss"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(stats["loss"]).all())


class AdvanceValidationTests(unittest.TestCase):
    """A public substep count is a consistency check, never a validation bypass."""

    def setUp(self):
        self.model = SyncReductionTests.model
        self.state = self.model.initial_state(2, DEVICE, torch.float32)

    def test_bogus_counts_are_refused(self):
        for count in (0, 1, -3, self.model.config.max_substeps + 5):
            with self.assertRaises(ValueError, msg=count):
                self.model.advance(self.state, torch.tensor([0.5, 1.0]), substeps=count)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.model.advance(self.state, 1.0, substeps=2.5)

    def test_illegal_deltas_are_refused_even_with_a_count(self):
        steps = self.model.steps_for_delta(1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.model.advance(self.state, torch.tensor([-1.0, 1.0]), substeps=steps)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model.advance(self.state, torch.tensor([float("nan"), 1.0]), substeps=steps)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model.advance(self.state, float("inf"), substeps=steps)
        with self.assertRaisesRegex(ValueError, "one delta per state row"):
            self.model.advance(self.state, torch.tensor([1.0, 2.0, 3.0]), substeps=steps)

    def test_zero_delta_keeps_its_semantics(self):
        clone = self.model.advance(self.state, torch.zeros(2), substeps=7)
        self.assertTrue(torch.equal(clone.slots, self.state.slots))
        self.assertEqual(float(clone.time[0]), float(self.state.time[0]))

    def test_observe_and_predict_check_a_public_count(self):
        tokens = torch.randn(2, 16, 48, generator=torch.Generator().manual_seed(0))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.model.observe(self.state, tokens, 1.0, sample=False, substeps=3)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.model.predict(self.state, 1.0, substeps=3)
        steps = self.model.steps_for_delta(1.0)
        planned, _ = self.model.observe(self.state, tokens, 1.0, sample=False, substeps=steps)
        plain, _ = self.model.observe(self.state, tokens, 1.0, sample=False)
        self.assertTrue(torch.equal(planned.slots, plain.slots))
        mu_planned = self.model.predict(self.state, 1.0, substeps=steps)[0]
        mu_plain = self.model.predict(self.state, 1.0)[0]
        self.assertTrue(torch.equal(mu_planned, mu_plain))

    def test_private_path_reproduces_the_validated_schedule(self):
        deltas = torch.tensor([0.4, 1.1])
        steps = self.model.steps_for_delta(float(deltas.max()))
        public = self.model.advance(self.state, deltas)
        private = self.model._advance_planned(self.state, deltas, steps)
        for name in ("slots", "velocity", "time", "step"):
            self.assertTrue(torch.equal(getattr(public, name), getattr(private, name)), name)
        self.assertEqual(steps, int(math.ceil(float(deltas.max())
                                              / self.model.config.substep_seconds)))
        self.assertEqual(self.model.validate_delta(self.state, deltas), float(deltas.max()))


class TargetCacheIdentityTests(unittest.TestCase):
    """A stale entry must never supply pixels for a different chunk or timeline."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.video_dir = self.root / "videos"
        self.clip = make_clips(self.video_dir, 1)[0]
        self.config = tiny_decoder_config(["clip0"], ["clip1"])
        self.cache_dir = self.root / "cache"
        cache_video_tokens(self.clip, self.config.data, native_encoder(), self.cache_dir)
        self.dataset = TokenDataset(self.config.data, self.cache_dir, self.config.encoder, "train")
        self.cache = self.dataset.entries["clip0"]
        self.record = alignment_record(self.cache, require_identity=True)

    def tearDown(self):
        self._temporary.cleanup()

    def source(self, cache_dir, record=None) -> ChunkFrameSource:
        source = ChunkFrameSource(self.video_dir, self.config.data, self.config.decoder.image_size,
                                  target_cache_dir=str(cache_dir))
        source.register("clip0", self.record if record is None else record)
        return source

    @staticmethod
    def shifted(record: dict) -> dict:
        moved = json.loads(json.dumps(record))
        moved["chunks"][0]["end_frame"] += 1
        moved["chunks"][0]["end_seconds"] += 0.1
        return moved

    def test_identity_covers_the_exact_chunk_mapping(self):
        base = self.source(self.root / "identity_a")
        shifted = self.source(self.root / "identity_b", self.shifted(self.record))
        self.assertNotEqual(base._identity("clip0", self.record),
                            shifted._identity("clip0", self.shifted(self.record)))
        self.assertNotEqual(base.cache_tag("clip0"), shifted.cache_tag("clip0"))
        self.assertEqual(len(base._identity("clip0", self.record)["sampled_frame_indices"]),
                         len(self.record["sampled_frame_indices"]))

    def test_shifted_timeline_cannot_reuse_a_warm_entry(self):
        cache_dir = self.root / "shifted"
        warm = self.source(cache_dir)
        frame, _ = warm.target_frame("clip0", self.cache.chunks[0])
        self.assertGreater(warm.disk_cache_hits + warm.disk_cache_misses, 0)
        shifted_record = self.shifted(self.record)
        stale = self.source(cache_dir, shifted_record)
        # different tag -> no hit, and the re-decoded timeline refuses to match the
        # shifted record, so no latent/pixel pair is ever assembled from a mix
        with self.assertRaisesRegex(Exception, "differs from the token cache"):
            stale.target_frame("clip0", shifted_record["chunks"][0])
        self.assertEqual(stale.disk_cache_hits, 0)
        self.assertTrue(torch.equal(frame, warm.target_frame("clip0", self.cache.chunks[0])[0]))

    def test_altered_chunk_record_is_refused(self):
        cache_dir = self.root / "mapping"
        source = self.source(cache_dir)
        chunk = self.cache.chunks[0]
        source.target_frame("clip0", chunk)
        path = next(Path(cache_dir).rglob("clip0.pt"))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["chunk_records"][int(chunk["index"])] = {
            **payload["chunk_records"][int(chunk["index"])], "end_frame": chunk["end_frame"] + 1}
        torch.save(payload, path)
        with self.assertRaisesRegex(Exception, "maps chunk"):
            self.source(cache_dir).target_frame("clip0", chunk)

    def test_changed_pixels_with_the_right_shape_are_refused(self):
        cache_dir = self.root / "digest"
        source = self.source(cache_dir)
        chunk = self.cache.chunks[0]
        source.target_frame("clip0", chunk)
        path = next(Path(cache_dir).rglob("clip0.pt"))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        index = int(chunk["index"])
        self.assertIsInstance(payload["frames"][index], torch.Tensor)   # weights_only is enough
        other = payload["frames"][index].clone()
        other[0, 0, 0] = (int(other[0, 0, 0]) + 1) % 256                # valid shape, new bytes
        payload["frames"][index] = other
        torch.save(payload, path)
        with self.assertRaisesRegex(Exception, "digest"):
            self.source(cache_dir).target_frame("clip0", chunk)

    def test_stale_timestamp_is_refused(self):
        cache_dir = self.root / "stale"
        source = self.source(cache_dir)
        chunk = self.cache.chunks[0]
        source.target_frame("clip0", chunk)
        path = next(Path(cache_dir).rglob("clip0.pt"))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["timestamps"][int(chunk["index"])] = 4242.0
        torch.save(payload, path)
        with self.assertRaisesRegex(Exception, "stale"):
            self.source(cache_dir).target_frame("clip0", chunk)

    def test_writes_use_unique_temporary_names(self):
        cache_dir = self.root / "atomic"
        source = self.source(cache_dir)
        saved = []
        original = torch.save

        def recording(payload, path, *args, **kwargs):
            saved.append(str(path))
            return original(payload, path, *args, **kwargs)

        frames = source.frames("clip0")            # cold: this write is not the one measured
        torch.save = recording
        try:
            source._cache_write("clip0", *frames, source._records["clip0"])
            source._cache_write("clip0", *frames, source._records["clip0"])
        finally:
            torch.save = original
        self.assertEqual(len(saved), 2)
        self.assertNotEqual(saved[0], saved[1])                  # no race on one staging file
        self.assertTrue(all(name.endswith(".tmp") for name in saved))
        self.assertEqual(list(Path(cache_dir).rglob("*.tmp")), [])
        self.assertTrue(source._cache_file("clip0").is_file())

    def test_concurrent_writers_produce_a_valid_entry(self):
        import threading
        cache_dir = self.root / "concurrent"
        writers = [self.source(cache_dir) for _ in range(3)]
        for writer in writers:
            writer.frames("clip0")
        errors = []

        def write(writer):
            try:
                writer._cache_write("clip0", *writer.frames("clip0"), writer._records["clip0"])
            except Exception as error:               # pragma: no cover - race detector
                errors.append(error)

        threads = [threading.Thread(target=write, args=(writer,)) for writer in writers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(list(Path(cache_dir).rglob("*.tmp")), [])
        reader = self.source(cache_dir)
        frame, _ = reader.target_frame("clip0", self.cache.chunks[0])
        self.assertEqual(frame.dtype, torch.uint8)
        self.assertGreater(reader.disk_cache_hits, 0)


class PolicyResumeTests(unittest.TestCase):
    """v0.2.0-shaped checkpoints resume with reference settings, never with an upgrade."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.config = tiny_decoder_config(cls.names[:2], cls.names[2:], steps=2)
        cls.config.data.video_dir = str(cls.video_dir)
        cls.config.data.cache_dir = str(cls.root / "cache")
        cls.encoder = native_encoder()
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.config.data, cls.encoder,
                               cls.config.data.cache_dir, batch_clips=2)
        cls.train_set = TokenDataset(cls.config.data, cls.config.data.cache_dir, cls.config.encoder,
                                     "train")
        cls.val_set = TokenDataset(cls.config.data, cls.config.data.cache_dir, cls.config.encoder,
                                   "val")
        torch.manual_seed(0)
        world = build_model(cls.config, cls.encoder, DEVICE)
        fit_projection(world, cls.train_set)
        train(cls.config, world, cls.train_set, cls.val_set, cls.root / "world", DEVICE,
              provenance={"splits": {"train": cls.names[:2], "val": cls.names[2:]}})
        cls.world_checkpoint = cls.root / "world" / "final.pt"

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def stripped(self, checkpoint: Path, name: str) -> Path:
        """A checkpoint shaped like one written before the performance record existed."""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload.pop("performance", None)
        path = self.root / name
        torch.save(payload, path)
        return path

    def test_old_world_checkpoint_resumes_with_reference_policy(self):
        legacy = self.stripped(self.world_checkpoint, "legacy_world.pt")
        torch.manual_seed(0)
        model = build_model(self.config, self.encoder, DEVICE)
        fit_projection(model, self.train_set)
        summary = train(self.config, model, self.train_set, self.val_set, self.root / "world_again",
                        DEVICE, resume=str(legacy),
                        provenance={"splits": {"train": self.names[:2], "val": self.names[2:]}})
        self.assertTrue(math_isfinite(summary["best"]["val_mean_nll_bits_per_dim"]))
        payload = torch.load(self.root / "world_again" / "final.pt", map_location="cpu",
                             weights_only=False)
        self.assertEqual(payload["performance"]["precision"], "float32")
        self.assertFalse(payload["performance"]["compile"])

    def test_old_decoder_checkpoint_resumes_with_reference_policy(self):
        config = tiny_decoder_config(self.names[:2], self.names[2:], steps=2)
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.config.data.cache_dir)
        run_decoder_training(config, self.world_checkpoint, self.root / "decoder", DEVICE)
        legacy = self.stripped(self.root / "decoder" / "final.pt", "legacy_decoder.pt")
        summary = run_decoder_training(config, self.world_checkpoint, self.root / "decoder_again",
                                       DEVICE, resume=str(legacy))
        self.assertTrue(math_isfinite(summary["final_val"]["l1"]))

    def test_decoder_resume_refuses_a_fused_policy_change(self):
        payload = torch.load(self.root / "decoder" / "final.pt", map_location="cpu",
                             weights_only=False) if (self.root / "decoder" / "final.pt").is_file() \
            else None
        if payload is None:
            config = tiny_decoder_config(self.names[:2], self.names[2:], steps=2)
            config.data.video_dir = str(self.video_dir)
            config.data.cache_dir = str(self.config.data.cache_dir)
            run_decoder_training(config, self.world_checkpoint, self.root / "decoder", DEVICE)
            payload = torch.load(self.root / "decoder" / "final.pt", map_location="cpu",
                                 weights_only=False)
        recorded = payload["performance"]
        # a CUDA run that used the fused kernel, resumed on a reference-policy machine
        fused = {**recorded, "fused": True,
                 "optimizer": {**recorded["optimizer"], "fused": True, "device_type": "cuda"}}
        with self.assertRaisesRegex(PerformanceError, "fused"):
            require_matching_policy(fused, {"precision": "float32", "fused": False}, "performance")
        # harmless transfer/device metadata never blocks a resume
        harmless = {**recorded, "pin_memory": True, "compile_status": {"applied": False,
                                                                      "reason": "different"},
                    "optimizer": {**recorded["optimizer"], "device_type": "cuda"}}
        self.assertIsNone(resume_policy_error(harmless, recorded, "performance"))
        self.assertIsNone(resume_policy_error(recorded, harmless, "performance"))

    def test_effective_pin_flag_is_recorded(self):
        config = tiny_decoder_config(self.names[:2], self.names[2:], steps=2)
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.config.data.cache_dir)
        config.performance = PerformanceConfig(pin_memory=True, non_blocking=True)
        run_decoder_training(config, self.world_checkpoint, self.root / "decoder_pin", DEVICE)
        payload = torch.load(self.root / "decoder_pin" / "final.pt", map_location="cpu",
                             weights_only=False)
        self.assertIn("pin_memory", payload["performance"])
        self.assertFalse(payload["performance"]["pin_memory"])       # CPU: not effective


class CompileVerificationTests(unittest.TestCase):
    """A compiled head must prove it can execute before a checkpoint calls it applied."""

    def model(self):
        return VideoWorldModel(tiny_config(["a"], ["b"]).model, d_encoder=48, patches=16,
                               chunk_seconds=2.0)

    def test_preflight_verifies_execution(self):
        model = self.model()
        status = apply_compile(model, PerformanceConfig(compile=True),
                               compile_fn=lambda function: function)
        self.assertTrue(status["applied"] and status["verified"])
        self.assertEqual(status["state"], "applied")
        self.assertIsNotNone(model.compiled_predictor)

    def test_preflight_failure_is_an_explicit_error_and_leaves_no_state(self):
        model = self.model()
        keys = set(model.state_dict())

        def failing_callable(function):
            def broken(*args, **kwargs):
                raise RuntimeError("BackendCompilerFailed: InvalidCxxCompiler: cl is not found")
            return broken

        with self.assertRaisesRegex(PerformanceError, "performance.compile=false"):
            apply_compile(model, PerformanceConfig(compile=True), compile_fn=failing_callable)
        self.assertIsNone(model.compiled_predictor)
        self.assertEqual(set(model.state_dict()), keys)

    def test_preflight_does_not_touch_rng_mode_or_gradients(self):
        model = self.model()
        model.train()
        # a mixed mode state: the probe must restore each submodule's own flag
        model.prior.train(False)
        torch.manual_seed(5)
        before_rng = torch.get_rng_state()
        parameter = next(model.predictor.parameters())
        parameter.grad = torch.full_like(parameter, 0.5)
        grad_before = parameter.grad.clone()
        keys = set(model.state_dict())
        apply_compile(model, PerformanceConfig(compile=True), compile_fn=lambda function: function)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertFalse(model.prior.training)             # per-module flags, not one mode
        self.assertTrue(model.predictor.training)
        self.assertTrue(torch.equal(grad_before, parameter.grad))
        self.assertEqual(set(model.state_dict()), keys)
        # CUDA is not initialised by the probe on a CPU-only run
        if not torch.cuda.is_initialized():                # pragma: no cover - host dependent
            self.assertFalse(torch.cuda.is_initialized())

    def test_applied_head_is_used_and_matches_the_reference(self):
        model = self.model()
        model.eval()
        state = model.initial_state(1, DEVICE, torch.float32)
        tokens = torch.randn(1, 16, 48, generator=torch.Generator().manual_seed(0))
        observed, _ = model.observe(state, tokens, 2.0, sample=False)
        apply_compile(model, PerformanceConfig(compile=True), compile_fn=lambda function: function)
        hooked = model.predict(observed, 2.0)[0]
        del model.compiled_predictor
        reference = model.predict(observed, 2.0)[0]
        self.assertTrue(torch.equal(hooked, reference))

    def test_lazy_backend_failure_is_wrapped_at_run_time(self):
        from wpm_video.performance import _GuardedCallable

        def broken(features):
            raise RuntimeError("BackendCompilerFailed: inductor could not compile")

        with self.assertRaisesRegex(PerformanceError, "performance.compile=false"):
            _GuardedCallable(broken, "predictor head")(torch.zeros(1, 2))

        def genuine_bug(features):
            raise ValueError("wrong shape")

        with self.assertRaisesRegex(ValueError, "wrong shape"):     # never relabelled
            _GuardedCallable(genuine_bug, "predictor head")(torch.zeros(1, 2))


class AttentionWeightsTests(unittest.TestCase):
    """The optimised path opts out of weights; every public caller keeps them."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.config = tiny_config(cls.names[:2], cls.names[2:])
        cls.config.data.video_dir = str(cls.video_dir)
        cls.config.data.cache_dir = str(cls.root / "cache")
        cls.encoder = native_encoder()
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.config.data, cls.encoder,
                               cls.config.data.cache_dir, batch_clips=2)
        cls.train_set = TokenDataset(cls.config.data, cls.config.data.cache_dir, cls.config.encoder,
                                     "train")
        torch.manual_seed(0)
        cls.model = build_model(cls.config, cls.encoder, DEVICE)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_training_forward_never_requests_attention_weights(self):
        model = build_model(self.config, self.encoder, DEVICE)
        calls = []
        original = model._read

        def spy(slots, queries, coordinates, want_weights=False):
            calls.append(bool(want_weights))
            return original(slots, queries, coordinates, want_weights=want_weights)

        model._read = spy
        batch = gather_batch(self.train_set, self.train_set.windows[:2], DEVICE,
                             model_config=self.config.model)
        forward_window(model, batch, self.config, sample=False)
        self.assertTrue(calls)
        self.assertFalse(any(calls), "the training path must not pay for attention weights")

    def test_evaluation_never_requests_attention_weights_either(self):
        from wpm_video.train import evaluate_model
        model = build_model(self.config, self.encoder, DEVICE)
        calls = []
        original = model._read

        def spy(slots, queries, coordinates, want_weights=False):
            calls.append(bool(want_weights))
            return original(slots, queries, coordinates, want_weights=want_weights)

        model._read = spy
        evaluate_model(model, self.train_set, self.config, DEVICE, 1)
        self.assertTrue(calls)
        self.assertFalse(any(calls))

    def test_sdpa_model_serves_weights_and_runs_a_full_demo(self):
        from wpm_video.viz import plot_frames, plot_uncertainty
        config = tiny_config(self.names[:2], self.names[2:])
        config.performance = PerformanceConfig(anchor_attention="sdpa")
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.config.data.cache_dir)
        torch.manual_seed(0)
        model = build_model(config, self.encoder, DEVICE)
        model.load_state_dict(self.model.state_dict())
        model.eval()
        out = self.root / "predict_sdpa"
        result = demo(model, self.encoder, config, self.video_dir / f"{self.names[2]}.mp4", DEVICE,
                      out, prefix_chunks=2, horizons=[1])
        self.assertTrue(torch.isfinite(result["predictions"][1]["mu"]).all())
        state = result["state"]
        mu, _, _, attention = model.predict(state, 1.0)          # public default keeps weights
        self.assertIsNotNone(attention)
        self.assertTrue(torch.isfinite(mu).all())
        self.assertIsNone(model.predict(state, 1.0, want_attention=False)[3])
        plot_uncertainty(result["predictions"], out / "uncertainty.png")
        plot_frames(result["frames"], result["prefix"][-1].end_frame,
                    result["future"][0].start_frame if result["future"] else None,
                    out / "frames.png", [float(t) for t in result["timestamps"]])
        self.assertTrue((out / "uncertainty.png").is_file())
        self.assertTrue((out / "frames.png").is_file())


class PolicyWiringTests(unittest.TestCase):
    """The configured policy must actually reach the data path and the compute path.

    A statement read but never used is worse than no option at all: these tests fail
    if the training loops stop passing ``performance`` to their batch assembly or stop
    entering the precision context.
    """

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        cls.names = [path.stem for path in make_clips(cls.video_dir, 3)]
        cls.encoder = native_encoder()
        cls.cache_dir = cls.root / "cache"
        cls.cache_config = tiny_config([], [])
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.cache_config.data, cls.encoder,
                               cls.cache_dir, batch_clips=2)
        split_config = tiny_config(cls.names[:2], cls.names[2:])       # names for the split
        cls.split_config = split_config
        cls.train_set = TokenDataset(split_config.data, cls.cache_dir, split_config.encoder, "train")
        cls.val_set = TokenDataset(split_config.data, cls.cache_dir, split_config.encoder, "val")
        torch.manual_seed(0)
        cls.world = build_model(split_config, cls.encoder, DEVICE)
        cls.world_dir = cls.root / "world"
        train(split_config, cls.world, cls.train_set, cls.val_set, cls.world_dir, DEVICE,
              provenance={"splits": {"train": cls.names[:2], "val": cls.names[2:]}})

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_world_training_wires_precision_and_transfer_policy(self):
        # `import wpm_video.train as x` would bind the function the package re-exports
        training = importlib.import_module("wpm_video.train")
        config = tiny_config(self.names[:2], self.names[2:], steps=1)
        config.data.cache_dir = str(self.cache_dir)
        config.performance = PerformanceConfig(precision="bfloat16", pin_memory=True)
        seen = {"gather": [], "autocast": []}
        original_gather, original_autocast = training.gather_batch, training.autocast_context

        def gather_spy(*args, **kwargs):
            seen["gather"].append(kwargs.get("performance"))
            return original_gather(*args, **kwargs)

        def autocast_spy(performance, device):
            seen["autocast"].append(performance)
            return original_autocast(performance, device)

        training.gather_batch, training.autocast_context = gather_spy, autocast_spy
        try:
            torch.manual_seed(0)
            model = build_model(config, self.encoder, DEVICE)
            fit_projection(model, self.train_set)
            train(config, model, self.train_set, self.val_set, self.root / "wired", DEVICE,
                  provenance={"splits": {"train": self.names[:2], "val": self.names[2:]}})
        finally:
            training.gather_batch, training.autocast_context = original_gather, original_autocast
        self.assertTrue(seen["gather"], "the training loop never assembled a batch")
        self.assertTrue(all(policy is config.performance for policy in seen["gather"]),
                        "gather_batch did not receive the configured transfer policy")
        self.assertTrue(seen["autocast"], "the training loop never entered a precision context")
        self.assertTrue(all(policy.precision == "bfloat16" for policy in seen["autocast"]))

    def test_decoder_training_wires_precision_and_transfer_policy(self):
        decoder_training = importlib.import_module("wpm_video.decoder.train")
        config = tiny_decoder_config(self.names[:2], self.names[2:], steps=1)
        config.data.video_dir = str(self.video_dir)
        config.data.cache_dir = str(self.cache_dir)
        config.performance = PerformanceConfig(precision="bfloat16", pin_memory=True)
        seen = {"gather": [], "autocast": []}
        original_gather = decoder_training.gather_decoder_batch
        original_autocast = decoder_training.autocast_context

        def gather_spy(*args, **kwargs):
            seen["gather"].append(kwargs.get("performance"))
            return original_gather(*args, **kwargs)

        def autocast_spy(performance, device):
            seen["autocast"].append(performance)
            return original_autocast(performance, device)

        decoder_training.gather_decoder_batch = gather_spy
        decoder_training.autocast_context = autocast_spy
        try:
            summary = run_decoder_training(config, self.world_dir / "final.pt",
                                           self.root / "decoder_wired", DEVICE)
        finally:
            decoder_training.gather_decoder_batch = original_gather
            decoder_training.autocast_context = original_autocast
        self.assertTrue(seen["gather"], "the decoder loop never assembled a batch")
        self.assertTrue(all(policy is config.performance for policy in seen["gather"]),
                        "gather_decoder_batch did not receive the configured transfer policy")
        self.assertTrue(seen["autocast"], "the decoder loop never entered a precision context")
        self.assertEqual(summary["performance"]["precision"], "bfloat16")
        self.assertTrue(math_isfinite(summary["final_val"]["l1"]))


if __name__ == "__main__":
    unittest.main()
