"""Architecture contracts: spatial addressing, times, determinism, units, persistence."""

import math
from pathlib import Path
import tempfile
import unittest

import torch

from wpm_video.config import DataConfig, ModelConfig
from wpm_video.data import build_chunks, probe_video, read_frames
from wpm_video.encoder import NativeEncoder
from wpm_video.model import VideoWorldModel, gaussian_nll_bits
from wpm_video.selfcheck import CHECKS
from wpm_video.world_state import WorldState

from .fixtures import SIZE, make_clips, native_encoder, synthetic_tokens, write_video

LN2 = math.log(2.0)
PATCHES = 16


def build_fixture(seed: int = 0, slots: int = 16, d_world: int = 32):
    model = VideoWorldModel(
        ModelConfig(d_world=d_world, d_hidden=64, slots=slots, heads=4, substep_seconds=0.25,
                    max_substeps=256, anchor_bandwidth=1.0),
        d_encoder=48, patches=PATCHES, chunk_seconds=2.0,
    )
    model.eval()
    return model, native_encoder()


class ArchitectureCheckTests(unittest.TestCase):
    def test_all_self_checks_pass(self):
        for check in CHECKS:
            name, passed, detail = check()
            self.assertTrue(passed, f"{name}: {detail}")


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        self.model, self.encoder = build_fixture()

    def test_constructor_validates_before_allocating(self):
        for overrides in ({"heads": 0}, {"heads": -1}, {"slots": 10}, {"d_world": 48, "heads": 5},
                          {"substep_seconds": 0.9}, {"damping_init": 3.0}, {"a_max": float("nan")}):
            kwargs = {"d_world": 64, "slots": 16, "heads": 4, **overrides}
            with self.assertRaises(ValueError, msg=overrides):
                VideoWorldModel(ModelConfig(**kwargs), d_encoder=64, patches=16, chunk_seconds=1.0)

    def test_default_observe_follows_module_mode(self):
        state = self.model.initial_state(2, torch.device("cpu"), torch.float32)
        tokens = torch.randn(2, PATCHES, 48, generator=torch.Generator().manual_seed(1))
        self.model.eval()
        first, _ = self.model.observe(state, tokens, 2.0)
        second, _ = self.model.observe(state, tokens, 2.0)
        self.assertTrue(torch.equal(first.slots, second.slots))
        self.model.train()
        third, _ = self.model.observe(state, tokens, 2.0)
        fourth, _ = self.model.observe(state, tokens, 2.0)
        self.assertFalse(torch.equal(third.slots, fourth.slots))
        generator = torch.Generator().manual_seed(11)
        fifth, _ = self.model.observe(state, tokens, 2.0, generator=generator)
        generator = torch.Generator().manual_seed(11)
        sixth, _ = self.model.observe(state, tokens, 2.0, generator=generator)
        self.assertTrue(torch.equal(fifth.slots, sixth.slots))

    def test_batched_and_scalar_deltas(self):
        state = self.model.initial_state(3, torch.device("cpu"), torch.float32)
        batched = self.model.predict(state, torch.tensor([1.0, 2.0, 4.0]))[0]
        self.assertEqual(tuple(batched.shape), (3, PATCHES, 32))
        self.assertEqual(tuple(self.model.predict(state, 2.0)[0].shape), (3, PATCHES, 32))
        with torch.no_grad():
            self.model.predictor[-1].weight.add_(
                0.1 * torch.randn_like(self.model.predictor[-1].weight))
        conditioned = self.model.predict(state, torch.tensor([1.0, 2.0, 4.0]))[0]
        self.assertFalse(torch.allclose(conditioned[0], conditioned[2]))
        with self.assertRaises(ValueError):
            self.model.predict(state, torch.tensor([1.0, 2.0]))

    def test_time_contract(self):
        state = self.model.initial_state(1, torch.device("cpu"), torch.float32)
        tokens = synthetic_tokens(4, 4)
        advanced, _ = self.model.observe(state, tokens, 2.0, sample=False)
        self.assertAlmostEqual(float(advanced.time), 2.0, places=9)
        for bad in (1.5, 2.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=bad):
                self.model.observe(advanced, tokens, bad, sample=False)
        clone = self.model.advance(advanced, 0)
        self.assertIsNot(clone, advanced)
        self.assertTrue(torch.equal(clone.slots, advanced.slots))
        with self.assertRaises(ValueError):
            self.model.advance(advanced, -1.0)
        with self.assertRaises(ValueError):
            self.model.advance(advanced, 10_000.0)  # above max_substeps

    def test_clocks_stay_float64(self):
        state = self.model.initial_state(2, torch.device("cpu"), torch.float32)
        self.assertEqual(state.time.dtype, torch.float64)
        self.assertEqual(state.to(dtype=torch.float32).time.dtype, torch.float64)
        advanced, _ = self.model.observe(state, synthetic_tokens(4, 4).expand(2, -1, -1),
                                         5.4054000000000004, sample=False)
        self.assertEqual(advanced.time.dtype, torch.float64)
        self.assertAlmostEqual(float(advanced.time[0]), 5.4054, places=12)

    def test_reset_restores_learned_initial_state(self):
        state = self.model.initial_state(2, torch.device("cpu"), torch.float32)
        tokens = torch.randn(2, PATCHES, 48, generator=torch.Generator().manual_seed(2))
        advanced, _ = self.model.observe(state, tokens, 2.0, sample=False)
        reset = self.model.reset_state(advanced, torch.tensor([True, False]))
        fresh = self.model.initial_state(2, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(reset.slots[0], fresh.slots[0]))
        self.assertFalse(torch.equal(reset.slots[0], torch.zeros_like(reset.slots[0])))
        self.assertEqual(float(reset.time[0]), 0.0)
        self.assertTrue(torch.equal(reset.slots[1], advanced.slots[1]))

    def test_state_round_trips_through_disk(self):
        state = self.model.initial_state(1, torch.device("cpu"), torch.float32)
        observation = synthetic_tokens(4, 4)
        other = synthetic_tokens(4, 4, pattern="row_gradient")
        first, _ = self.model.observe(state, observation, 2.0, sample=False)
        second, _ = self.model.observe(state, other, 2.0, sample=False)
        # the state depends on what was observed (non-vacuous), and a saved state
        # reproduces its predictions exactly
        self.assertFalse(torch.equal(first.slots, second.slots))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            first.save(path)
            restored = WorldState.load(path, device=torch.device("cpu"), dtype=torch.float32)
            for name in ("slots", "velocity", "time", "step"):
                self.assertTrue(torch.equal(getattr(first, name), getattr(restored, name)), name)
            self.assertTrue(torch.equal(self.model.predict(first, 4.0)[0],
                                        self.model.predict(restored, 4.0)[0]))

    def test_units_and_aggregation(self):
        mu = torch.zeros(1, 4, 3)
        logvar = torch.zeros(1, 4, 3)
        per_dim = gaussian_nll_bits(torch.zeros(1, 4, 3), mu, logvar)
        expected = (0.5 * math.log(2 * math.pi)) / LN2
        self.assertAlmostEqual(float(per_dim.mean()), expected, places=5)
        self.assertAlmostEqual(float(per_dim.sum(dim=(1, 2)).mean()), expected * 12, places=5)
        perturbed = torch.zeros_like(mu)
        perturbed[0, 0, 0] = 2.0
        self.assertAlmostEqual(float(gaussian_nll_bits(perturbed, mu, logvar).mean()),
                               expected + (4.0 / 2.0) / 12 / LN2, places=5)


class TimestampTests(unittest.TestCase):
    def test_chunk_times_come_from_the_source_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_video(Path(temporary) / "clip30.mp4", seed=7, fps=30.0, frames=80)
            info = probe_video(path)
            config = DataConfig(fps=4.0, chunk_frames=8, chunk_stride_frames=8, image_size=SIZE)
            frames, timestamps = read_frames(path, config.fps, config.image_size)
            step = max(1, round(info.fps / config.fps))
            self.assertEqual(step, 8)
            chunks = build_chunks(info, config, timestamps)
            self.assertTrue(chunks)
            for chunk in chunks:
                self.assertAlmostEqual(chunk.start_seconds, chunk.start_frame * step / info.fps,
                                       places=6)
                self.assertAlmostEqual(chunk.end_seconds,
                                       (chunk.end_frame - 1) * step / info.fps + info.frame_period,
                                       places=6)
            # the naive sampled_index / requested_fps formula drifts on this clip
            self.assertGreater(abs(chunks[-1].end_seconds - chunks[-1].end_frame / config.fps), 0.05)

    def test_overlapping_chunks_are_rejected(self):
        config = DataConfig(chunk_frames=8, chunk_stride_frames=4)
        with self.assertRaisesRegex(ValueError, "chunk_stride_frames"):
            config and RunConfigValidate(config)


def RunConfigValidate(data: DataConfig):
    from wpm_video.config import RunConfig
    RunConfig(data=data).validate()


if __name__ == "__main__":
    unittest.main()
