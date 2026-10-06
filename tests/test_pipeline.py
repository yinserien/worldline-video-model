"""End-to-end pipeline on the native encoder: cache, train, resume, evaluate, demo, query."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from wpm_video.dataset import TokenDataset, cache_video_tokens, gather_batch
from wpm_video.evaluate import baseline_summary, compare, evaluate_baselines, fit_baseline_stats
from wpm_video.model import VideoWorldModel
from wpm_video.predict import demo, query_saved_state, stream_chunks
from wpm_video.train import build_model, evaluate_model, fit_projection, train

from .fixtures import make_clips, native_encoder, tiny_config

CHILD_SCRIPT = '''
import json
import sys
from pathlib import Path
import torch
from wpm_video.config import RunConfig
from wpm_video.dataset import TokenDataset
from wpm_video.encoder import NativeEncoder
from wpm_video.train import build_model, fit_projection, train

payload = json.loads(sys.argv[1])
config = RunConfig.load(Path(payload["config_path"]))
train_set = TokenDataset(config.data, Path(payload["cache_dir"]), config.encoder, "train")
val_set = TokenDataset(config.data, Path(payload["cache_dir"]), config.encoder, "val")
torch.manual_seed(0)
encoder = NativeEncoder(d_model=48, image_size=64, patch=16, tubelet=2, seed=5)
model = build_model(config, encoder, torch.device("cpu"))
fit_projection(model, train_set)
train(config, model, train_set, val_set, Path(payload["out"]), torch.device("cpu"),
      resume=payload["resume"], provenance=payload["provenance"])
print("ok")
'''


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video_dir = cls.root / "videos"
        clips = make_clips(cls.video_dir, 3)
        cls.names = [path.stem for path in clips]
        cls.config = tiny_config(cls.names[:2], cls.names[2:])
        cls.cache_dir = cls.root / "cache"
        cls.encoder = native_encoder()
        for name in cls.names:
            cache_video_tokens(cls.video_dir / f"{name}.mp4", cls.config.data, cls.encoder,
                               cls.cache_dir, batch_clips=2)
        cls.train_set = cls.dataset("train")
        cls.val_set = cls.dataset("val")
        cls.train_dir = cls.root / "train"
        torch.manual_seed(0)
        cls.model = build_model(cls.config, cls.encoder, torch.device("cpu"))
        fit_projection(cls.model, cls.train_set)
        # snapshot after the train-only standardisation is fitted, as training does
        cls.before = {k: v.clone() for k, v in cls.model.state_dict().items()}
        cls.summary = train(cls.config, cls.model, cls.train_set, cls.val_set, cls.train_dir,
                            torch.device("cpu"),
                            provenance={"splits": {"train": cls.names[:2], "val": cls.names[2:]}})

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    @classmethod
    def dataset(cls, split: str) -> TokenDataset:
        return TokenDataset(cls.config.data, cls.cache_dir, cls.config.encoder, split)

    def test_training_runs_and_saves_checkpoints(self):
        self.assertEqual(self.summary["steps"], self.config.train.max_steps)
        for name in ("initial.pt", "best.pt", "final.pt", "train_summary.json"):
            self.assertTrue((self.train_dir / name).is_file(), name)
        moved = sum(1 for k, v in self.model.state_dict().items()
                    if not torch.equal(v, self.before[k]))
        self.assertGreater(moved, 20)
        self.assertTrue(math.isfinite(self.summary["best"]["val_mean_nll_bits_per_dim"]))
        for key, value in self.before.items():
            if key.startswith("projection."):
                self.assertTrue(torch.equal(self.model.state_dict()[key], value), key)

    def test_window_targets_are_strictly_in_the_future(self):
        batch = gather_batch(self.train_set, self.train_set.windows[:2], torch.device("cpu"))
        self.assertEqual(batch["observed"].dtype, torch.float32)
        self.assertEqual(batch["end_seconds"].dtype, torch.float64)
        for key, entry in batch["targets"].items():
            self.assertEqual(entry["delta_seconds"].dtype, torch.float64, key)
            valid = entry["valid"]
            if bool(valid.any()):
                self.assertGreater(float(entry["delta_seconds"][valid].min()), 0.0)

    def test_training_loss_is_differentiable(self):
        from wpm_video.train import forward_window
        model = build_model(self.config, self.encoder, torch.device("cpu"))
        model.train()
        batch = gather_batch(self.train_set, self.train_set.windows[:2], torch.device("cpu"))
        loss, stats = forward_window(model, batch, self.config, sample=True)
        self.assertTrue(loss.requires_grad)
        model.zero_grad(set_to_none=True)
        loss.backward()
        touched = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0]
        self.assertGreater(len(touched), 20)
        for key in ("nll_bits_per_dim_horizon", "kl_bits_per_dim", "kl_bits_per_event", "mse"):
            self.assertIn(key, stats)

    def test_resume_matches_an_uninterrupted_run(self):
        provenance = {"splits": {"train": self.names[:2], "val": self.names[2:]}}

        def fresh(steps):
            torch.manual_seed(0)
            config = tiny_config(self.names[:2], self.names[2:], steps=steps)
            model = build_model(config, self.encoder, torch.device("cpu"))
            fit_projection(model, self.train_set)
            return config, model

        short_config, short_model = fresh(3)
        train(short_config, short_model, self.train_set, self.val_set, self.root / "short",
              torch.device("cpu"), provenance=provenance)
        long_config, live_model = fresh(6)
        train(long_config, live_model, self.train_set, self.val_set, self.root / "live",
              torch.device("cpu"), provenance=provenance)
        resumed_config, resumed_model = fresh(6)
        train(resumed_config, resumed_model, self.train_set, self.val_set, self.root / "resumed",
              torch.device("cpu"), resume=str(self.root / "short" / "final.pt"),
              provenance=provenance)
        live = torch.load(self.root / "live" / "final.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(self.root / "resumed" / "final.pt", map_location="cpu",
                             weights_only=False)
        self.assertEqual(live["step"], resumed["step"])
        for key, value in live["state_dict"].items():
            self.assertTrue(torch.equal(value, resumed["state_dict"][key]), key)

    def test_resume_across_a_new_process(self):
        """A fresh interpreter must reproduce the same resumed trajectory."""
        provenance = {"splits": {"train": self.names[:2], "val": self.names[2:]}}
        torch.manual_seed(0)
        config = tiny_config(self.names[:2], self.names[2:], steps=2)
        model = build_model(config, self.encoder, torch.device("cpu"))
        fit_projection(model, self.train_set)
        train(config, model, self.train_set, self.val_set, self.root / "proc_short",
              torch.device("cpu"), provenance=provenance)
        long_config = tiny_config(self.names[:2], self.names[2:], steps=4)
        torch.manual_seed(0)
        live_model = build_model(long_config, self.encoder, torch.device("cpu"))
        fit_projection(live_model, self.train_set)
        train(long_config, live_model, self.train_set, self.val_set, self.root / "proc_live",
              torch.device("cpu"), resume=str(self.root / "proc_short" / "final.pt"),
              provenance=provenance)
        config_path = self.root / "proc_config.json"
        long_config.save(config_path)
        payload = {
            "config_path": str(config_path),
            "cache_dir": str(self.cache_dir),
            "resume": str(self.root / "proc_short" / "final.pt"),
            "out": str(self.root / "proc_resumed"),
            "provenance": provenance,
        }
        result = subprocess.run([sys.executable, "-c", CHILD_SCRIPT, json.dumps(payload)],
                                capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        live = torch.load(self.root / "proc_live" / "final.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(self.root / "proc_resumed" / "final.pt", map_location="cpu",
                             weights_only=False)
        self.assertEqual(live["step"], resumed["step"])
        for key, value in live["state_dict"].items():
            self.assertTrue(torch.equal(value, resumed["state_dict"][key]), key)

    def test_validation_is_deterministic_and_consumes_no_randomness(self):
        torch.manual_seed(3)
        before = torch.get_rng_state()
        evaluate_model(self.model, self.val_set, self.config, torch.device("cpu"), 1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_baselines_and_comparison(self):
        model, _ = VideoWorldModel.from_checkpoint(self.train_dir / "final.pt")
        metrics = evaluate_model(model, self.val_set, self.config, torch.device("cpu"), 2)
        stats = fit_baseline_stats(model, self.train_set, self.config, torch.device("cpu"), 2)
        baselines = evaluate_baselines(model, self.val_set, self.config, torch.device("cpu"), 2, stats)
        table = compare(metrics, baselines, self.config)
        for key, row in table["per_anchor_horizon"].items():
            self.assertIsNotNone(row["best_baseline_nll_bits_per_dim"], key)
            self.assertIsNotNone(row["best_baseline_mse"], key)
            self.assertEqual(row["n"], metrics["per_anchor_horizon"][key]["n"])
        for kind, entries in baselines.items():
            for key, entry in entries.items():
                self.assertTrue(math.isfinite(entry["nll_bits_per_dim"]), f"{kind}/{key}")
                self.assertTrue(math.isfinite(entry["mse"]), f"{kind}/{key}")
        summary = baseline_summary(baselines)
        self.assertIn("mean_mse", summary["persistence"])
        self.assertIn("model_mean_mse", table["summary"])
        self.assertIn("not learning evidence", table["summary"]["interpretation"])

    def test_demo_and_state_only_query(self):
        model, _ = VideoWorldModel.from_checkpoint(self.train_dir / "final.pt")
        out_dir = self.root / "predict"
        result = demo(model, self.encoder, self.config, self.video_dir / f"{self.names[2]}.mp4",
                      torch.device("cpu"), out_dir, prefix_chunks=2, horizons=[1])
        self.assertTrue((out_dir / "world_state.pt").is_file())
        self.assertTrue((out_dir / "future_latents.pt").is_file())
        self.assertLess(result["summary"]["state_reload_max_abs_mu_difference"], 1e-9)
        entry = result["predictions"][1]
        self.assertEqual(tuple(entry["mu"].shape), (16, 32))
        self.assertGreater(float(entry["mu"].std()), 0.0)
        queried = query_saved_state(model, out_dir / "world_state.pt", [1.0, 4.0],
                                    torch.device("cpu"), self.root / "query")
        self.assertEqual(set(queried["outputs"]), {1.0, 4.0})
        for delta, item in queried["outputs"].items():
            self.assertAlmostEqual(item["target_time_seconds"],
                                   float(result["state"].time) + delta, places=4)

    def test_continuation_skips_observed_chunks(self):
        model, _ = VideoWorldModel.from_checkpoint(self.train_dir / "final.pt")
        from wpm_video.data import build_chunks, probe_video, read_frames
        path = self.video_dir / f"{self.names[2]}.mp4"
        info = probe_video(path)
        frames, timestamps = read_frames(path, self.config.data.fps, self.config.data.image_size)
        chunks = build_chunks(info, self.config.data, timestamps)
        first, _ = stream_chunks(model, self.encoder, frames, chunks[:1], torch.device("cpu"))
        continued, log = stream_chunks(model, self.encoder, frames, chunks[:3],
                                       torch.device("cpu"), state=first)
        self.assertTrue(log[0]["skipped"])
        self.assertFalse(log[1]["skipped"])
        self.assertGreater(float(continued.time), float(first.time))


if __name__ == "__main__":
    unittest.main()
