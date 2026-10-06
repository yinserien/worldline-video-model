"""CLI and public API smoke tests, run against the installed package."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

import wpm_video
from wpm_video.cli import build_parser, main
from wpm_video.config import RunConfig
from wpm_video.dataset import TokenDataset, cache_video_tokens
from wpm_video.encoder import build_encoder
from wpm_video.train import build_model, fit_projection, train

from .fixtures import make_clips, tiny_config


class PublicApiTests(unittest.TestCase):
    def test_package_exports_the_documented_api(self):
        for name in ("RunConfig", "WorldState", "VideoWorldModel", "build_encoder", "train",
                     "stream_chunks", "predict_at", "query_saved_state", "cache_video_tokens",
                     "evaluate_model", "compare", "source_signature"):
            self.assertTrue(hasattr(wpm_video, name), name)
            self.assertIn(name, wpm_video.__all__)
        self.assertTrue(wpm_video.__version__)

    def test_source_signature_follows_the_installed_package(self):
        """No src/ tree is assumed: the signature hashes the module that was imported."""
        from wpm_video.train import source_signature
        package_dir = Path(wpm_video.__file__).resolve().parent
        self.assertEqual(source_signature(), source_signature(package_dir))
        self.assertEqual(len(source_signature()), 64)
        self.assertNotEqual(source_signature(package_dir), source_signature(Path(tempfile.mkdtemp())))

    def test_config_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = tiny_config(["a"], ["b"])
            path = Path(temporary) / "config.json"
            config.save(path)
            restored = RunConfig.load(path)
            self.assertEqual(restored.to_dict(), config.to_dict())
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["unknown_section"] = {}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                RunConfig.load(path)


class CliTests(unittest.TestCase):
    def test_parser_exposes_every_command(self):
        parser = build_parser()
        commands = {
            "cache": [], "train": [], "provenance": [], "selfcheck": [],
            "eval": ["--checkpoint", "best.pt"],
            "predict": ["--checkpoint", "best.pt", "--video", "clip.mp4"],
            "query": ["--checkpoint", "best.pt", "--state", "state.pt"],
        }
        for command, extra in commands.items():
            args = parser.parse_args([command, "--config", "configs/vjepa2_256.json", *extra])
            self.assertEqual(args.command, command)

    def test_selfcheck_runs_through_the_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            tiny_config(["a"], ["b"]).save(config_path)
            self.assertEqual(main(["selfcheck", "--config", str(config_path)]), 0)

    def test_query_command_reports_state_only_futures(self):
        from wpm_video.selfcheck import build_fixture

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _ = build_fixture()
            checkpoint = root / "model.pt"
            model.save(checkpoint)
            state = model.initial_state(1, torch.device("cpu"), torch.float32)
            tokens = torch.randn(1, 16, 48, generator=torch.Generator().manual_seed(0))
            advanced, _ = model.observe(state, tokens, 2.0, sample=False)
            state_path = root / "state.pt"
            advanced.save(state_path)
            config_path = root / "config.json"
            tiny_config(["a"], ["b"]).save(config_path)
            out = root / "query"
            code = main(["query", "--config", str(config_path), "--checkpoint", str(checkpoint),
                         "--state", str(state_path), "--deltas", "1", "4", "--out", str(out)])
            self.assertEqual(code, 0)
            payload = json.loads((out / "state_query.json").read_text(encoding="utf-8"))
            self.assertEqual(set(payload["queries"]), {"1.0", "4.0"})
            self.assertAlmostEqual(payload["state_time_seconds"], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()


class PredictPortabilityTests(unittest.TestCase):
    """`predict` must work from a checkpoint + config + video alone, with no training data.

    These drive the real CLI entry point (no mocking of the prediction path): a
    minimal model saved with ``VideoWorldModel.save`` has no provenance at all, and
    a trained checkpoint may sit next to a machine that never had the token cache.
    """

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.video = make_clips(cls.root / "videos", 1)[0]
        cls.config_path = cls.root / "config.json"
        cls.config = tiny_config(["a"], ["b"])
        cls.config.data.cache_dir = str(cls.root / "absent_cache")  # never created
        cls.config.save(cls.config_path)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def run_predict(self, checkpoint: Path, out_name: str) -> Path:
        out = self.root / out_name
        code = main(["predict", "--config", str(self.config_path), "--checkpoint", str(checkpoint),
                     "--video", str(self.video), "--out", str(out), "--prefix-chunks", "2",
                     "--allow-native"])
        self.assertEqual(code, 0)
        return out

    def assert_inference_artifacts(self, out: Path) -> None:
        for name in ("world_state.pt", "future_latents.pt", "predict_summary.json",
                     "frames.png", "uncertainty.png", "pca_meta.json"):
            self.assertTrue((out / name).is_file(), name)
        payload = json.loads((out / "predict_summary.json").read_text(encoding="utf-8"))
        self.assertGreater(payload["state_time_seconds"], 0.0)
        self.assertTrue(payload["horizons"])

    def native_encoder(self, config):
        """Same construction path the CLI uses, so widths agree everywhere."""
        return build_encoder(config.encoder, torch.device("cpu"), allow_native=True)

    def native_model(self, config, encoder=None):
        return build_model(config, encoder or self.native_encoder(config), torch.device("cpu"))

    def test_basic_saved_checkpoint_predicts_without_provenance_or_train_cache(self):
        model = self.native_model(self.config)
        checkpoint = self.root / "native_model.pt"
        model.save(checkpoint)                       # no provenance, as a bare model.save would
        out = self.run_predict(checkpoint, "out_basic")
        self.assert_inference_artifacts(out)
        meta = json.loads((out / "pca_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "skipped")
        self.assertIn("provenance", meta["reason"])
        self.assertFalse((out / "latent_pca.png").exists())
        # inference needed no token cache: the directory holds no cached clip
        cache_dir = Path(self.config.data.cache_dir)
        self.assertEqual(list(cache_dir.rglob("*.pt")) if cache_dir.exists() else [], [])

    def test_trained_checkpoint_without_its_token_cache_still_predicts(self):
        """A moved checkpoint keeps working; only the optional PCA is skipped."""
        config = tiny_config(["clip0"], ["clip1"], steps=2)
        config.encoder = type(config.encoder)(kind="native", device="cpu", batch_clips=2,
                                              allow_native_fallback=True)
        config.data.cache_dir = str(self.root / "train_cache")
        clips = make_clips(self.root / "train_videos", 2)
        encoder = self.native_encoder(config)
        for clip in clips:
            cache_video_tokens(clip, config.data, encoder, config.data.cache_dir)
        train_set = TokenDataset(config.data, Path(config.data.cache_dir), config.encoder, "train")
        val_set = TokenDataset(config.data, Path(config.data.cache_dir), config.encoder, "val")
        torch.manual_seed(0)
        model = self.native_model(config, encoder)
        fit_projection(model, train_set)
        train(config, model, train_set, val_set, self.root / "run", torch.device("cpu"),
              provenance={"splits": {"train": ["clip0"], "val": ["clip1"]}})
        checkpoint = self.root / "run" / "final.pt"
        self.assertIn("provenance", torch.load(checkpoint, map_location="cpu", weights_only=False))
        # the moved machine does not have the training cache
        moved_config = tiny_config(["a"], ["b"])
        moved_config.data.cache_dir = str(self.root / "machine_without_cache")
        moved_config.save(self.config_path)
        out = self.run_predict(checkpoint, "out_trained")
        self.assert_inference_artifacts(out)
        meta = json.loads((out / "pca_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "skipped")
        self.assertIn("token cache", meta["reason"])
        self.assertTrue(meta["checkpoint_has_provenance"])

    def test_pca_is_written_when_the_training_cache_is_present(self):
        """With provenance and the cache available the plot is produced from train data."""
        config = tiny_config(["clip0"], ["clip1"], steps=2)
        config.data.cache_dir = str(self.root / "train_cache2")
        clips = make_clips(self.root / "train_videos2", 2)
        encoder = self.native_encoder(config)
        for clip in clips:
            cache_video_tokens(clip, config.data, encoder, config.data.cache_dir)
        train_set = TokenDataset(config.data, Path(config.data.cache_dir), config.encoder, "train")
        val_set = TokenDataset(config.data, Path(config.data.cache_dir), config.encoder, "val")
        torch.manual_seed(0)
        model = self.native_model(config, encoder)
        fit_projection(model, train_set)
        train(config, model, train_set, val_set, self.root / "run2", torch.device("cpu"),
              provenance={"splits": {"train": ["clip0"], "val": ["clip1"]}})
        config.save(self.config_path)
        out = self.run_predict(self.root / "run2" / "final.pt", "out_pca")
        self.assert_inference_artifacts(out)
        meta = json.loads((out / "pca_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "written")
        self.assertGreater(meta["n_fit_latents"], 0)
        self.assertTrue((out / "latent_pca.png").is_file())
