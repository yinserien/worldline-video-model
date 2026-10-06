"""Provenance and path-boundary contracts: declared coordinates, stale hashes, str paths."""

from pathlib import Path
import tempfile
import unittest

from wpm_video import RunConfig, WorldState, probe_video, read_frames
from wpm_video.data import (PUBLIC_DATASET, build_source_manifest, list_videos, load_source_manifest,
                            repository_url, resolve_source)
from wpm_video.dataset import TokenDataset, cache_video_tokens

from .fixtures import make_clips, native_encoder, tiny_config

CUSTOM_SHA = "c" * 64


class PathBoundaryTests(unittest.TestCase):
    """Every public entry point that takes a path must accept str as well as Path."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.video = make_clips(self.root / "videos", 2)[0]

    def tearDown(self):
        self._temporary.cleanup()

    def test_str_paths_are_accepted(self):
        info = probe_video(str(self.video))
        self.assertEqual(info.name, self.video.stem)
        self.assertTrue(info.sha256)
        self.assertEqual(probe_video(self.video).sha256, info.sha256)

    def test_str_paths_for_frames_chunking_and_dataset(self):
        frames, timestamps = read_frames(str(self.video), 4.0, 64)
        self.assertEqual(frames.shape[1:], (3, 64, 64))
        self.assertEqual(timestamps.dtype, __import__("torch").float64)
        self.assertEqual(len(list_videos(str(self.root / "videos"))), 2)
        config = tiny_config(["clip0"], ["clip1"])
        cache_dir = self.root / "cache"
        for name in ("clip0", "clip1"):
            cache_video_tokens(str(self.root / "videos" / f"{name}.mp4"), config.data,
                               native_encoder(), str(cache_dir))
        dataset = TokenDataset(config.data, str(cache_dir), config.encoder, "train")
        self.assertTrue(dataset.windows)

    def test_str_paths_for_config_and_state(self):
        config = tiny_config(["a"], ["b"])
        path = self.root / "config.json"
        config.save(str(path))
        self.assertEqual(RunConfig.load(str(path)).name, config.name)
        state = WorldState.init(1, 4, 8)
        state_path = self.root / "state.pt"
        state.save(str(state_path))
        self.assertEqual(tuple(WorldState.load(str(state_path)).slots.shape), (1, 4, 8))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.video = make_clips(self.root / "videos", 1)[0]
        self.info = probe_video(self.video)

    def tearDown(self):
        self._temporary.cleanup()

    def test_url_is_built_from_the_declared_coordinates(self):
        """A custom repository must never inherit the default dataset URL."""
        record = resolve_source(self.info, {self.video.stem: {
            "repo_id": "my-org/my-dataset", "repo_type": "dataset", "revision": "abc123",
            "source_path": "train/thing.mp4", "content_sha256": self.info.sha256}})
        self.assertEqual(record["kind"], "public_dataset")
        self.assertIn("my-org/my-dataset", record["dataset_url"])
        self.assertIn("/resolve/abc123/train/thing.mp4", record["dataset_url"])
        self.assertNotIn(PUBLIC_DATASET["repo_id"], record["dataset_url"])
        self.assertEqual(record["dataset_revision"], "abc123")
        self.assertTrue(record["content_sha256_matches_declaration"])
        self.assertEqual(repository_url("me/data", "r1", "a/b.mp4"),
                         "https://huggingface.co/datasets/me/data/resolve/r1/a/b.mp4")
        self.assertIn("/models/", repository_url("me/model", "r1", "w.bin", "model"))

    def test_stale_declaration_is_demoted_to_local(self):
        """A declared hash that no longer matches the bytes must not stay public."""
        record = resolve_source(self.info, {self.video.stem: {
            "repo_id": "my-org/my-dataset", "revision": "abc123",
            "source_path": "train/thing.mp4", "content_sha256": CUSTOM_SHA}})
        self.assertEqual(record["kind"], "local_file")
        self.assertEqual(record["demoted_from"], "public_dataset")
        self.assertNotIn("dataset_url", record)
        self.assertNotIn("dataset_repo_id", record)

    def test_undeclared_clip_has_no_dataset_fields(self):
        record = resolve_source(self.info, {})
        self.assertEqual(record["kind"], "local_file")
        self.assertEqual([k for k in record if k.startswith("dataset")], [])
        self.assertEqual(load_source_manifest(self.root / "videos"), {})

    def test_manifest_only_declares_byte_identical_clips(self):
        """build_source_manifest verifies the snapshot bytes, and skips what it cannot."""
        videos = self.root / "staged"
        videos.mkdir()
        good = videos / "archery_ABC123DEF45_000005_000015.mp4"
        other = videos / "bowling_XYZ987WVU65_000010_000020.mp4"
        good.write_bytes(self.video.read_bytes())
        other.write_bytes(make_clips(self.root / "tmp_videos", 2)[1].read_bytes())
        snapshot = self.root / "snapshot" / "train"
        (snapshot / "archery").mkdir(parents=True)
        (snapshot / "bowling").mkdir(parents=True)
        (snapshot / "archery" / "ABC123DEF45_000005_000015.mp4").write_bytes(good.read_bytes())
        (snapshot / "bowling" / "XYZ987WVU65_000010_000020.mp4").write_bytes(b"not the same bytes")
        manifest = build_source_manifest(videos, self.root / "snapshot", "rev1",
                                         repo_id="my-org/my-dataset")
        self.assertIn(good.stem, manifest["clips"])
        self.assertEqual(manifest["clips"][good.stem]["snapshot_sha256"],
                         manifest["clips"][good.stem]["content_sha256"])
        self.assertTrue(any(other.stem in entry for entry in manifest["skipped"]))
        self.assertNotIn(other.stem, manifest["clips"])


if __name__ == "__main__":
    unittest.main()
