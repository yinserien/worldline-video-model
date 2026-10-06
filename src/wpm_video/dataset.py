"""Cached-token dataset: causal windows, real timestamps and multi-horizon targets.

Training never runs the video encoder: chunks are encoded once into a cache that
records real source timestamps, the source video hash and the source identity.
The cache key carries a timeline schema version, so caches written with the older
sample-index timeline are never silently reused.

A target is the cached token block of a chunk that starts at or after the anchor
chunk's end time, so a target can never be an input of the state that predicts it,
and horizon deltas come from the real timestamps rather than an assumed constant
chunk spacing.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import DataConfig
from .data import (Chunk, VideoInfo, build_chunks, list_videos, load_source_manifest, probe_video,
                   read_frames, resolve_source, windows_for_video)
from .encoder import TokenCache, cache_path

TIMELINE_SCHEMA = "t2-source-timestamps"

# Provenance of the public clip collection the staged videos come from.
SOURCE_DATASET = "nateraw/kinetics-mini"
SOURCE_DATASET_REVISION = "9f4ed38128a355c352527209101be3e326471816"


@dataclass
class Window:
    video: str
    split: str
    start: int
    chunk_count: int
    end_seconds: float

    @property
    def key(self) -> str:
        return f"{self.video}:{self.start}"


class TokenDataset:
    """Token cache plus the causal window index for one split."""

    def __init__(self, config: DataConfig, cache_dir, encoder_config, split: str):
        cache_dir = Path(cache_dir)
        self.config = config
        self.split = split
        self.entries: dict[str, TokenCache] = {}
        names = config.train_videos if split == "train" else config.val_videos
        for name in names:
            path = cache_path(cache_dir, name, config, encoder_config)
            if not path.is_file():
                raise FileNotFoundError(f"Missing token cache for {name}: {path}")
            self.entries[name] = TokenCache(path)
        self.windows: list[Window] = []
        for name, cache in self.entries.items():
            for start in windows_for_video(cache.tokens.shape[0], config):
                self.windows.append(Window(name, split, start, cache.tokens.shape[0],
                                           cache.chunks[start + config.context_chunks - 1]["end_seconds"]))
        if not self.windows:
            raise RuntimeError(f"No causal windows available for split {split!r}")

    def tokens(self, video: str, chunk: int) -> torch.Tensor:
        return self.entries[video].tokens[chunk]

    def chunk(self, video: str, chunk: int) -> dict:
        return self.entries[video].chunks[chunk]

    def __len__(self) -> int:
        return len(self.windows)

    def provenance(self) -> dict:
        return {
            "split": self.split,
            "timeline_schema": TIMELINE_SCHEMA,
            "videos": sorted(self.entries),
            "chunks_per_video": {name: int(len(cache)) for name, cache in self.entries.items()},
            "windows": len(self.windows),
            "window_keys": [window.key for window in self.windows],
            "sources": {name: cache.meta.get("source") for name, cache in self.entries.items()},
        }


def cache_video_tokens(video, config: DataConfig, encoder, out_dir, batch_clips: int = 4,
                       source_manifest: dict | None = None) -> dict:
    """Encode every causal chunk of one video once and store tokens with real timestamps.

    Provenance comes from the explicit source manifest when the clip is declared
    there; otherwise the clip is recorded as a local file and no public dataset is
    claimed. The manifest declaration is verified against the file content hash.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    info = probe_video(video)
    source = resolve_source(info, source_manifest if source_manifest is not None else {})
    if source["kind"] == "public_dataset" and not source["content_sha256_matches_declaration"]:
        raise ValueError(
            f"{video.name}: content hash does not match the declared source "
            f"{source['declared_content_sha256']}; refusing to record a false provenance"
        )
    frames, timestamps = read_frames(video, config.fps, config.image_size)
    chunks = build_chunks(info, config, timestamps)
    if not chunks:
        raise RuntimeError(f"Video too short for one chunk: {video}")
    tokens = []
    with torch.no_grad():
        for start in range(0, len(chunks), batch_clips):
            for chunk in chunks[start:start + batch_clips]:
                clip = frames[chunk.start_frame:chunk.end_frame]
                tokens.append(encoder.encode_clip(clip).to("cpu").to(torch.float16))
    stacked = torch.stack(tokens)
    meta = {
        "timeline_schema": TIMELINE_SCHEMA,
        "video": info.name, "path": info.path, "source_fps": info.fps,
        "frame_count": info.frame_count, "duration_seconds": info.duration_seconds,
        "sha256": info.sha256, "source_id": info.source_id, "source_uri": info.source_uri,
        "source": source,
        "sampled_fps_target": config.fps,
        "sampled_frame_indices": [c.start_frame for c in chunks],
        "chunk_frames": config.chunk_frames, "chunk_stride_frames": config.chunk_stride_frames,
        "image_size": config.image_size,
        "encoder": encoder.kind, "encoder_pretrained": bool(encoder.is_pretrained),
        "encoder_revision": getattr(encoder, "revision", None),
        "encoder_d_model": encoder.d_model, "patches": encoder.patches,
        "chunks": [
            {"index": c.index, "start_frame": c.start_frame, "end_frame": c.end_frame,
             "start_seconds": c.start_seconds, "end_seconds": c.end_seconds}
            for c in chunks
        ],
    }
    path = cache_path(out_dir, info.name, config, encoder)
    TokenCache.save(path, stacked, meta["chunks"], meta)
    meta["cache_path"] = path.as_posix()
    meta["cache_sha256"] = file_digest(path)
    return meta


def file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_split_lists(video_dir: Path, val_count: int = 2) -> dict:
    """Split by whole video; validation frames never belong to a training video."""
    videos = list_videos(video_dir)
    if len(videos) <= val_count:
        raise RuntimeError("Not enough videos to hold out a validation set")
    infos = [probe_video(v, with_hash=False) for v in videos]
    val = [info.name for info in infos[-val_count:]]
    train = [info.name for info in infos if info.name not in val]
    return {"train": sorted(train), "val": sorted(val),
            "videos": [{"name": i.name, "path": i.path, "fps": i.fps, "frames": i.frame_count,
                        "seconds": i.duration_seconds, "source_id": i.source_id} for i in infos]}


def _split_identities(dataset: TokenDataset) -> tuple:
    """(hashes, source identities) with missing values dropped, never as (None, None)."""
    hashes = {cache.meta["sha256"] for cache in dataset.entries.values()
              if cache.meta.get("sha256")}
    sources = {cache.meta["source_id"] for cache in dataset.entries.values()
               if cache.meta.get("source_id")}
    return hashes, sources


def check_split_integrity(dataset_a: TokenDataset, dataset_b: TokenDataset) -> None:
    """Refuse splits that share a video file, a content hash, or a source identity.

    The two keys are compared independently: two different clips cut from the same
    original video (same source identity, different file hash) leak, and so do two
    identical files declared under different source names. Entries whose metadata
    is missing are skipped rather than collapsed into a shared ``None`` key, which
    would otherwise reject unrelated splits.
    """
    overlap = set(dataset_a.entries) & set(dataset_b.entries)
    if overlap:
        raise ValueError(f"Splits share source videos: {sorted(overlap)}")
    hashes_a, sources_a = _split_identities(dataset_a)
    hashes_b, sources_b = _split_identities(dataset_b)
    shared_hashes = hashes_a & hashes_b
    if shared_hashes:
        raise ValueError(
            f"Train and validation share a file content hash: {sorted(shared_hashes)}"
        )
    shared_sources = sources_a & sources_b
    if shared_sources:
        raise ValueError(
            f"Train and validation clips come from the same original video: {sorted(shared_sources)}"
        )


class WindowSampler:
    """Deterministic window sampler with its own generator."""

    def __init__(self, dataset: TokenDataset, seed: int):
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int) -> list[Window]:
        order = torch.randint(0, len(self.dataset), (batch_size,), generator=self.generator)
        return [self.dataset.windows[int(i)] for i in order]


def gather_batch(dataset: TokenDataset, windows: list[Window], device, dtype=torch.float32) -> dict:
    """Stack observed chunks, timestamps and per-horizon targets for a batch of windows."""
    config = dataset.config
    observed, end_seconds = [], []
    targets: dict[tuple[int, int], dict] = {}
    for offset in range(config.context_chunks):
        for horizon in config.horizon_chunks:
            targets[(offset, horizon)] = {"tokens": [], "valid": [], "delta_seconds": []}
    for window in windows:
        blocks, ends = [], []
        for i in range(config.context_chunks):
            chunk = dataset.chunk(window.video, window.start + i)
            blocks.append(dataset.tokens(window.video, window.start + i))
            ends.append(chunk["end_seconds"])
        observed.append(torch.stack(blocks))
        end_seconds.append(torch.tensor(ends, dtype=torch.float64))
        for (offset, horizon), entry in targets.items():
            target_chunk = window.start + offset + horizon
            if target_chunk < window.chunk_count:
                target = dataset.chunk(window.video, target_chunk)
                if target["start_seconds"] < ends[offset] - 1e-9:
                    raise ValueError(
                        "A prediction target starts before the anchor chunk ended; "
                        "the horizon is not in the future of its anchor"
                    )
                entry["tokens"].append(dataset.tokens(window.video, target_chunk))
                entry["valid"].append(True)
                entry["delta_seconds"].append(target["end_seconds"] - ends[offset])
            else:
                entry["tokens"].append(torch.zeros_like(blocks[0]))
                entry["valid"].append(False)
                entry["delta_seconds"].append(0.0)
    batch = {
        "observed": torch.stack(observed).to(device=device, dtype=dtype),
        # Times are bookkeeping, not activations: they stay float64 even though the
        # token tensors use the model dtype, so a chunk boundary keeps full precision
        # until the integrator converts it.
        "end_seconds": torch.stack(end_seconds).to(device=device, dtype=torch.float64),
        "targets": {
            key: {
                "tokens": torch.stack(entry["tokens"]).to(device=device, dtype=dtype),
                "valid": torch.tensor(entry["valid"], device=device),
                "delta_seconds": torch.tensor(entry["delta_seconds"], device=device,
                                              dtype=torch.float64),
            }
            for key, entry in targets.items()
        },
        "windows": [window.key for window in windows],
    }
    for key, entry in batch["targets"].items():
        valid = entry["valid"]
        if bool(valid.any()) and bool((entry["delta_seconds"][valid] <= 0).any()):
            raise ValueError(f"Target delta must be strictly positive for {key}")
    return batch
