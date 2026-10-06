"""RGB target frames for the decoder, decoded lazily and verified against the cache.

The decoder learns ``projected latent of a chunk -> the last sampled frame of that
chunk``. Targets therefore come from the *source videos*, not from the token
cache, and three things have to line up exactly:

- the video content: the SHA-256 recorded in the token cache must equal the SHA-256
  of the file on disk, so a replaced or truncated clip is refused instead of
  silently pairing latents with unrelated pixels;
- the sampling grid: re-decoding with the cache's ``fps`` must reproduce the same
  sampled frames, and every chunk's index/frames/timestamps must match both the
  chunk list stored inside the token payload (the one ``TokenDataset`` reads) and the
  copy in its metadata;
- the timestamp: the target frame is ``frames[end_frame - 1]``, whose source
  timestamp is ``chunk.end_seconds - 1/source_fps``, asserted rather than assumed.

Every number involved in these comparisons is checked to be finite first: a NaN
timestamp compares false against any tolerance and would otherwise pass silently.

Nothing is loaded eagerly: at most ``max_videos`` videos are decoded and held in
RAM (LRU eviction), one resolution-sized uint8 tensor per video.
"""

from collections import OrderedDict
import math
from pathlib import Path

import torch

from ..config import DataConfig
from ..data import build_chunks, file_sha256, probe_video, read_frames
from ..dataset import TIMELINE_SCHEMA

CHUNK_FIELDS = ("index", "start_frame", "end_frame")


class CacheAlignmentError(ValueError):
    """Raised when a source video or a cache record no longer matches its token cache."""


def _finite(value, what: str, video_name: str) -> float:
    """Numbers used in alignment checks must be real numbers, never NaN.

    ``abs(nan - x) > tol`` is ``False``, so a NaN timestamp would slip through every
    tolerance comparison; this turns it into a hard error instead.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CacheAlignmentError(f"{video_name}: {what} is not a number: {value!r}") from None
    if not math.isfinite(number):
        raise CacheAlignmentError(f"{video_name}: {what} is not finite: {value!r}")
    return number


def _chunk_records(cache, video_name: str) -> list:
    """The chunk records training actually uses, verified against the cache metadata.

    ``TokenDataset.chunk`` reads ``cache.chunks`` (the token payload), while the
    sampling provenance lives in ``cache.meta['chunks']``. If the two ever disagree,
    a latent and its target pixels would come from different chunk boundaries, so they
    are compared field by field here and the payload is treated as authoritative.
    """
    payload_chunks = list(getattr(cache, "chunks", None) or [])
    meta_chunks = cache.meta.get("chunks")
    if not payload_chunks:
        raise CacheAlignmentError(f"{video_name}: token cache carries no chunk records")
    tokens = getattr(cache, "tokens", None)
    if tokens is None or not hasattr(tokens, "shape") or tokens.dim() < 1:
        raise CacheAlignmentError(f"{video_name}: token cache carries no token tensor")
    if int(tokens.shape[0]) != len(payload_chunks):
        # a truncated or padded token payload would otherwise surface much later, as a
        # missing row in some arbitrary sampled batch
        raise CacheAlignmentError(
            f"{video_name}: token payload holds {int(tokens.shape[0])} rows but the cache lists "
            f"{len(payload_chunks)} chunks; the cache is inconsistent"
        )
    if not isinstance(meta_chunks, list) or len(meta_chunks) != len(payload_chunks):
        raise CacheAlignmentError(
            f"{video_name}: the token cache chunk list ({len(payload_chunks)} chunks) and its "
            f"recorded metadata ({len(meta_chunks) if isinstance(meta_chunks, list) else 'missing'}) "
            "disagree; the cache is inconsistent"
        )
    records = []
    for position, (payload, meta) in enumerate(zip(payload_chunks, meta_chunks)):
        for field in CHUNK_FIELDS:
            left, right = payload.get(field), meta.get(field)
            if left is None or right is None:
                raise CacheAlignmentError(f"{video_name}: chunk {position} has no '{field}'")
            if int(left) != int(right):
                raise CacheAlignmentError(
                    f"{video_name}: chunk {position} {field} differs between the cache payload "
                    f"({left!r}) and its metadata ({right!r})"
                )
        for field in ("start_seconds", "end_seconds"):
            left = _finite(payload.get(field), f"chunk {position} {field}", video_name)
            right = _finite(meta.get(field), f"recorded chunk {position} {field}", video_name)
            if abs(left - right) > 1e-9:
                raise CacheAlignmentError(
                    f"{video_name}: chunk {position} {field} differs between the cache payload "
                    f"({left!r}) and its metadata ({right!r})"
                )
        if int(payload["end_frame"]) <= int(payload["start_frame"]):
            raise CacheAlignmentError(f"{video_name}: chunk {position} is empty")
        records.append({field: payload[field] for field in CHUNK_FIELDS} | {
            "start_seconds": float(payload["start_seconds"]),
            "end_seconds": float(payload["end_seconds"]),
        })
    return records


def alignment_record(cache, require_identity: bool = False) -> dict:
    """The sampling facts of one token cache that a decoder target must reproduce.

    ``require_identity`` is used by the training path: without a content hash, a
    source fps and a chunk list there is nothing to verify the pixels against, so a
    missing record fails loudly instead of training on unverified targets.
    """
    meta = cache.meta
    name = str(meta.get("video") or "video")
    record = {
        "video": meta.get("video"),
        "path": meta.get("path"),
        "sha256": meta.get("sha256") or "",
        "source_fps": _finite(meta.get("source_fps") or 0.0, "source_fps", name),
        "fps": _finite(meta.get("sampled_fps_target") or 0.0, "sampled_fps_target", name),
        "chunk_frames": int(meta.get("chunk_frames") or 0),
        "chunk_stride_frames": int(meta.get("chunk_stride_frames") or 0),
        "image_size": int(meta.get("image_size") or 0),
        "timeline_schema": meta.get("timeline_schema"),
        "sampled_frame_indices": [int(value) for value in (meta.get("sampled_frame_indices") or [])],
        "chunks": _chunk_records(cache, name),
    }
    if require_identity:
        missing = [field for field in ("sha256", "timeline_schema", "chunks", "sampled_frame_indices")
                   if not record[field]]
        if record["source_fps"] <= 0 or record["fps"] <= 0:
            missing.append("source_fps/sampled_fps_target")
        if missing:
            raise CacheAlignmentError(
                f"{name}: token cache is missing {sorted(set(missing))}, which the decoder training "
                "path needs to verify target pixels against; re-run `wpm-video cache` with this "
                "package version"
            )
    return record


def check_alignment(record: dict, config: DataConfig, video_name: str) -> None:
    """Refuse a cache that was written with different sampling than this config."""
    expected = {
        "fps": float(config.fps),
        "chunk_frames": int(config.chunk_frames),
        "chunk_stride_frames": int(config.chunk_stride_frames),
        "image_size": int(config.image_size),
        "timeline_schema": TIMELINE_SCHEMA,
    }
    for name, value in expected.items():
        recorded = record.get(name)
        if name != "timeline_schema":
            recorded = _finite(recorded, name, video_name)
        if recorded != value:
            raise CacheAlignmentError(
                f"{video_name}: token cache {name}={record.get(name)!r} does not match the "
                f"configuration ({value!r}); the decoder would be trained against a timeline it "
                "does not describe. Re-run `wpm-video cache` with this configuration."
            )


class ChunkFrameSource:
    """Bounded, lazy per-video RGB frames for decoder targets.

    ``image_size`` is the decoder's output edge: frames are decoded directly at the
    resolution the decoder predicts, so no separate resampling step can drift from
    the token pipeline. Videos are decoded on first use and evicted least-recently
    used once more than ``max_videos`` are held.
    """

    def __init__(self, video_dir, data_config: DataConfig, image_size: int, max_videos: int = 2):
        self.video_dir = Path(video_dir)
        self.config = data_config
        self.image_size = int(image_size)
        self.max_videos = max(1, int(max_videos))
        self._records: dict[str, dict] = {}
        self._frames: OrderedDict[str, tuple] = OrderedDict()
        self.decoded_videos = 0     # how many videos were decoded (for summaries)
        self.verified_hashes = 0

    # -- registration --------------------------------------------------------
    def register(self, name: str, record: dict) -> None:
        """Bind a video name to the token-cache record it must stay consistent with."""
        check_alignment(record, self.config, name)
        self._records[name] = record

    def resolve(self, name: str) -> Path:
        """Locate the source video: the portable path first, the recorded one second.

        The cache records where the clip lived when it was encoded, which may be a
        path from another machine; ``video_dir/name.mp4`` is tried first so a moved
        working directory keeps working. The content hash check decides, not the path.
        """
        candidates = [self.video_dir / f"{name}.mp4"]
        recorded = self._records.get(name, {}).get("path")
        if recorded:
            candidates.append(Path(recorded))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"source video for {name!r} not found; looked at "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    # -- decoding ------------------------------------------------------------
    def _decode(self, name: str) -> tuple:
        if name in self._frames:
            self._frames.move_to_end(name)
            return self._frames[name]
        record = self._records.get(name, {})
        path = self.resolve(name)
        expected_hash = record.get("sha256") or ""
        if expected_hash:
            actual = file_sha256(path)
            if actual != expected_hash:
                raise CacheAlignmentError(
                    f"{path} no longer matches the token cache: recorded sha256 {expected_hash[:12]}…, "
                    f"file on disk {actual[:12]}…. The clip was replaced after caching, so cached "
                    "latents and target pixels would not describe the same video."
                )
            self.verified_hashes += 1
        frames, timestamps = read_frames(path, self.config.fps, self.image_size)
        self._check_timeline(name, path, frames, timestamps)
        self._frames[name] = (frames, timestamps)
        self.decoded_videos += 1
        while len(self._frames) > self.max_videos:
            self._frames.popitem(last=False)
        return self._frames[name]

    def _check_timeline(self, name: str, path: Path, frames: torch.Tensor,
                        timestamps: torch.Tensor) -> None:
        """The re-decoded timeline must reproduce the cache's sampled grid exactly."""
        record = self._records.get(name)
        if not record or not record.get("chunks"):
            return
        info = probe_video(path, with_hash=False)
        rebuilt = build_chunks(info, self.config, timestamps)
        cached = record["chunks"]
        if len(rebuilt) != len(cached):
            raise CacheAlignmentError(
                f"{name}: re-decoding the source gives {len(rebuilt)} chunks but the token cache "
                f"recorded {len(cached)}; the video, its fps or the sampling configuration changed"
            )
        indices = record.get("sampled_frame_indices") or []
        if indices and indices != [chunk.start_frame for chunk in rebuilt]:
            raise CacheAlignmentError(
                f"{name}: sampled frame indices differ from the token cache record; the decoder "
                "targets would not sit on the cached timeline"
            )
        for rebuilt_chunk, cached_chunk in zip(rebuilt, cached):
            for field in CHUNK_FIELDS:
                if int(getattr(rebuilt_chunk, field)) != int(cached_chunk[field]):
                    raise CacheAlignmentError(
                        f"{name}: chunk {cached_chunk['index']} {field} differs from the token cache"
                    )
            for field in ("start_seconds", "end_seconds"):
                recorded = _finite(cached_chunk[field], f"chunk {cached_chunk['index']} {field}", name)
                rebuilt_value = _finite(getattr(rebuilt_chunk, field),
                                        f"re-decoded chunk {cached_chunk['index']} {field}", name)
                if abs(rebuilt_value - recorded) > 1e-6:
                    raise CacheAlignmentError(
                        f"{name}: chunk {cached_chunk['index']} {field} differs from the token "
                        "cache by more than a microsecond"
                    )
        if frames.shape[0] < cached[-1]["end_frame"]:
            raise CacheAlignmentError(
                f"{name}: decoded {frames.shape[0]} sampled frames but the cache refers to frame "
                f"{cached[-1]['end_frame'] - 1}"
            )

    # -- public API ----------------------------------------------------------
    def frames(self, name: str) -> tuple:
        """``(frames (T, 3, S, S) uint8, timestamps (T,) float64)`` for one video."""
        return self._decode(name)

    def target_frame(self, name: str, chunk: dict) -> tuple:
        """The target keyframe of one cached chunk: its last sampled frame.

        Returns ``(frame (3, S, S) uint8, timestamp_seconds)``. The frame index comes
        from the cache's own chunk record, so a latent and a pixel can never refer to
        different chunks, and the timestamp is *asserted* to be exactly one source
        frame before the chunk end (``end_seconds - 1/source_fps``), which is what
        makes "the last sampled frame of the chunk" checkable rather than assumed.
        """
        frames, timestamps = self._decode(name)
        chunk_end = _finite(chunk.get("end_seconds"), f"chunk {chunk.get('index')} end_seconds", name)
        index = int(chunk["end_frame"]) - 1
        if not 0 <= index < frames.shape[0]:
            raise CacheAlignmentError(
                f"{name}: chunk {chunk['index']} ends at sampled frame {index}, outside the "
                f"{frames.shape[0]} frames decoded from the source"
            )
        timestamp = _finite(timestamps[index], f"timestamp of frame {index}", name)
        source_fps = _finite(self._records.get(name, {}).get("source_fps") or 0.0, "source_fps", name)
        if source_fps > 0:
            expected = chunk_end - 1.0 / source_fps
            if abs(timestamp - expected) > 1e-6:
                raise CacheAlignmentError(
                    f"{name}: chunk {chunk['index']} target frame is at t={timestamp:.9f}s but the "
                    f"chunk end ({chunk_end:.9f}s) minus one source frame ({expected:.9f}s) does "
                    "not match; the target is not the chunk's last sampled frame on the recorded "
                    "timeline"
                )
        return frames[index], timestamp

    def target(self, name: str, chunk: dict) -> torch.Tensor:
        """Just the target keyframe of one cached chunk, ``(3, S, S)`` uint8."""
        return self.target_frame(name, chunk)[0]

    def release(self, name: str) -> None:
        self._frames.pop(name, None)
