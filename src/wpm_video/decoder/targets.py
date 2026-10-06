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
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

from ..config import DataConfig
from ..data import build_chunks, file_sha256, probe_video, read_frames
from ..dataset import TIMELINE_SCHEMA
from .compat import TARGET_SEMANTICS

CHUNK_FIELDS = ("index", "start_frame", "end_frame")


class CacheAlignmentError(ValueError):
    """Raised when a source video or a cache record no longer matches its token cache."""


def _replace_entry(temporary: Path, path: Path, attempts: int = 5) -> bool:
    """Atomically publish a staged entry, tolerating a concurrent writer.

    On Windows ``os.replace`` refuses while another handle holds the destination, so a
    short retry loop is needed for concurrent writers. If the destination exists after
    the retries, our staged file is discarded and ``False`` is returned: the existing
    entry was produced for the same identity, so it holds the same frames. A genuine
    problem (a read-only directory, a missing parent) still raises.
    """
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return True
        except PermissionError:
            if attempt == attempts - 1:
                if path.is_file():
                    temporary.unlink(missing_ok=True)
                    return False
                raise
            time.sleep(0.01 * (attempt + 1))
    return False                                # pragma: no cover - loop always returns


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

    ``target_cache_dir`` optionally adds an on-disk cache of the *chunk-last frames
    only* -- never whole videos -- so a later run can reuse targets without decoding.
    Entries are addressed by the source SHA-256 plus the full sampling/timeline/target
    identity, and every entry is re-validated on read, so a changed source, a
    different sampling configuration or a different target size simply cannot hit a
    stale entry. The source-content check is *not* bypassed: the first use of a video
    still hashes the file against the token cache before any target, cached or
    decoded, is handed out.
    """

    def __init__(self, video_dir, data_config: DataConfig, image_size: int, max_videos: int = 2,
                 target_cache_dir="", ):
        self.video_dir = Path(video_dir)
        self.config = data_config
        self.image_size = int(image_size)
        self.max_videos = max(1, int(max_videos))
        self.cache_dir = Path(target_cache_dir) if target_cache_dir else None
        self._records: dict[str, dict] = {}
        self._frames: OrderedDict[str, tuple] = OrderedDict()
        self._verified: dict[str, Path] = {}
        self.decoded_videos = 0     # how many videos were decoded (for summaries)
        self.verified_hashes = 0
        self.disk_cache_hits = 0
        self.disk_cache_misses = 0
        self.disk_cache_writes = 0
        self.disk_cache_entries = 0     # chunk-last frames written to disk
        self.disk_cache_bytes = 0

    # -- counters -------------------------------------------------------------
    def counters(self) -> dict:
        """What this source actually did, for summaries and tests."""
        return {
            "videos_decoded": self.decoded_videos,
            "content_hashes_verified": self.verified_hashes,
            "target_cache_enabled": self.cache_dir is not None,
            "target_cache_dir": str(self.cache_dir) if self.cache_dir else "",
            "target_cache_hits": self.disk_cache_hits,
            "target_cache_misses": self.disk_cache_misses,
            "target_cache_entries": self.disk_cache_entries,
            "target_cache_bytes": self.disk_cache_bytes,
            "max_videos_in_ram": self.max_videos,
        }

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

    # -- source integrity ----------------------------------------------------
    def _verified_path(self, name: str) -> Path:
        """Resolve the source and check its content hash, once per video.

        This runs before *any* target is handed out -- decoded or served from the
        target cache -- so a replaced clip is refused even when every frame is already
        on disk. Hashing reads bytes, it does not decode video.
        """
        if name in self._verified:
            return self._verified[name]
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
        self._verified[name] = path
        return path

    # -- optional on-disk target cache ---------------------------------------
    def cache_tag(self, name: str) -> str:
        """Directory name for one video: sha256 of the full target identity."""
        record = self._records.get(name, {})
        payload = json.dumps(self._identity(name, record), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _identity(self, name: str, record: dict) -> dict:
        """Everything a stored target must agree with to be usable again.

        The chunk records and the sampled frame grid are part of the identity, not just
        the global sampling settings: a cache whose timeline moved (a different
        ``chunk_frames`` boundary, a re-sampled grid, a shifted chunk) must not be able
        to hand back the previous chunk's pixels. Every number is taken through
        :func:`_finite`, so a NaN cannot make two different timelines compare equal.
        """
        chunks = record.get("chunks") or []
        return {
            "semantics": TARGET_SEMANTICS,
            "video": name,
            "source_sha256": record.get("sha256", ""),
            "timeline_schema": record.get("timeline_schema"),
            "source_fps": _finite(record.get("source_fps"), "source_fps", name),
            "fps": _finite(record.get("fps"), "fps", name),
            "chunk_frames": int(record.get("chunk_frames") or 0),
            "chunk_stride_frames": int(record.get("chunk_stride_frames") or 0),
            "token_image_size": int(record.get("image_size") or 0),
            "target_image_size": self.image_size,
            "sampled_frame_indices": [int(value)
                                      for value in (record.get("sampled_frame_indices") or [])],
            "chunks": [
                [int(chunk["index"]), int(chunk["start_frame"]), int(chunk["end_frame"]),
                 round(_finite(chunk["start_seconds"], "chunk start_seconds", name), 9),
                 round(_finite(chunk["end_seconds"], "chunk end_seconds", name), 9)]
                for chunk in chunks
            ],
        }

    def _chunk_identity(self, name: str, chunk: dict) -> dict:
        """The per-chunk slice of the identity: the exact mapping index -> frames/time."""
        return {
            "index": int(chunk["index"]),
            "start_frame": int(chunk["start_frame"]),
            "end_frame": int(chunk["end_frame"]),
            "start_seconds": round(_finite(chunk["start_seconds"], "chunk start_seconds", name), 9),
            "end_seconds": round(_finite(chunk["end_seconds"], "chunk end_seconds", name), 9),
        }

    @staticmethod
    def _frame_digest(frame: torch.Tensor, timestamp: float, chunk_record: dict) -> str:
        """SHA-256 over the stored pixels, their timestamp and the chunk mapping.

        dtype/shape checks cannot tell two valid-shaped frames apart, so the stored
        bytes are hashed and re-checked on read: a silently swapped or corrupted
        keyframe is refused instead of trained on.
        """
        digest = hashlib.sha256()
        digest.update(str(frame.dtype).encode())
        digest.update(str(tuple(frame.shape)).encode())
        digest.update(frame.detach().to("cpu").contiguous().numpy().tobytes())
        digest.update(f"{timestamp!r}".encode())
        digest.update(json.dumps(chunk_record, sort_keys=True).encode())
        return digest.hexdigest()

    def _cache_file(self, name: str) -> Path:
        return self.cache_dir / self.cache_tag(name) / f"{name}.pt"

    def _cache_read(self, name: str) -> dict | None:
        """Load and validate one video's entry, or ``None`` when it is absent."""
        if self.cache_dir is None:
            return None
        path = self._cache_file(name)
        if not path.is_file():
            return None
        try:
            # the entry is plain tensors, floats, ints and strings by construction, so
            # it never needs the pickle-unrestricted loader
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise CacheAlignmentError(
                f"target cache entry {path} cannot be read ({type(error).__name__}: {error}); "
                "delete it or point decoder_train.target_cache_dir at another directory"
            ) from error
        identity = self._identity(name, self._records.get(name, {}))
        if not isinstance(payload, dict) or payload.get("identity") != identity:
            raise CacheAlignmentError(
                f"target cache entry {path} does not describe the current source/sampling/target "
                "identity (stale entry, or the file was copied from another run); delete it or "
                "point decoder_train.target_cache_dir at another directory"
            )
        return payload

    def _cache_write(self, name: str, frames: torch.Tensor, timestamps: torch.Tensor,
                     record: dict) -> None:
        """Store the chunk-last frames of one decoded video (never the whole clip)."""
        if self.cache_dir is None:
            return
        stored, stamps, records, digests = {}, {}, {}, {}
        for chunk in record.get("chunks") or []:
            index = int(chunk["end_frame"]) - 1
            if not 0 <= index < frames.shape[0]:
                continue
            key = int(chunk["index"])
            chunk_record = self._chunk_identity(name, chunk)
            frame = frames[index].clone()
            stamp = float(timestamps[index])
            stored[key] = frame
            stamps[key] = stamp
            records[key] = chunk_record
            digests[key] = self._frame_digest(frame, stamp, chunk_record)
        if not stored:
            return
        path = self._cache_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # a unique temporary name per write: two writers of the same entry must never
        # race on one staging file
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        torch.save({"identity": self._identity(name, record), "frames": stored,
                    "timestamps": stamps, "chunk_records": records, "digests": digests},
                   temporary)
        if not _replace_entry(temporary, path):
            # another writer won the race for the same path. The path is derived from
            # the identity (source hash, timeline, sizes), so the entry already there
            # holds exactly the same frames; leaving it is correct, and the counters
            # below simply do not claim a write that did not happen.
            return
        self.disk_cache_writes += 1
        self.disk_cache_entries += len(stored)
        self.disk_cache_bytes += path.stat().st_size

    def _cached_target(self, name: str, chunk: dict) -> tuple | None:
        """The stored keyframe for one chunk, or ``None`` when it must be decoded."""
        payload = self._cache_read(name)
        if payload is None:
            self.disk_cache_misses += 1
            return None
        index = int(chunk["index"])
        frame = (payload.get("frames") or {}).get(index)
        timestamp = (payload.get("timestamps") or {}).get(index)
        if frame is None or timestamp is None:
            self.disk_cache_misses += 1
            return None
        if not torch.is_tensor(frame) or frame.dtype != torch.uint8 \
                or tuple(frame.shape) != (3, self.image_size, self.image_size):
            raise CacheAlignmentError(
                f"target cache entry {self._cache_file(name)} holds a frame of shape "
                f"{tuple(frame.shape) if torch.is_tensor(frame) else type(frame)} / "
                f"{getattr(frame, 'dtype', None)}; expected (3, {self.image_size}, "
                f"{self.image_size}) uint8"
            )
        # the exact chunk mapping, not just the global settings: a cache written for a
        # different chunk boundary must not supply this chunk's pixels
        chunk_record = self._chunk_identity(name, chunk)
        stored_record = (payload.get("chunk_records") or {}).get(index)
        if stored_record != chunk_record:
            raise CacheAlignmentError(
                f"target cache entry {self._cache_file(name)} maps chunk {index} to "
                f"{stored_record!r}, but the current timeline maps it to {chunk_record!r}; "
                "the entry is stale"
            )
        expected = self._expected_timestamp(name, chunk)
        timestamp = _finite(timestamp, f"cached timestamp of chunk {index}", name)
        if expected is not None and abs(timestamp - expected) > 1e-6:
            raise CacheAlignmentError(
                f"target cache entry {self._cache_file(name)} gives t={timestamp:.9f}s for chunk "
                f"{index}, but the chunk timeline expects {expected:.9f}s; the entry is stale"
            )
        digest = (payload.get("digests") or {}).get(index)
        if digest != self._frame_digest(frame, timestamp, chunk_record):
            raise CacheAlignmentError(
                f"target cache entry {self._cache_file(name)} chunk {index} does not match its "
                "recorded digest: the stored pixels were altered or the file is corrupt"
            )
        self.disk_cache_hits += 1
        return frame, timestamp

    def _expected_timestamp(self, name: str, chunk: dict) -> float | None:
        """The chunk's last-frame timestamp, derived from the cached timeline."""
        source_fps = float(self._records.get(name, {}).get("source_fps") or 0.0)
        if source_fps <= 0:
            return None
        return _finite(chunk.get("end_seconds"), "chunk end_seconds", name) - 1.0 / source_fps

    # -- decoding ------------------------------------------------------------
    def _decode(self, name: str) -> tuple:
        if name in self._frames:
            self._frames.move_to_end(name)
            return self._frames[name]
        path = self._verified_path(name)
        frames, timestamps = read_frames(path, self.config.fps, self.image_size)
        self._check_timeline(name, path, frames, timestamps)
        self._frames[name] = (frames, timestamps)
        self.decoded_videos += 1
        self._cache_write(name, frames, timestamps, self._records.get(name, {}))
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
        # the source is always verified first: a replaced clip is refused even when
        # every keyframe is already sitting in the on-disk cache
        self._verified_path(name)
        cached = self._cached_target(name, chunk)
        if cached is not None:
            return cached
        frames, timestamps = self._decode(name)
        chunk_end = _finite(chunk.get("end_seconds"), f"chunk {chunk.get('index')} end_seconds", name)
        index = int(chunk["end_frame"]) - 1
        if not 0 <= index < frames.shape[0]:
            raise CacheAlignmentError(
                f"{name}: chunk {chunk['index']} ends at sampled frame {index}, outside the "
                f"{frames.shape[0]} frames decoded from the source"
            )
        timestamp = _finite(timestamps[index], f"timestamp of frame {index}", name)
        expected = self._expected_timestamp(name, chunk)
        if expected is not None and abs(timestamp - expected) > 1e-6:
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
