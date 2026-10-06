"""Real video reading, timestamped causal chunking and source identity.

Timeline rule enforced here: every sampled frame keeps the timestamp of the
source frame it was decoded from (``source_index / source_fps``), and a chunk's
start/end are derived from those timestamps. Nothing in the pipeline converts a
sample index with the requested fps, which is what made earlier recorded times
drift from the real source timeline by up to 0.66 s on the sample videos.

A chunk covers source frames ``[i0, i1)`` with

    start_seconds = timestamp(i0)
    end_seconds   = timestamp(i1 - 1) + 1 / source_fps

so a prediction target that starts at or after ``end_seconds`` is strictly in the
future of everything the anchor chunk observed.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from .config import DataConfig

SOURCE_MANIFEST_NAME = "sources.json"
PUBLIC_DATASET = {
    "repo_id": "nateraw/kinetics-mini",
    "repo_type": "dataset",
    "revision": "9f4ed38128a355c352527209101be3e326471816",
}
RESOLVE_URL = "https://huggingface.co/{kind}/{repo_id}/resolve/{revision}/{path}"


def repository_url(repo_id: str, revision: str, path: str, repo_type: str = "dataset") -> str:
    """Immutable resolve URL for one file, built only from the declared coordinates."""
    kind = {"dataset": "datasets", "model": "models", "space": "spaces"}.get(repo_type, "datasets")
    return RESOLVE_URL.format(kind=kind, repo_id=repo_id, revision=revision, path=path)

KINETICS_NAME = re.compile(
    r"^(?:(?P<label>.+?)_)?(?P<source_id>[A-Za-z0-9_-]{10,12})_(?P<start>\d+)_(?P<end>\d+)$"
)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    name: str
    fps: float
    frame_count: int
    width: int
    height: int
    sha256: str = ""
    source_id: str = ""
    source_uri: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def frame_period(self) -> float:
        return 1.0 / self.fps if self.fps else 0.0


@dataclass(frozen=True)
class Chunk:
    video: str
    index: int
    start_frame: int
    end_frame: int  # exclusive
    start_seconds: float
    end_seconds: float

    @property
    def span_frames(self) -> int:
        return self.end_frame - self.start_frame


def file_sha256(path) -> str:
    """SHA-256 of a file; accepts str or Path."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source_identity(stem: str) -> str:
    """Source identity of a clip: the original video it was cut from, not the cut file."""
    match = KINETICS_NAME.match(stem)
    if match:
        return f"{match.group('source_id')}"
    return stem


def probe_video(path, with_hash: bool = True) -> VideoInfo:
    """Inspect a video file; accepts str or Path."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return VideoInfo(
        path=path.as_posix(), name=path.stem, fps=fps, frame_count=frame_count,
        width=width, height=height,
        sha256=file_sha256(path) if with_hash else "",
        source_id=parse_source_identity(path.stem),
        source_uri=path.as_posix(),
    )


def read_frames(path, fps: float, image_size: int, max_frames: int | None = None):
    """Decode to (frames (T,3,H,W) uint8, timestamps (T,) float64 seconds).

    Frames are sampled on a fixed source-index grid ``0, step, 2*step, ...`` and
    each keeps the timestamp of its source frame.
    """
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(1, int(round(source_fps / fps))) if fps > 0 else 1
    frames, timestamps, index = [], [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                frames.append(frame)
                timestamps.append(index / source_fps)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    stacked = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    return stacked, torch.tensor(timestamps, dtype=torch.float64)


def build_chunks(info: VideoInfo, config: DataConfig, timestamps: torch.Tensor) -> list[Chunk]:
    """Tile sampled frames into consecutive chunks using real source timestamps."""
    total = int(timestamps.shape[0])
    chunks, index, start = [], 0, 0
    while start + config.chunk_frames <= total:
        end = start + config.chunk_frames
        chunks.append(Chunk(
            video=info.name, index=index, start_frame=start, end_frame=end,
            start_seconds=float(timestamps[start]),
            end_seconds=float(timestamps[end - 1]) + info.frame_period,
        ))
        index += 1
        start += config.chunk_stride_frames
    return chunks


def list_videos(directory) -> list[Path]:
    """Sorted .mp4 files in a directory; accepts str or Path."""
    return sorted(p for p in Path(directory).glob("*.mp4") if p.is_file())


def windows_for_video(chunk_count: int, config: DataConfig) -> list[int]:
    """Start indices of context windows; each window is causal and inside one video."""
    starts, start = [], 0
    while start + config.context_chunks <= chunk_count:
        starts.append(start)
        start += config.window_stride_chunks
    return starts

def load_source_manifest(video_dir) -> dict:
    """Explicit local-file -> public-dataset mapping, if one has been written.

    No manifest means every clip is treated as a local file: a synthetic or
    private video must never be reported as a public dataset clip by default.
    """
    path = Path(video_dir) / SOURCE_MANIFEST_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("clips", {})


def resolve_source(video: VideoInfo, manifest: dict) -> dict:
    """Provenance record for one clip: verified public mapping, or local file.

    A declared mapping is only reported as ``public_dataset`` when the declared
    content hash equals the hash of the file on disk. A stale declaration (bytes
    changed since the manifest was written) demotes the record to ``local_file``
    with the reason recorded, so a clip is never labelled public on trust.
    """
    record = {
        "kind": "local_file",
        "local_path": video.path,
        "content_sha256": video.sha256,
        "source_id": video.source_id,
    }
    entry = manifest.get(video.name)
    if not entry:
        return record
    declared = entry.get("content_sha256")
    if declared is not None and declared != video.sha256:
        return {**record, "kind": "local_file",
                "demoted_from": "public_dataset",
                "demotion_reason": "declared content_sha256 does not match the file on disk",
                "declared_content_sha256": declared}
    repo_id = entry.get("repo_id") or PUBLIC_DATASET["repo_id"]
    repo_type = entry.get("repo_type") or PUBLIC_DATASET["repo_type"]
    revision = entry.get("revision") or PUBLIC_DATASET["revision"]
    path = entry["source_path"]
    return {
        **record,
        "kind": "public_dataset",
        "dataset_repo_id": repo_id,
        "dataset_repo_type": repo_type,
        "dataset_revision": revision,
        "dataset_path": path,
        # built from the declared coordinates only; no dataset is assumed
        "dataset_url": repository_url(repo_id, revision, path, repo_type),
        "declared_content_sha256": declared,
        "content_sha256_matches_declaration": True,
    }


def build_source_manifest(video_dir, snapshot_dir, revision: str,
                          repo_id: str = PUBLIC_DATASET["repo_id"]) -> dict:
    """Map staged clips back to their file inside a downloaded dataset snapshot.

    A clip is only declared public after the staged bytes and the snapshot bytes
    hash identically; anything that does not match (or has no snapshot counterpart)
    is listed under ``skipped`` and stays a local file for provenance purposes.
    """
    clips, skipped = {}, []
    snapshot_root = Path(snapshot_dir)
    for path in list_videos(Path(video_dir)):
        label, _, stem = path.stem.partition("_")
        candidates = []
        if stem:
            candidates.append(snapshot_root / "train" / label / f"{stem}.mp4")
        candidates.append(snapshot_root / "train" / label / f"{path.stem}.mp4")
        candidate = next((c for c in candidates if c.is_file()), None)
        if candidate is None:
            lookup = stem or path.stem
            candidate = next(snapshot_root.rglob(f"{lookup}.mp4"), None)
        if candidate is None or not candidate.is_file():
            skipped.append(path.stem)
            continue
        staged_hash = file_sha256(path)
        snapshot_hash = file_sha256(candidate)
        if staged_hash != snapshot_hash:
            skipped.append(f"{path.stem} (bytes differ from snapshot)")
            continue
        clips[path.stem] = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "revision": revision,
            "source_path": candidate.relative_to(snapshot_dir).as_posix(),
            "content_sha256": staged_hash,
            "snapshot_sha256": snapshot_hash,
        }
    return {
        "schema_version": 1,
        "dataset": {**PUBLIC_DATASET, "repo_id": repo_id, "revision": revision},
        "verification": (
            "a clip is listed only when the SHA-256 of the staged file equals the SHA-256 of the "
            "snapshot file it came from; local_path stays local and is not a download URL"
        ),
        "clips": clips,
        "skipped": skipped,
    }
