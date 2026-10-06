"""Decoding latents to keyframe PNGs and a tensor artifact.

One helper serves every caller (``predict`` with a video, ``query`` with only a
saved state), so the two paths cannot drift in resolution, colour handling or
metadata. Each decoded image is labelled with the time it refers to and the
horizon it came from: they are separate keyframes ordered by horizon, never a
continuous video, and the pixels come from the latent alone -- no ground-truth
frame ever enters the decoder.
"""

import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from .compat import decoder_fingerprint

SAFE = re.compile(r"[^0-9A-Za-z._-]+")


def safe_name(name: str) -> str:
    """Filesystem-safe stem for a horizon/delta key (``4.0`` -> ``4.0``, ``h 1`` -> ``h_1``)."""
    return SAFE.sub("_", str(name)).strip("_") or "frame"


@torch.no_grad()
def decode_latents(decoder, latents: dict, device) -> dict:
    """Decode ``{key: (P, d_world) latent}`` to ``{key: (3, S, S) float in [0, 1]}``.

    ``device`` applies to **both** sides of the call: the decoder is moved there
    before it runs and the latents are moved to match, so a module loaded on CPU
    decodes GPU-resident latents (and vice versa) without a device mismatch. The
    decoder stays on that device afterwards -- moving it back would silently undo the
    caller's request -- and its training/eval mode is restored, so calling this
    mid-training cannot leave the module in a different state than it found it.
    """
    was_training = decoder.training
    decoder.to(device).eval()
    images = {}
    for key, latent in latents.items():
        batch = latent.unsqueeze(0) if latent.dim() == 2 else latent
        images[key] = decoder(batch.to(device).float())[0].detach().cpu()
    decoder.train(was_training)
    return images


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """``(3, S, S)`` float in [0, 1] -> ``(S, S, 3)`` uint8 RGB (clipped, never wrapped)."""
    array = image.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def save_frame_png(image: torch.Tensor, path: Path) -> Path:
    """Write one RGB keyframe as a PNG through OpenCV (BGR byte order on disk)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), to_uint8(image)[:, :, ::-1]):
        raise RuntimeError(f"could not write {path}")
    return path


@torch.no_grad()
def render_latents(decoder, entries: dict, out_dir, device, payload: dict | None = None,
                   prefix: str = "decoded", name: str = "frames") -> dict:
    """Decode latents, write one PNG each and return the artifact record.

    ``entries`` maps a key to ``(latent (P, d_world), metadata dict)``. The metadata
    (horizon, delta seconds, target time, ...) is copied into the returned record and
    into ``<out_dir>/decoded_summary.json`` next to the images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = decode_latents(decoder, {key: value[0] for key, value in entries.items()}, device)
    names = {}
    for key in images:
        file_name = f"{prefix}_{safe_name(key)}.png"
        if file_name in names:
            # two keys that sanitise to the same file name would silently overwrite
            # one keyframe with another; make the caller disambiguate instead
            raise ValueError(
                f"decode keys {names[file_name]!r} and {key!r} both map to {file_name!r}; "
                "use keys that differ by more than punctuation"
            )
        names[file_name] = key
    records, files = {}, {}
    for key, image in images.items():
        meta = dict(entries[key][1])
        file_name = f"{prefix}_{safe_name(key)}.png"
        save_frame_png(image, out_dir / file_name)
        records[str(key)] = {**meta, "png": file_name, "range": "[0, 1]"}
        files[str(key)] = image
    artifact = {
        "kind": "rgb_decoder_frames",
        "decoder": decoder_fingerprint(payload or {}),
        "decoder_checkpoint": (payload or {}).get("path"),
        "name": name,
        "frames": files,
        "records": records,
        "note": (
            "each frame is the decoder's output for one latent; frames are separate keyframes "
            "ordered by horizon/target time, not a continuous video"
        ),
    }
    return artifact


def write_artifact(path, artifact: dict, meta: dict | None = None) -> None:
    """Save the decoded tensors plus their metadata next to the PNGs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**artifact, "meta": dict(meta or {})}, path)


def write_records(path, artifact: dict, meta: dict | None = None) -> None:
    """JSON view of a rendered artifact: file names and timestamps, no tensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decoder": artifact.get("decoder"),
        "decoder_checkpoint": artifact.get("decoder_checkpoint"),
        "note": artifact.get("note"),
        "frames": artifact.get("records", {}),
        "meta": dict(meta or {}),
    }
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
