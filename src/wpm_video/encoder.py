"""Video encoders.

Two backends share one interface:

- ``VJEPA2Encoder``: the real, publicly pretrained Meta V-JEPA 2 ViT-L/16 model,
  loaded from a pinned Hub revision and frozen. This is the encoder used in every
  real run.
- ``NativeEncoder``: a small randomly initialized 3D-convolution encoder used
  only by offline unit tests. A real run refuses to fall back to it unless the
  configuration explicitly allows that, so a demo can never silently present
  random weights as a pretrained model.

For both backends ``encode_clip`` sees only the frames of one chunk, so tokens
carry no information from later chunks.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import DataConfig, EncoderConfig

VJEPA2_MEAN = (0.485, 0.456, 0.406)
VJEPA2_STD = (0.229, 0.224, 0.225)


class Encoder:
    """Interface: ``encode_clip`` maps (T, 3, H, W) uint8 frames to (P, D) tokens.

    ``encode_clips`` is the batch entry point used by caching: it takes
    ``(B, T, 3, H, W)`` and returns a list of ``B`` ``(P, D)`` token tensors, one per
    clip, in input order. Each clip still covers exactly one chunk's frames, so no
    chunk ever sees another chunk's future. Backends that can process a real batch
    override it; the base implementation is a serial fallback, which keeps
    third-party encoders that only define ``encode_clip`` working unchanged.
    """

    kind = "base"
    is_pretrained = False
    model_id = "none"
    revision = None

    def __init__(self, d_model: int, grid: tuple[int, int]):
        self.d_model = d_model
        self.grid = grid

    @property
    def patches(self) -> int:
        return self.grid[0] * self.grid[1]

    def encode_clip(self, clip: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_clips(self, clips: torch.Tensor) -> list:
        """Batched encoding; the default is a safe serial fallback per clip."""
        check_clips(clips)
        return [self.encode_clip(clips[index]) for index in range(clips.shape[0])]


def check_clips(clips: torch.Tensor) -> None:
    """Shared shape contract for ``encode_clips``: (B, T, 3, H, W) uint8."""
    if not torch.is_tensor(clips) or clips.dim() != 5:
        raise ValueError(
            f"encode_clips expects a (B, T, 3, H, W) uint8 tensor, got "
            f"{tuple(clips.shape) if torch.is_tensor(clips) else type(clips)}"
        )
    if clips.shape[0] < 1:
        raise ValueError("encode_clips needs at least one clip")
    if clips.shape[2] != 3:
        raise ValueError(f"encode_clips expects 3 colour channels, got {clips.shape[2]}")
    if clips.dtype != torch.uint8:
        raise ValueError(f"encode_clips expects uint8 frames, got {clips.dtype}")


class VJEPA2Encoder(Encoder):
    """Frozen V-JEPA 2 ViT-L/16 loaded through transformers."""

    kind = "vjepa2"
    is_pretrained = True

    def __init__(self, config: EncoderConfig, device: torch.device):
        from transformers import VJEPA2Model

        self.config = config
        self.device = device
        self.model_id = config.model_id
        self.model = VJEPA2Model.from_pretrained(config.model_id, revision=config.revision)
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        patch = self.model.config.patch_size
        image = self.model.config.image_size
        super().__init__(self.model.config.hidden_size, (image // patch, image // patch))
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.revision
        self.tubelet = self.model.config.tubelet_size
        # The normalisation constants never change, so they are built once here
        # instead of per clip; the shape broadcasts over (B, T, C, H, W).
        self.register_normalisation()

    def register_normalisation(self) -> None:
        self.mean = torch.tensor(VJEPA2_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(VJEPA2_STD, device=self.device).view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def encode_clips(self, clips: torch.Tensor) -> list:
        """(B, T, 3, H, W) uint8 -> B (P, D) token tensors of the last temporal position."""
        check_clips(clips)
        pixels = clips.to(self.device).float().div_(255.0)
        pixels = (pixels - self.mean) / self.std
        if pixels.shape[1] % self.tubelet:
            pixels = pixels[:, : pixels.shape[1] - (pixels.shape[1] % self.tubelet)]
        output = self.model(pixel_values_videos=pixels, skip_predictor=True)
        tokens = output.last_hidden_state                       # (B, T' * P, D)
        temporal = tokens.shape[1] // self.patches
        per_clip = tokens.view(tokens.shape[0], temporal, self.patches, self.d_model)
        return list(per_clip[:, -1].contiguous())

    @torch.no_grad()
    def encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """(T, 3, H, W) uint8 -> (P, D) tokens of the last temporal position."""
        return self.encode_clips(clip.unsqueeze(0))[0]


class NativeEncoder(Encoder, nn.Module):
    """Randomly initialized patch encoder for offline tests only."""

    kind = "native"
    is_pretrained = False

    model_id = "native"
    revision = "native"

    def __init__(self, d_model: int = 64, image_size: int = 64, patch: int = 16, tubelet: int = 2, seed: int = 0):
        nn.Module.__init__(self)
        Encoder.__init__(self, d_model, (image_size // patch, image_size // patch))
        generator = torch.Generator().manual_seed(seed)
        weight = torch.empty(d_model, 3, tubelet, patch, patch)
        bound = (6.0 / (3 * tubelet * patch * patch)) ** 0.5
        weight.uniform_(-bound, bound, generator=generator)
        self.projection = nn.Parameter(weight, requires_grad=False)
        self.tubelet = tubelet

    @torch.no_grad()
    def encode_clips(self, clips: torch.Tensor) -> list:
        """(B, T, 3, H, W) uint8 -> B (P, D) token tensors, one batched conv3d call."""
        check_clips(clips)
        pixels = clips.float().div_(255.0).permute(0, 2, 1, 3, 4)   # (B, 3, T, H, W)
        if pixels.shape[2] % self.tubelet:
            pixels = pixels[:, :, : pixels.shape[2] - (pixels.shape[2] % self.tubelet)]
        stride = (self.tubelet, pixels.shape[3] // self.grid[0], pixels.shape[4] // self.grid[1])
        features = torch.nn.functional.conv3d(pixels, self.projection, stride=stride)
        # (B, D, T', gh, gw) -> last temporal position -> (B, P, D)
        stacked = features[:, :, -1].flatten(2).transpose(1, 2).contiguous()
        return list(stacked)

    @torch.no_grad()
    def encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        return self.encode_clips(clip.unsqueeze(0))[0]


def build_encoder(config: EncoderConfig, device: torch.device, allow_native: bool = False) -> Encoder:
    if config.kind == "vjepa2":
        return VJEPA2Encoder(config, device)
    if config.kind == "native":
        if not (allow_native or config.allow_native_fallback):
            raise RuntimeError(
                "The native encoder is randomly initialized and must not back a real run; "
                "set encoder.allow_native_fallback for offline tests only"
            )
        return NativeEncoder()
    raise ValueError(f"Unknown encoder kind: {config.kind}")


class TokenCache:
    """Cached encoder tokens for one video, with provenance."""

    def __init__(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.tokens = payload["tokens"].float()  # (chunks, P, D)
        self.chunks = payload["chunks"]
        self.meta = payload["meta"]

    @staticmethod
    def save(path: Path, tokens: torch.Tensor, chunks: list, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tokens": tokens.to(torch.float16), "chunks": chunks, "meta": meta}, path)

    def __len__(self) -> int:
        return self.tokens.shape[0]


def cache_identity(encoder) -> tuple:
    """(kind, model_id, revision) for cache naming, from an encoder or its config.

    An ``EncoderConfig`` exposes an ``identity`` property that neutralises the
    pretrained fields for the native backend; using it here keeps every caller on
    the same key, so a cache written through one path is always found by another.
    """
    identity = getattr(encoder, "identity", None)
    if identity is not None:
        kind = encoder.kind
        model_id, revision = identity
        return kind, model_id, revision
    return (getattr(encoder, "kind", "native"), getattr(encoder, "model_id", "native"),
            getattr(encoder, "revision", "native"))


def cache_path(cache_dir: Path, video_name: str, config: DataConfig, encoder) -> Path:
    """Cache identity: timeline schema, encoder revision, sampling and chunking.

    The timeline schema is part of the key because caches written before the
    source-timestamp fix must never be reused silently.
    """
    from .dataset import TIMELINE_SCHEMA

    kind, model_id, revision = cache_identity(encoder)
    tag = (f"{TIMELINE_SCHEMA}-{kind}-{Path(model_id).name}-{str(revision)[:8]}-{config.fps}-"
           f"{config.chunk_frames}-{config.chunk_stride_frames}-{config.image_size}")
    return Path(cache_dir) / tag / f"{video_name}.pt"
