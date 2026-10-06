"""Optional RGB decoder: projected world latents of one chunk -> one RGB keyframe.

The decoder turns the projected latent of a chunk, ``(B, P, d_world)`` with its
``(gh, gw)`` patch layout, into ``(B, 3, S, S)`` pixels in ``[0, 1]`` with a small
convolutional upsampler. Each output is **one keyframe**: the last sampled frame
of that chunk. It is not a synthesis of every frame in the chunk and not a
high-frame-rate video; a sequence of decoded horizons is a sparse set of
keyframes, never a continuous clip.

The decoder is deliberately independent of the dynamics: it consumes latents and
knows nothing about state, time or the predictor. It is also lossy by
construction, because it inverts a frozen encoder plus a random projection that
was never trained to be invertible. Expect blurry output and missing texture;
this is a reconstruction aid, not a photorealistic generator. Nothing here should
be called generated RGB until a decoder checkpoint trained on real data is
supplied and its held-out metrics are reported.
"""

import math
from pathlib import Path

import torch
from torch import nn

from ..config import DecoderConfig, decoder_architecture, validate_decoder_config

# Checkpoint schema of this decoder. Bumping it invalidates older checkpoints
# instead of silently reading them with new expectations.
DECODER_SCHEMA_VERSION = 1


def group_count(channels: int) -> int:
    """Largest usable GroupNorm group count for a channel width (1 when unusual)."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(c_in: int, c_out: int) -> list:
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(group_count(c_out), c_out), nn.SiLU()]


class LatentRGBDecoder(nn.Module):
    """Compact convolutional upsampler from a patch-latent grid to RGB pixels.

    The latent is reshaped to its ``(gh, gw)`` grid (row-major patch order), a
    learned position embedding is added, then ``stem_blocks`` convolutions run at
    grid resolution and ``log2(image_size / grid_side)`` nearest-neighbour
    upsampling stages double the resolution until it reaches ``image_size``. A
    3x3 convolution maps to three channels and a sigmoid bounds the output.
    """

    def __init__(self, config: DecoderConfig, patches: int, d_world: int):
        super().__init__()
        # The module is a public entry point: validate the architecture here, not only
        # through RunConfig, so a hand-built DecoderConfig cannot build a network whose
        # channel shapes are wrong (e.g. zero stem blocks would drop the width change).
        validate_decoder_config(config)
        if type(patches) is not int or patches < 1:
            raise ValueError("patches must be a positive integer")
        side = int(round(math.sqrt(patches)))
        if side * side != patches:
            raise ValueError(f"patches must be a perfect square, got {patches}")
        if type(d_world) is not int or d_world < 1:
            raise ValueError("d_world must be a positive integer")
        if config.image_size < side:
            raise ValueError(
                f"decoder.image_size ({config.image_size}) must not be smaller than the patch "
                f"grid side ({side}); every latent cell has to cover at least one pixel"
            )
        if config.image_size % side:
            raise ValueError(
                f"decoder.image_size ({config.image_size}) must be a multiple of the patch grid "
                f"side ({side}); the decoder upsamples the latent grid by a whole factor"
            )
        factor = config.image_size // side
        if factor & (factor - 1):
            raise ValueError(
                f"decoder.image_size / patch grid side ({config.image_size}/{side} = {factor}) "
                "must be a power of two so every upsampling stage is exact"
            )
        if not config.channel_multipliers:
            raise ValueError("decoder.channel_multipliers must not be empty")
        self.config = config
        self.patches = patches
        self.d_world = d_world
        self.grid = (side, side)
        self.stages_count = int(math.log2(factor))
        base = config.base_channels
        widths = [base * multiplier for multiplier in config.channel_multipliers]
        self.stem = nn.Sequential(*[
            layer for index in range(config.stem_blocks)
            for layer in _conv_block(d_world if index == 0 else base, base)
        ])
        stages, c_in = [], base
        for index in range(self.stages_count):
            c_out = widths[min(index, len(widths) - 1)]
            layers = [nn.Upsample(scale_factor=2, mode="nearest")]
            for block in range(config.blocks_per_stage):
                layers += _conv_block(c_in if block == 0 else c_out, c_out)
            stages.append(nn.Sequential(*layers))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv2d(c_in, 3, 3, padding=1)
        nn.init.zeros_(self.head.bias)
        # Position embedding: the projection is content-blind, so the grid offset
        # is the only place the decoder learns *where* a patch sits.
        self.position = nn.Parameter(torch.randn(1, d_world, side, side) * 0.02)

    @property
    def output_size(self) -> int:
        return self.config.image_size

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def check_input(self, latents: torch.Tensor) -> None:
        """Shape contract, checked without a device synchronisation."""
        if not torch.is_tensor(latents) or latents.dim() != 3:
            raise ValueError(f"decoder input must be a (B, P, d_world) tensor, got {type(latents)}")
        batch, patches, width = latents.shape
        if patches != self.patches or width != self.d_world:
            raise ValueError(
                f"decoder expects latents of shape (B, {self.patches}, {self.d_world}) for the "
                f"checkpoint's patch grid, got (B, {patches}, {width})"
            )
        if batch < 1:
            raise ValueError("decoder input must have at least one sample")
        if not latents.is_floating_point():
            raise ValueError(f"decoder input must be a floating point tensor, got {latents.dtype}")

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.check_input(latents)
        rows, cols = self.grid
        hidden = latents.transpose(1, 2).reshape(latents.shape[0], self.d_world, rows, cols)
        hidden = hidden + self.position.to(hidden.dtype)
        hidden = self.stem(hidden)
        for stage in self.stages:
            hidden = stage(hidden)
        return torch.sigmoid(self.head(hidden))


def build_decoder(config: DecoderConfig, patches: int, d_world: int) -> LatentRGBDecoder:
    """Construct a decoder for a world model with ``patches`` and ``d_world``."""
    return LatentRGBDecoder(config, patches, d_world)


def save_decoder(path, decoder: LatentRGBDecoder, extra: dict | None = None) -> None:
    """Write a decoder checkpoint; ``extra`` carries the identity and provenance."""
    payload = {
        "kind": "rgb_decoder",
        "schema_version": DECODER_SCHEMA_VERSION,
        "decoder_config": vars(decoder.config),
        "architecture": decoder_architecture(decoder.config),
        "patches": decoder.patches,
        "d_world": decoder.d_world,
        "grid": list(decoder.grid),
        "state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_decoder(path, map_location="cpu"):
    """Load a decoder checkpoint; returns ``(decoder, payload)``.

    The returned ``payload["path"]`` is the absolute resolved path this checkpoint was
    read from, so artifacts written from it name the decoder that produced them.

    The checkpoint must be a decoder of the current schema; a world-model checkpoint,
    a truncated file or a future schema is refused with a readable message rather than
    loaded partially.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "rgb_decoder":
        raise ValueError(
            f"{path} is not an RGB decoder checkpoint (expected kind='rgb_decoder'); a world "
            "model checkpoint cannot be used as --decoder-checkpoint"
        )
    version = payload.get("schema_version")
    if version != DECODER_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has decoder schema_version {version!r}, this package writes and reads "
            f"{DECODER_SCHEMA_VERSION}; retrain the decoder with this version"
        )
    for field in ("decoder_config", "patches", "d_world", "state_dict"):
        if field not in payload:
            raise ValueError(f"{path} is missing '{field}'; it is not a complete decoder checkpoint")
    decoder = LatentRGBDecoder(DecoderConfig(**payload["decoder_config"]), payload["patches"],
                               payload["d_world"])
    decoder.load_state_dict(payload["state_dict"])
    # where this checkpoint was actually read from, so artifacts and summaries can
    # name it; the saved file never records its own location, the loader adds it
    payload["path"] = str(Path(path).resolve())
    return decoder, payload
