"""The built-in convolutional RGB decoder (``kind = "conv"``).

This is the original and default decoder: a compact convolutional upsampler. Its
architecture, parameter names and initialisation order are frozen -- checkpoints
written before decoder components existed (schema 1) describe exactly this network,
so the class body below is a move of the historical implementation, not a rewrite.

The module is independent of the dynamics: it consumes latents and knows nothing
about state, time or the predictor. It is also lossy by construction, because it
inverts a frozen encoder plus a random projection that was never trained to be
invertible. Expect blurry output and missing texture; this is a reconstruction aid,
not a photorealistic generator.
"""

import math

import torch
from torch import nn

from ...config import CONV_DECODER_KIND, DecoderConfig
from ..base import RGBDecoder
from ..registry import register_decoder


def group_count(channels: int) -> int:
    """Largest usable GroupNorm group count for a channel width (1 when unusual)."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(c_in: int, c_out: int) -> list:
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(group_count(c_out), c_out), nn.SiLU()]


class LatentRGBDecoder(RGBDecoder):
    """Compact convolutional upsampler from a patch-latent grid to RGB pixels.

    The latent is reshaped to its ``(gh, gw)`` grid (row-major patch order), a
    learned position embedding is added, then ``stem_blocks`` convolutions run at
    grid resolution and ``log2(image_size / grid_side)`` nearest-neighbour
    upsampling stages double the resolution until it reaches ``image_size``. A
    3x3 convolution maps to three channels and a sigmoid bounds the output.
    """

    def __init__(self, config: DecoderConfig, patches: int, d_world: int):
        # The module is a public entry point: the base class validates the shared
        # geometry (and the configuration) before anything is allocated.
        super().__init__(config, patches, d_world)
        if config.kind != CONV_DECODER_KIND:
            raise ValueError(
                f"LatentRGBDecoder implements kind {CONV_DECODER_KIND!r}, but the configuration "
                f"declares kind {config.kind!r}; build a custom architecture through its own "
                "registered factory instead"
            )
        # This architecture upsamples the latent grid by doubling it at each stage, so
        # the output must cover the grid by a whole power of two. Those constraints
        # belong to the conv stack, not to the decoder contract: an architecture that
        # resizes or reshapes differently may accept other output sizes.
        if config.image_size < self.grid_side:
            raise ValueError(
                f"decoder.image_size ({config.image_size}) must not be smaller than the patch "
                f"grid side ({self.grid_side}); every latent cell has to cover at least one pixel"
            )
        if config.image_size % self.grid_side:
            raise ValueError(
                f"decoder.image_size ({config.image_size}) must be a multiple of the patch grid "
                f"side ({self.grid_side}); the decoder upsamples the latent grid by a whole factor"
            )
        factor = config.image_size // self.grid_side
        if factor & (factor - 1):
            raise ValueError(
                f"decoder.image_size / patch grid side ({config.image_size}/{self.grid_side} = "
                f"{factor}) must be a power of two so every upsampling stage is exact"
            )
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
        self.position = nn.Parameter(torch.randn(1, d_world, self.grid_side, self.grid_side) * 0.02)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.check_input(latents)
        rows, cols = self.grid
        hidden = latents.transpose(1, 2).reshape(latents.shape[0], self.d_world, rows, cols)
        hidden = hidden + self.position.to(hidden.dtype)
        hidden = self.stem(hidden)
        for stage in self.stages:
            hidden = stage(hidden)
        return torch.sigmoid(self.head(hidden))


def build_conv_decoder(config: DecoderConfig, patches: int, d_world: int) -> LatentRGBDecoder:
    """Factory of the built-in conv architecture (the registry entry for ``"conv"``)."""
    return LatentRGBDecoder(config, patches, d_world)


register_decoder(CONV_DECODER_KIND, build_conv_decoder)
