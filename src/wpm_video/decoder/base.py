"""Public contract of a replaceable RGB decoder component.

A decoder turns the projected world latents of **one chunk** -- ``(B, P, d_world)``
with their ``(gh, gw)`` patch layout -- into one RGB keyframe ``(B, 3, S, S)`` in
``[0, 1]``. Each output is the last sampled frame of that chunk, not a synthesis of
every frame in it, and a sequence of decoded horizons stays a set of separate
keyframes rather than a clip.

Everything the rest of the package relies on is declared here, and nothing about a
particular network is: the base class does not know about convolutions, stems,
channel widths or upsampling factors. It owns the shared geometry (patch grid,
latent width, output edge) and the input contract; an architecture adds its own
parameters and its own constraints on top. A component is built by a registered
factory (see :mod:`wpm_video.decoder.registry`) and stored in a checkpoint together
with the ``kind`` that produced it, so training, evaluation, resuming and rendering
all work the same way for any implementation.
"""

import copy
import math

import torch
from torch import nn

from ..config import DecoderConfig, validate_decoder_config


def grid_side(patches: int) -> int:
    """Edge of the square patch grid of ``patches`` latents, refusing non-squares."""
    if type(patches) is not int or patches < 1:
        raise ValueError("patches must be a positive integer")
    side = int(round(math.sqrt(patches)))
    if side * side != patches:
        raise ValueError(f"patches must be a perfect square, got {patches}")
    return side


def validate_latent_geometry(config: DecoderConfig, patches: int, d_world: int) -> int:
    """Shared geometry rules; returns the patch grid side.

    These describe the *data* every decoder consumes, not how it turns latents into
    pixels: the world model fixes a square patch grid and a positive latent width,
    and ``image_size`` is the common output edge (see
    :func:`~wpm_video.config.validate_decoder_config` for its range).

    How the output is produced -- whether it must cover the grid by a whole factor,
    be a power-of-two multiple, or anything else -- is the architecture's business
    and is checked by the architecture.
    """
    validate_decoder_config(config)
    side = grid_side(patches)
    if type(d_world) is not int or d_world < 1:
        raise ValueError("d_world must be a positive integer")
    return side


def _check_rgb_output(output, batch_size: int, image_size: int | None = None,
                      *, check_values: bool = False) -> None:
    """Check RGB geometry without synchronising; value checks are for CPU rendering."""
    if not torch.is_tensor(output) or output.dim() != 4 or not output.is_floating_point():
        raise ValueError("decoder output must be a floating (B, 3, S, S) tensor")
    batch, channels, height, width = output.shape
    if (batch != batch_size or channels != 3 or height != width or height < 1
            or (image_size is not None and height != image_size)):
        raise ValueError(
            f"decoder output must have shape ({batch_size}, 3, S, S)"
            f" with S={image_size or 'a positive edge'}, got {tuple(output.shape)}"
        )
    if check_values and (not torch.isfinite(output).all()
                         or (output < 0).any() or (output > 1).any()):
        raise ValueError("decoder output must contain finite RGB values in [0, 1]")


class RGBDecoder(nn.Module):
    """Base class of every RGB decoder component: ``(B, P, d_world)`` -> ``(B, 3, S, S)``.

    Subclasses call ``super().__init__(config, patches, d_world)`` first -- it
    validates the shared geometry and stores it -- and then build their own
    parameters and implement ``forward``. The base deliberately assumes nothing
    about how the latent becomes pixels, so a component may be convolutional, an
    MLP over latent cells, a pixel-shuffle network or anything else that honours the
    contract below.

    Contract:

    - input is a floating point ``(B, P, d_world)`` tensor of projected latents of
      one chunk, ``P`` and ``d_world`` matching the world model (checked by
      :meth:`check_input`); the module reads **no other state** -- no world state, no
      timestamps, no files;
    - output is ``(B, 3, S, S)`` with ``S == output_size``, RGB in ``[0, 1]``;
    - the architecture is fully described by ``config`` (``kind``, ``image_size``
      and ``options``), which is what a checkpoint records and compares.

    Layers whose behaviour depends on the module mode (batch normalisation, dropout)
    are allowed: rendering and evaluation run a decoder in ``eval()`` mode
    (``decode_latents`` switches it and restores the mode it found), while training
    runs it in ``train()`` mode -- so a mode-dependent decoder is legitimate, it just
    has to be *used* in the mode that is meant. The built-in conv decoder has no such
    layer and produces identical pixels in both modes.
    """

    def __init__(self, config: DecoderConfig, patches: int, d_world: int):
        super().__init__()
        if not isinstance(config, DecoderConfig):
            raise ValueError(
                f"a decoder is built from a DecoderConfig, got {type(config).__name__}"
            )
        side = validate_latent_geometry(config, patches, d_world)
        # The configuration is copied, not referenced: a module (and the checkpoint
        # written from it) must keep describing the architecture it was built with,
        # even if the caller keeps editing the configuration it passed in.
        self.config = copy.deepcopy(config)
        self.patches = patches
        self.d_world = d_world
        self.grid = (side, side)

    @property
    def kind(self) -> str:
        """Registered component name that built (and identifies) this decoder."""
        return self.config.kind

    @property
    def output_size(self) -> int:
        """Edge ``S`` of the square RGB output."""
        return int(self.config.image_size)

    @property
    def grid_side(self) -> int:
        """Edge of the square patch grid (``gh == gw``)."""
        return self.grid[0]

    def parameter_count(self) -> int:
        """Total number of parameters (the same number checkpoints record)."""
        return sum(parameter.numel() for parameter in self.parameters())

    def check_input(self, latents) -> None:
        """Enforce the input contract without a device synchronisation.

        Subclasses may extend this (calling ``super().check_input(latents)`` first)
        when they have additional input requirements; they must not weaken it.
        """
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

    def forward(self, latents):  # pragma: no cover - abstract contract
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(latents) -> (B, 3, S, S) in [0, 1]; "
            "see wpm_video.decoder.base.RGBDecoder for the contract"
        )
