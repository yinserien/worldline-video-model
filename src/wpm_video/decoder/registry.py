"""Process-local registry of RGB decoder architectures.

A decoder is selected by the ``kind`` field of :class:`~wpm_video.config.DecoderConfig`
and built by the factory registered under that name. The built-in convolutional
architecture is registered when :mod:`wpm_video.decoder` is imported; a custom
implementation lives outside the package, is imported by the program that uses it,
and registers itself before any build, load or training call::

    import torch
    from torch import nn

    from wpm_video.decoder import RGBDecoder, register_decoder

    class MyDecoder(RGBDecoder):
        def __init__(self, config, patches, d_world):
            super().__init__(config, patches, d_world)
            self.head = nn.Linear(d_world, 3 * self.output_size ** 2)

        def forward(self, latents):                      # (B, P, d_world)
            self.check_input(latents)
            pooled = latents.mean(dim=1)                 # (B, d_world)
            pixels = self.head(pooled)                   # (B, 3 * S * S)
            return torch.sigmoid(pixels).reshape(-1, 3, self.output_size, self.output_size)

    register_decoder("my_decoder_v1", MyDecoder)

Registration is deliberately **not** persistent and **not** discovered: it is
process-local, so a checkpoint can only be loaded where its kind is registered;
nothing is imported or downloaded because a checkpoint or a configuration names a
kind (an unknown kind is an error, never a lookup); and re-registering a name is
refused, including the built-in ``"conv"``.

A kind names an *implementation*, semantics included -- not just a tensor shape.
Keeping one name for a changed network would make old checkpoints load into a
different model, so an author publishes a new name instead (``my_decoder_v2``) and
keeps the old registration to read the old weights.

The stock CLI (`wpm-video train-decoder`, `predict`, `query`) therefore uses the
built-in conv architecture unless the process that calls it registered another
component first -- for example a small script that imports the architecture, calls
``register_decoder`` and then calls ``wpm_video.cli.main``.
"""

from collections.abc import Callable

from ..config import DecoderConfig, validate_decoder_config, validate_decoder_kind
from .base import RGBDecoder, grid_side

# kind -> factory(config, patches, d_world) -> RGBDecoder. Rebound by
# register_decoder; never edited from outside this module.
DECODER_FACTORIES: dict = {}


class DecoderRegistrationError(ValueError):
    """Raised when a decoder kind is unknown, malformed or already registered."""


def _validated_kind(kind) -> str:
    """Kind syntax, reported as a registration error (the registry's own vocabulary)."""
    try:
        return validate_decoder_kind(kind)
    except ValueError as error:
        raise DecoderRegistrationError(str(error)) from None


def register_decoder(kind: str, factory: Callable | None = None):
    """Register ``factory`` as the implementation of ``kind`` (usable as a decorator).

    ``factory(config, patches, d_world)`` receives the validated
    :class:`~wpm_video.config.DecoderConfig`, the patch count and the latent width of
    the world model, and must return an :class:`~wpm_video.decoder.base.RGBDecoder`
    whose geometry matches them. Names are lowercase identifiers and are registered
    once: a duplicate -- including the built-in ``"conv"`` -- is refused rather than
    silently replacing an architecture that existing checkpoints may refer to.
    """
    if factory is None:                      # used as @register_decoder("name")
        def decorator(inner):
            register_decoder(kind, inner)
            return inner
        return decorator
    name = _validated_kind(kind)
    if not callable(factory):
        raise DecoderRegistrationError(
            f"the factory for decoder kind {name!r} must be callable, got {type(factory).__name__}"
        )
    if name in DECODER_FACTORIES:
        raise DecoderRegistrationError(
            f"decoder kind {name!r} is already registered; a kind is registered once so that a "
            "checkpoint always describes the same architecture (use another name for another "
            "implementation)"
        )
    DECODER_FACTORIES[name] = factory
    return factory


def available_decoders() -> tuple:
    """Registered decoder kinds, sorted: what ``build_decoder`` can dispatch to."""
    return tuple(sorted(DECODER_FACTORIES))


def decoder_factory(kind: str) -> Callable:
    """Factory registered for ``kind``, or a readable error listing what exists."""
    name = _validated_kind(kind)
    try:
        return DECODER_FACTORIES[name]
    except KeyError:
        raise DecoderRegistrationError(
            f"decoder kind {name!r} is not registered in this process; registered kinds: "
            f"{list(available_decoders())}. Custom architectures are process-local: import the "
            "module that defines the decoder and call register_decoder(kind, factory) before "
            "building a decoder, loading a checkpoint or training (see docs/decoder_components.md)"
        ) from None


def _check_component(decoder, config: DecoderConfig, patches: int, d_world: int) -> None:
    """Refuse a factory result that does not describe what it was asked to build.

    A custom implementation is only useful if the rest of the pipeline can trust the
    metadata it returns: every geometry check and every compatibility decision reads
    it, so a component that reports a different kind, grid, output size or
    configuration is rejected here instead of corrupting a checkpoint later.
    """
    name = type(decoder).__name__
    if not isinstance(decoder, RGBDecoder):
        raise DecoderRegistrationError(
            f"the factory for decoder kind {config.kind!r} returned {name}, which is not an "
            "RGBDecoder; custom architectures must subclass wpm_video.decoder.RGBDecoder"
        )
    if decoder.kind != config.kind:
        raise DecoderRegistrationError(
            f"the factory for decoder kind {config.kind!r} returned a decoder of kind "
            f"{decoder.kind!r}; the kind a component reports must be the one it was built under"
        )
    if decoder.config != config:
        raise DecoderRegistrationError(
            f"the {config.kind!r} decoder does not describe the configuration it was built from: "
            f"built={decoder.config!r}, requested={config!r}"
        )
    side = grid_side(patches)
    if (decoder.patches, decoder.d_world, tuple(decoder.grid)) != (patches, d_world, (side, side)):
        raise DecoderRegistrationError(
            f"the {config.kind!r} decoder reports patches={decoder.patches}, "
            f"d_world={decoder.d_world}, grid={tuple(decoder.grid)}, but was built for "
            f"patches={patches}, d_world={d_world}, grid={(side, side)}"
        )
    if decoder.output_size != int(config.image_size):
        raise DecoderRegistrationError(
            f"the {config.kind!r} decoder outputs {decoder.output_size}px but the configuration "
            f"asks for {config.image_size}px; output_size must equal decoder.image_size"
        )


def build_decoder(config: DecoderConfig, patches: int, d_world: int) -> RGBDecoder:
    """Construct the decoder ``config.kind`` describes for this world model.

    The configuration is validated, the registered factory is called and its result
    is checked against the request, so a broken custom component fails here with a
    message instead of producing a checkpoint that cannot be loaded again.
    """
    validate_decoder_config(config)
    factory = decoder_factory(config.kind)
    decoder = factory(config, patches, d_world)
    _check_component(decoder, config, patches, d_world)
    return decoder
