"""Decoder checkpoints and the backward-compatible decoder entry points.

This module owns the on-disk format of a decoder checkpoint and keeps the imports
that existed before decoders became replaceable components working unchanged:
``LatentRGBDecoder``, ``build_decoder``, ``save_decoder``, ``load_decoder``,
``group_count`` and ``DECODER_SCHEMA_VERSION`` are all still here, and
``LatentRGBDecoder`` is still *the* conv network with the same parameter names,
shapes and initialisation order, so state dicts written by earlier versions keep
loading exactly as they did.

What changed is that a checkpoint now says **which component** it holds:

- **schema 1** (written before decoder kinds existed) describes exactly one
  architecture, the built-in conv upsampler. It has no ``kind`` and no ``options``;
  reading it means "legacy conv" and nothing else. A schema-1 file that carries a
  kind, or whose architecture record is not the historical conv one, is refused as
  malformed instead of being normalized into whatever it might have meant.
- **schema 2** (written from now on) records every configuration field explicitly
  (``kind``/``image_size``/``options``, plus the conv shape fields for conv) and the
  component identity in the architecture record, so two architectures with the same
  output size can never be confused for one another and a truncated record is an
  error rather than something completed with defaults.

Both versions are read strictly: the architecture record must agree with the stored
configuration, the grid and the dimensions must agree with each other, and the state
dict must match the built module. A checkpoint of an unknown or future schema, or of
a kind that is not registered in this process, fails before any weight or pixel is
produced.
"""

from collections.abc import Mapping
from pathlib import Path

import torch

from ..config import (CONV_ARCHITECTURE_FIELDS, CONV_DECODER_KIND, DecoderConfig,
                      decoder_architecture, decoder_config_snapshot, validate_decoder_config)
from .architectures.conv import LatentRGBDecoder, group_count
from .base import RGBDecoder, grid_side
from .registry import available_decoders, build_decoder, register_decoder

# Checkpoint schema written by this version. Bumping it invalidates older
# checkpoints instead of silently reading them with new expectations.
DECODER_SCHEMA_VERSION = 2
# Schema of checkpoints written before decoder architectures were selectable. They
# can only describe the built-in conv decoder, which is what makes reading them
# unambiguous.
LEGACY_DECODER_SCHEMA_VERSION = 1
SUPPORTED_DECODER_SCHEMA_VERSIONS = (LEGACY_DECODER_SCHEMA_VERSION, DECODER_SCHEMA_VERSION)

# Top-level fields the format owns. Extra checkpoint metadata (provenance,
# optimiser state, the run configuration) is stored beside them and may not
# override them: the architecture and the weights a checkpoint holds are facts, not
# annotations.
RESERVED_CHECKPOINT_FIELDS = frozenset({
    "kind", "schema_version", "decoder_config", "architecture", "patches", "d_world", "grid",
    "state_dict", "path",
})

# FIELDS of a schema-1 decoder_config: exactly the conv architecture record.
LEGACY_DECODER_CONFIG_FIELDS = CONV_ARCHITECTURE_FIELDS

__all__ = [
    "DECODER_SCHEMA_VERSION", "LEGACY_DECODER_SCHEMA_VERSION",
    "SUPPORTED_DECODER_SCHEMA_VERSIONS", "RESERVED_CHECKPOINT_FIELDS",
    "LatentRGBDecoder", "RGBDecoder", "group_count", "build_decoder", "register_decoder",
    "available_decoders", "checkpoint_schema_version", "checkpoint_component_kind",
    "save_decoder", "load_decoder",
]


def checkpoint_schema_version(payload: Mapping) -> "int | None":
    """The checkpoint's schema version, or ``None`` unless it is a supported integer.

    ``True`` is not schema 1: a boolean (or a string) is a malformed version, not a
    value to be compared loosely.
    """
    version = payload.get("schema_version") if isinstance(payload, Mapping) else None
    return version if type(version) is int and version in SUPPORTED_DECODER_SCHEMA_VERSIONS \
        else None


def checkpoint_component_kind(payload: Mapping) -> "str | None":
    """Decoder component kind a checkpoint records, or ``None`` if it records none.

    A schema-1 checkpoint predates kinds and can only hold the built-in conv
    architecture, so "no kind" there means ``"conv"`` rather than "unknown". For a
    later schema the kind must be declared explicitly: guessing it from the
    architecture record is exactly the silent mismatch this function exists to
    prevent.
    """
    if not isinstance(payload, Mapping):
        return None
    if checkpoint_schema_version(payload) == LEGACY_DECODER_SCHEMA_VERSION:
        return CONV_DECODER_KIND
    stored = payload.get("decoder_config")
    kind = stored.get("kind") if isinstance(stored, Mapping) else None
    return kind if isinstance(kind, str) and kind else None


def decoder_config_from_payload(payload: Mapping, version: int, source=None) -> DecoderConfig:
    """Rebuild the decoder configuration a checkpoint records, or explain why it cannot.

    ``version`` is the checkpoint's schema, already checked to be supported. Schema 1
    is normalized to ``kind="conv"``/``options={}`` because that is what it means --
    and it must carry exactly the historical conv fields, so a truncated or
    hand-edited record fails here instead of being completed with defaults that
    describe a network the checkpoint never held.
    """
    where = str(source) if source is not None else "the decoder checkpoint"
    stored = payload.get("decoder_config")
    if not isinstance(stored, Mapping):
        raise ValueError(f"{where} is missing 'decoder_config'; it is not a complete decoder "
                         "checkpoint")
    if version == LEGACY_DECODER_SCHEMA_VERSION:
        # schema 1 has exactly the historical conv fields: complete, and nothing else
        unexpected = sorted(set(stored) - set(LEGACY_DECODER_CONFIG_FIELDS))
        if unexpected:
            raise ValueError(
                f"{where} is a schema-1 checkpoint, which predates selectable decoder "
                f"architectures, but its decoder_config carries {unexpected}; it is not read as "
                "legacy conv"
            )
        required = LEGACY_DECODER_CONFIG_FIELDS
    else:
        shared = ("kind", "image_size", "options")
        unexpected = sorted(set(stored) - set(DecoderConfig.__dataclass_fields__))
        if unexpected:
            raise ValueError(
                f"{where} records decoder_config fields this version does not know: {unexpected}"
            )
        if not isinstance(stored.get("kind"), str) or not stored.get("kind"):
            raise ValueError(
                f"{where} does not record 'decoder_config.kind'; a schema-2 decoder checkpoint "
                "must name the component architecture that produced it"
            )
        # the shared fields, plus the conv fields for conv: a truncated record is an
        # error, never completed with defaults that describe a network it never held
        required = shared + (CONV_ARCHITECTURE_FIELDS
                             if stored["kind"] == CONV_DECODER_KIND else ())
    missing = [name for name in required if name not in stored]
    if missing:
        raise ValueError(
            f"{where} is missing 'decoder_config.{missing[0]}'; a schema-{version} decoder "
            "configuration records "
            + ("the complete conv architecture" if version == LEGACY_DECODER_SCHEMA_VERSION
               else "kind, image_size and options")
            + " and is not completed with defaults"
        )
    try:
        config = DecoderConfig(**stored)
    except TypeError as error:
        raise ValueError(f"{where} has a malformed decoder_config: {error}") from error
    validate_decoder_config(config)
    return config


def normalize_architecture_record(payload: Mapping, version: int, source=None) -> dict:
    """The architecture record of a checkpoint in canonical (current) form.

    Schema 1 predates decoder kinds: its record is exactly the historical conv shape
    dictionary, so it is completed with ``kind="conv"`` for comparison. Anything
    else -- a missing field, an extra one, a schema-2 record without a kind -- is
    refused; the function never repairs a record into agreeing with a configuration.
    """
    where = str(source) if source is not None else "the decoder checkpoint"
    stored = payload.get("architecture")
    if not isinstance(stored, Mapping) or not stored:
        raise ValueError(f"{where} does not record 'architecture'; it is not a complete decoder "
                         "checkpoint")
    if version == LEGACY_DECODER_SCHEMA_VERSION:
        unexpected = sorted(set(stored) - set(CONV_ARCHITECTURE_FIELDS))
        missing = [name for name in CONV_ARCHITECTURE_FIELDS if name not in stored]
        if unexpected or missing:
            raise ValueError(
                f"{where} is a schema-1 checkpoint, whose architecture record is exactly the conv "
                f"architecture; got keys {sorted(stored)}"
                + (f" (missing {missing})" if missing else "")
            )
        return {"kind": CONV_DECODER_KIND,
                **{name: stored[name] for name in CONV_ARCHITECTURE_FIELDS}}
    return dict(stored)


def save_decoder(path, decoder: RGBDecoder, extra: dict | None = None) -> None:
    """Write a decoder checkpoint; ``extra`` carries the identity and provenance.

    ``extra`` is merged beside the format fields and may not redefine them: a caller
    cannot replace the recorded architecture, dimensions or weights with metadata.
    """
    if not isinstance(decoder, RGBDecoder):
        raise ValueError(
            f"save_decoder expects an RGBDecoder, got {type(decoder).__name__}; a custom "
            "architecture must subclass wpm_video.decoder.RGBDecoder"
        )
    config = decoder.config
    validate_decoder_config(config)
    side = grid_side(decoder.patches)
    if tuple(decoder.grid) != (side, side):
        raise ValueError(
            f"cannot save a decoder whose grid {tuple(decoder.grid)} does not match its "
            f"{decoder.patches} patches ({(side, side)})"
        )
    if decoder.output_size != int(config.image_size):
        raise ValueError(
            f"cannot save a decoder that outputs {decoder.output_size}px while its configuration "
            f"declares decoder.image_size={config.image_size}"
        )
    if extra:
        if not isinstance(extra, Mapping):
            raise ValueError(f"decoder checkpoint metadata must be a mapping, got {type(extra)}")
        reserved = sorted(set(extra) & RESERVED_CHECKPOINT_FIELDS)
        if reserved:
            raise ValueError(
                f"decoder checkpoint metadata may not override {reserved}; these fields describe "
                "the architecture, the weights and the dimensions the file actually holds"
            )
    payload = {
        "kind": "rgb_decoder",
        "schema_version": DECODER_SCHEMA_VERSION,
        "decoder_config": decoder_config_snapshot(config),
        "architecture": decoder_architecture(config),
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

    The checkpoint must be a decoder of a supported schema (1, the legacy conv
    architecture, or 2, which names its component). A world-model checkpoint, a
    truncated file, a future schema, an unknown component kind or a record whose
    architecture disagrees with its configuration is refused with a readable message
    rather than loaded partially -- before any weight is transferred into a module.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("kind") != "rgb_decoder":
        raise ValueError(
            f"{path} is not an RGB decoder checkpoint (expected kind='rgb_decoder'); a world "
            "model checkpoint cannot be used as --decoder-checkpoint"
        )
    version = checkpoint_schema_version(payload)
    if version is None:
        raise ValueError(
            f"{path} has decoder schema_version {payload.get('schema_version')!r}, this package "
            f"reads {list(SUPPORTED_DECODER_SCHEMA_VERSIONS)} and writes "
            f"{DECODER_SCHEMA_VERSION}; retrain the decoder with this version"
        )
    for field in ("decoder_config", "architecture", "patches", "d_world", "grid", "state_dict"):
        if field not in payload:
            raise ValueError(f"{path} is missing '{field}'; it is not a complete decoder checkpoint")
    config = decoder_config_from_payload(payload, version, path)
    patches, d_world = payload["patches"], payload["d_world"]
    if type(patches) is not int or type(d_world) is not int or patches < 1 or d_world < 1:
        raise ValueError(
            f"{path} records patches={patches!r} and d_world={d_world!r}; both must be positive "
            "integers"
        )
    stored_architecture = normalize_architecture_record(payload, version, path)
    expected_architecture = decoder_architecture(config)
    if stored_architecture != expected_architecture:
        raise ValueError(
            f"{path} is inconsistent: its architecture record {stored_architecture} does not "
            f"describe its decoder_config {expected_architecture}; the file was edited or "
            "truncated and is not loaded"
        )
    stored_grid = payload["grid"]
    side = grid_side(patches)
    if not isinstance(stored_grid, (list, tuple)) or list(stored_grid) != [side, side]:
        raise ValueError(
            f"{path} records grid={stored_grid!r} for {patches} patches (expected {[side, side]})"
        )
    stored_state = payload["state_dict"]
    if not isinstance(stored_state, Mapping) or not stored_state:
        raise ValueError(f"{path} records no decoder weights")
    not_tensors = sorted(key for key, value in stored_state.items() if not torch.is_tensor(value))
    if not_tensors:
        raise ValueError(
            f"{path} records {not_tensors} as something other than tensors; the weights were "
            "rewritten (a checkpoint must be saved and loaded as a file, not through JSON)"
        )
    # dispatch through the registry: an unregistered kind is refused here, with the
    # registered names listed, instead of falling back to any built-in architecture
    decoder = build_decoder(config, patches, d_world)
    if tuple(decoder.grid) != (side, side):
        raise ValueError(f"{path}: the {config.kind!r} decoder does not match the recorded grid")
    try:
        decoder.load_state_dict(stored_state)
    except RuntimeError as error:
        raise ValueError(
            f"{path} holds weights that do not fit its recorded {config.kind!r} architecture "
            f"({expected_architecture}): {error}"
        ) from error
    # where this checkpoint was actually read from, so artifacts and summaries can
    # name it; the saved file never records its own location, the loader adds it
    payload["path"] = str(Path(path).resolve())
    return decoder, payload
