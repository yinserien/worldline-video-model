"""Decoder/world compatibility: fingerprints and the checks run before any image.

A decoder is only meaningful for the exact latent space it was trained on. Two
world checkpoints can share ``d_world`` and a patch count and still describe
different spaces, because the projection is a seeded random matrix whose
standardisation is fitted during training; a different encoder revision, fps,
chunk length or source image size silently changes the token distribution too.
Matching dimensions are therefore never enough, so a decoder checkpoint records

- the shape it consumes (``d_world``, patch count and grid),
- the component architecture that produced it (``kind``, its options and the
  canonical architecture record), so a different architecture with the same output
  size can never be mistaken for the one that was trained,
- a SHA-256 fingerprint of the world projection buffers (weight, mean, std),
- the encoder identity (kind, model id, revision, width, patch count),
- the sampling/preprocessing it was trained with (fps, chunking, token image
  size, timeline schema),
- the semantics and resolution of the RGB target frame,

and every one of those is compared before an image is written. A checkpoint that
does not carry the full identity is refused rather than accepted on the strength of
a matching shape: the checks are exhaustive by construction, there is no "field
missing, skip it" path. World *dynamics* weights are deliberately not part of the
fingerprint: they may keep evolving, the frozen projection is what defines the
decoder's input space.
"""

from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch

from ..config import DataConfig, EncoderConfig, RunConfig, decoder_architecture
from .base import RGBDecoder
from .model import (SUPPORTED_DECODER_SCHEMA_VERSIONS, checkpoint_component_kind,
                    checkpoint_schema_version, decoder_config_from_payload,
                    normalize_architecture_record)

# Bumping this string invalidates every decoder checkpoint: it is the contract of
# what an RGB target *is*, not a formatting detail.
TARGET_SEMANTICS = (
    "one RGB keyframe per chunk: the last sampled frame of the chunk, square "
    "resize to decoder.image_size, uint8 RGB [0,255] on disk [0,1] in the model, "
    "timestamp = chunk end_seconds - 1/source_fps"
)

# Fields that define the token space a decoder consumes. ``context_chunks`` is
# deliberately absent: it changes how many chunks a world-model window observes, not
# how a chunk is encoded, sampled or projected, so it is recorded as provenance
# (``sampling_identity``) but never compared or required.
IDENTITY_FIELDS = {
    "world_projection": ("weight", "mean", "std", "sha256", "shape", "seed"),
    "encoder": ("kind", "model_id", "revision", "d_model", "patches"),
    "sampling": ("timeline_schema", "fps", "chunk_frames", "chunk_stride_frames", "image_size",
                 "patches"),
    "frame_target": ("semantics", "role", "resolution", "range"),
}


class DecoderCompatibilityError(ValueError):
    """Raised when a decoder checkpoint does not belong to this world/config."""


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """SHA-256 over dtype, shape and raw little-endian bytes of one tensor."""
    flat = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(flat.dtype).encode())
    digest.update(str(tuple(flat.shape)).encode())
    digest.update(flat.numpy().tobytes())
    return digest.hexdigest()


def projection_fingerprint(model) -> dict:
    """Fingerprint of the world projection buffers: weight, mean and std.

    These three buffers define the decoder's input space. They are frozen in the
    world model (never trained), so an exact match means two checkpoints project
    encoder tokens identically.
    """
    projection = model.projection
    parts = {
        "weight": tensor_fingerprint(projection.weight),
        "mean": tensor_fingerprint(projection.mean),
        "std": tensor_fingerprint(projection.std),
    }
    digest = hashlib.sha256()
    for name in ("weight", "mean", "std"):
        digest.update(f"{name}:{parts[name]}".encode())
    return {
        **parts,
        "sha256": digest.hexdigest(),
        "shape": list(projection.weight.shape),
        "seed": int(getattr(model.config, "projection_seed", -1)),
    }


def encoder_identity(config: EncoderConfig, model=None) -> dict:
    """Encoder identity as the cache key sees it, plus the token width and grid.

    ``EncoderConfig.identity`` is ``(model_id, revision)`` -- neutralised to
    ``("native", "native")`` for the native backend -- and ``kind`` completes the
    triple the cache key uses.
    """
    model_id, revision = config.identity
    projection = getattr(model, "projection", None)
    return {
        "kind": config.kind,
        "model_id": model_id,
        "revision": str(revision),
        "d_model": int(projection.weight.shape[0]) if projection is not None else None,
        "patches": int(model.patches) if model is not None else None,
    }


def sampling_identity(data: DataConfig, model=None) -> dict:
    """How frames were sampled and chunked, and which timeline schema was used.

    ``context_chunks`` is recorded for provenance but is not part of the compared
    token space (see ``IDENTITY_FIELDS``): it describes the world model's observation
    window, not the latent a decoder consumes.
    """
    from ..dataset import TIMELINE_SCHEMA

    return {
        "timeline_schema": TIMELINE_SCHEMA,
        "fps": float(data.fps),
        "chunk_frames": int(data.chunk_frames),
        "chunk_stride_frames": int(data.chunk_stride_frames),
        "image_size": int(data.image_size),          # token-cache resolution
        "context_chunks": int(data.context_chunks),
        "patches": int(model.patches) if model is not None else None,
    }


def decoder_identity(decoder: RGBDecoder, model, config: RunConfig) -> dict:
    """Everything a compatibility check needs, as stored in a decoder checkpoint."""
    return {
        "world_projection": projection_fingerprint(model),
        "encoder": encoder_identity(config.encoder, model),
        "sampling": sampling_identity(config.data, model),
        "frame_target": {
            "semantics": TARGET_SEMANTICS,
            "role": "last sampled frame of the chunk",
            "resolution": int(decoder.output_size),
            "range": "[0, 1] float, uint8 PNG",
        },
    }


def stored_architecture(payload: Mapping, source=None) -> dict:
    """Canonical architecture record of a checkpoint (schema 1 = the legacy conv one).

    Kept next to the compatibility checks so a caller comparing a checkpoint with a
    configuration uses the same normalization the loader does, including its refusal
    to repair a malformed record.
    """
    return normalize_architecture_record(payload, checkpoint_schema_version(payload), source)


def identity_gaps(payload: Mapping) -> list:
    """Missing or ``None`` identity fields, so an incomplete checkpoint is refused."""
    gaps = []
    if not isinstance(payload, Mapping):
        return ["checkpoint is not a mapping"]
    version = checkpoint_schema_version(payload)
    if version is None:
        gaps.append(
            f"schema_version {payload.get('schema_version')!r} is not one of the supported "
            f"{list(SUPPORTED_DECODER_SCHEMA_VERSIONS)}"
        )
    if not isinstance(payload.get("decoder_config"), Mapping):
        gaps.append("missing 'decoder_config'")
    kind = checkpoint_component_kind(payload)
    if kind is None:
        gaps.append("missing 'decoder_config.kind'")
    for section, fields in IDENTITY_FIELDS.items():
        stored = payload.get(section)
        if not isinstance(stored, Mapping):
            gaps.append(f"missing '{section}' block")
            continue
        for field in fields:
            if stored.get(field) is None:
                gaps.append(f"missing '{section}.{field}'")
    if payload.get("d_world") is None or payload.get("patches") is None:
        gaps.append("missing d_world/patches")
    if not payload.get("architecture"):
        gaps.append("missing 'architecture'")
    else:
        try:
            stored_architecture(payload)
        except ValueError as error:
            gaps.append(str(error))
    if version in SUPPORTED_DECODER_SCHEMA_VERSIONS and kind is not None:
        recorded = payload.get("architecture") or {}
        if isinstance(recorded, Mapping) and recorded.get("kind") not in (None, kind):
            gaps.append(
                f"'decoder_config.kind' ({kind!r}) and 'architecture.kind' "
                f"({recorded.get('kind')!r}) disagree"
            )
    return gaps


def check_decoder_compatibility(payload: dict, model, config: RunConfig,
                                decoder: RGBDecoder | None = None) -> dict:
    """Reject a decoder that does not match this world model and configuration.

    Collects every mismatch and raises once, so a user sees all the reasons in a
    single message instead of fixing them one at a time. Returns the verified
    identity record on success.
    """
    if not isinstance(payload, dict) or payload.get("kind") != "rgb_decoder":
        raise DecoderCompatibilityError("not an RGB decoder checkpoint (kind != 'rgb_decoder')")
    gaps = identity_gaps(payload)
    if gaps:
        raise DecoderCompatibilityError(
            "this decoder checkpoint does not carry a complete identity, so it cannot be matched "
            "against a world model:\n  - " + "\n  - ".join(gaps)
            + "\nRetrain it with `wpm-video train-decoder` from this package version."
        )
    reasons = []
    live_projection = projection_fingerprint(model)
    stored_projection = payload["world_projection"]
    for name in ("weight", "mean", "std", "sha256"):
        if stored_projection[name] != live_projection[name]:
            reasons.append(
                f"world projection {name} differs: the decoder was trained on a different "
                "projection (or on a checkpoint whose standardisation was fitted differently)"
            )
            break
    if payload["d_world"] != int(model.config.d_world):
        reasons.append(
            f"latent width differs: decoder d_world={payload['d_world']}, "
            f"world model d_world={model.config.d_world}"
        )
    if payload["patches"] != int(model.patches):
        reasons.append(
            f"patch layout differs: decoder patches={payload['patches']}, "
            f"world model patches={model.patches}"
        )
    if list(payload.get("grid") or []) != [int(model.grid[0]), int(model.grid[1])]:
        reasons.append(
            f"patch grid differs: decoder grid={payload.get('grid')}, "
            f"world model grid={list(model.grid)}"
        )
    live_encoder = encoder_identity(config.encoder, model)
    for name in IDENTITY_FIELDS["encoder"]:
        if payload["encoder"][name] != live_encoder[name]:
            reasons.append(
                f"encoder {name} differs: decoder={payload['encoder'][name]!r}, "
                f"config={live_encoder[name]!r}"
            )
    live_sampling = sampling_identity(config.data, model)
    for name in IDENTITY_FIELDS["sampling"]:
        if payload["sampling"][name] != live_sampling[name]:
            reasons.append(
                f"sampling {name} differs: decoder={payload['sampling'][name]!r}, "
                f"config={live_sampling[name]!r}"
            )
    if payload["frame_target"]["semantics"] != TARGET_SEMANTICS:
        reasons.append(
            "target frame semantics changed since this decoder was trained; retrain it "
            "(the recorded semantics no longer describe what a target image is)"
        )
    # the checkpoint must be internally consistent even when no module is compared:
    # its architecture record has to describe the configuration it also records
    recorded_architecture = None
    stored_kind = checkpoint_component_kind(payload)
    try:
        recorded_config = decoder_config_from_payload(payload, checkpoint_schema_version(payload),
                                                      "this decoder checkpoint")
        recorded_architecture = stored_architecture(payload)
    except ValueError as error:
        reasons.append(str(error))
    else:
        expected = decoder_architecture(recorded_config)
        if recorded_architecture != expected:
            reasons.append(
                f"decoder architecture differs from the decoder_config the checkpoint records: "
                f"architecture={recorded_architecture}, decoder_config={expected}"
            )
    if decoder is not None:
        if payload["frame_target"]["resolution"] != int(decoder.output_size):
            reasons.append(
                f"frame target resolution differs: checkpoint={payload['frame_target']['resolution']}, "
                f"module={decoder.output_size}"
            )
        if stored_kind is None:
            reasons.append(
                "this checkpoint does not record which decoder architecture produced it, so it "
                "cannot be matched against a module"
            )
        elif stored_kind != decoder.kind:
            # a different kind is a different component, whatever the shapes say
            reasons.append(
                f"decoder component kind differs: checkpoint={stored_kind!r}, "
                f"module={decoder.kind!r}"
            )
        elif recorded_architecture is not None and recorded_architecture != \
                decoder_architecture(decoder.config):
            reasons.append(
                f"decoder architecture differs: checkpoint={recorded_architecture}, "
                f"module={decoder_architecture(decoder.config)}"
            )
    if reasons:
        raise DecoderCompatibilityError(
            "decoder checkpoint is incompatible with this world model/configuration:\n  - "
            + "\n  - ".join(reasons)
            + "\nRetrain the decoder for this checkpoint (wpm-video train-decoder)."
        )
    return {section: dict(payload[section]) for section in IDENTITY_FIELDS}


def check_cache_matches_world(dataset, model, config: RunConfig) -> None:
    """The cached tokens must have been produced by the encoder this world model expects.

    The cache path already encodes the encoder identity, but a cache directory can be
    copied around and a config edited; the recorded metadata is compared against the
    loaded model's projection width, its patch count and the encoder identity, so a
    width or revision mismatch fails with a readable message instead of a matmul
    shape error (or, worse, a silent mismatch of token semantics).
    """
    d_encoder = int(model.projection.weight.shape[0])
    _, revision = config.encoder.identity
    for name, cache in dataset.entries.items():
        meta = cache.meta
        checks = {
            "encoder": (meta.get("encoder"), config.encoder.kind),
            "encoder_d_model": (meta.get("encoder_d_model"), d_encoder),
            "patches": (meta.get("patches"), int(model.patches)),
            "image_size": (meta.get("image_size"), int(config.data.image_size)),
        }
        for field, (recorded, expected) in checks.items():
            if recorded is None:
                raise DecoderCompatibilityError(
                    f"{name}: token cache has no '{field}' metadata, so it cannot be tied to this "
                    "world model; re-run `wpm-video cache` with this package version"
                )
            if recorded != expected:
                raise DecoderCompatibilityError(
                    f"{name}: token cache {field}={recorded!r} does not match this world model / "
                    f"configuration ({expected!r}); cache these clips with the encoder the world "
                    "checkpoint was trained with"
                )
        # revisions are compared at the same 8-character granularity the cache key
        # uses, so a short pin and a full commit hash of the same revision agree
        recorded_revision = meta.get("encoder_revision")
        if recorded_revision is None:
            raise DecoderCompatibilityError(
                f"{name}: token cache records no encoder revision; re-run `wpm-video cache`"
            )
        if str(recorded_revision)[:8] != str(revision)[:8]:
            raise DecoderCompatibilityError(
                f"{name}: token cache was written by encoder revision "
                f"{recorded_revision!r}, but the configuration pins {revision!r}"
            )
        expected_shape = (int(model.patches), d_encoder)
        if tuple(cache.tokens.shape[1:]) != expected_shape:
            raise DecoderCompatibilityError(
                f"{name}: cached tokens have shape {tuple(cache.tokens.shape[1:])}, but the world "
                f"model expects {expected_shape}"
            )


def check_world_preprocessing(world_payload: dict, config: RunConfig, world_checkpoint=None) -> dict:
    """The caller's config must describe the same tokens as the world checkpoint.

    Training a decoder is training it *for a specific world model*, so the encoder
    identity and the token preprocessing/sampling of the run must equal the ones the
    world checkpoint was trained with -- otherwise the cached latents would come from a
    different space than the predictions the decoder will later be applied to. A world
    checkpoint that does not record its configuration cannot be used at all here.

    Returns the recorded world configuration as a dict.
    """
    recorded = world_payload.get("config") or (world_payload.get("provenance") or {}).get("config")
    if not isinstance(recorded, Mapping) or not recorded.get("encoder") or not recorded.get("data"):
        raise DecoderCompatibilityError(
            "this world checkpoint does not record the configuration it was trained with, so the "
            "decoder cannot be tied to the world's token space. Train the world model with "
            "`wpm-video train` from this package version, or keep this checkpoint for latent-only "
            f"inference{' (' + str(world_checkpoint) + ')' if world_checkpoint else ''}."
        )
    world_encoder, world_data = recorded["encoder"], recorded["data"]
    live_encoder, live_data = config.encoder, config.data
    reasons = []
    # Compare through EncoderConfig.identity, exactly as the cache key does: the
    # native backend neutralises model_id/revision, so editing those inert fields in a
    # native configuration must not invalidate an otherwise identical token space.
    known = set(EncoderConfig.__dataclass_fields__)
    world_identity = EncoderConfig(**{key: value for key, value in world_encoder.items()
                                      if key in known})
    for label, expected, actual in (("kind", world_identity.kind, live_encoder.kind),
                                    ("model_id", world_identity.identity[0],
                                     live_encoder.identity[0])):
        if expected != actual:
            reasons.append(f"encoder.{label}: world={expected!r}, config={actual!r}")
    if str(world_identity.identity[1])[:8] != str(live_encoder.identity[1])[:8]:
        reasons.append(f"encoder.revision: world={world_identity.identity[1]!r}, "
                       f"config={live_encoder.identity[1]!r}")
    for name in ("fps", "chunk_frames", "chunk_stride_frames", "image_size"):
        expected, actual = world_data.get(name), getattr(live_data, name)
        if expected is None or float(expected) != float(actual):
            reasons.append(f"data.{name}: world={expected!r}, config={actual!r}")
    if reasons:
        raise DecoderCompatibilityError(
            "the configuration does not match the world checkpoint's token space:\n  - "
            + "\n  - ".join(reasons)
            + "\nUse the config.json written by the world training run (it records the encoder and "
            "the sampling that produced the cached latents)."
        )
    return dict(recorded)


def decoder_fingerprint(payload: dict) -> dict:
    """Small, JSON-safe identity record used in artifacts and summaries."""
    projection = payload.get("world_projection") or {}
    return {
        "kind": payload.get("kind"),
        "component_kind": checkpoint_component_kind(payload),
        "schema_version": payload.get("schema_version"),
        "d_world": payload.get("d_world"),
        "patches": payload.get("patches"),
        "grid": payload.get("grid"),
        "image_size": (payload.get("decoder_config") or {}).get("image_size"),
        "architecture": payload.get("architecture"),
        "projection_sha256": projection.get("sha256"),
        "encoder": payload.get("encoder"),
        "step": payload.get("step"),
        "checkpoint": Path(str(payload["path"])).name if payload.get("path") else None,
    }
