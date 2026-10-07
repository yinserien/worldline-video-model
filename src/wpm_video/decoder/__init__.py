"""Optional RGB decoder for projected world latents.

This subpackage is the only place that knows how a latent becomes an image. The
world model, the persistent state and the dynamics never import it, so every
command keeps working exactly as before when no decoder checkpoint is given::

    from wpm_video import build_decoder, run_decoder_training, decode_latents

    decoder = build_decoder(config.decoder, model.patches, model.config.d_world)
    run_decoder_training(config, "runs/run1/best.pt", "runs/decoder", device)
    images = decode_latents(decoder, {1: mu}, device)      # {1: (3, S, S) in [0, 1]}

Each decoded image is one keyframe: the last sampled frame of a chunk. Ordered
horizons are separate keyframes, not a continuous video, and reconstruction from a
frozen encoder plus a random projection is lossy -- blur and missing texture are
expected, and the output is only "decoded RGB", never generated video.

The decoder itself is a replaceable component. ``config.decoder.kind`` selects a
registered architecture (:class:`~wpm_video.decoder.base.RGBDecoder`), ``"conv"`` --
the compact convolutional upsampler above -- is the built-in default, and a custom
implementation registers itself with :func:`register_decoder` before anything is
built, loaded or trained. Registration is process-local and never inferred from a
checkpoint or a configuration file; see docs/decoder_components.md.
"""

from .architectures import conv as _conv_architecture  # noqa: F401  registers the built-in "conv"
from .base import RGBDecoder
from .compat import (IDENTITY_FIELDS, TARGET_SEMANTICS, DecoderCompatibilityError,
                     check_cache_matches_world, check_decoder_compatibility,
                     check_world_preprocessing, decoder_fingerprint, decoder_identity,
                     encoder_identity, identity_gaps, projection_fingerprint, sampling_identity,
                     tensor_fingerprint)
from .model import (DECODER_SCHEMA_VERSION, LEGACY_DECODER_SCHEMA_VERSION, LatentRGBDecoder,
                    SUPPORTED_DECODER_SCHEMA_VERSIONS, build_decoder, checkpoint_component_kind,
                    group_count, load_decoder, save_decoder)
from .registry import DecoderRegistrationError, available_decoders, register_decoder
from .render import (decode_latents, render_latents, safe_name, save_frame_png, to_uint8,
                     write_artifact, write_records)
from .targets import (CacheAlignmentError, ChunkFrameSource, alignment_record, check_alignment)
from .train import (METRIC_NOTE, context_chunk_indices, decoder_loss, edge_l1, evaluate_decoder,
                    gather_decoder_batch, prepare_decoder_data, psnr_db, run_decoder_training,
                    sample_chunk_indices, splits_from_world_checkpoint, train_decoder,
                    train_mean_frame)

__all__ = [
    "LatentRGBDecoder", "RGBDecoder", "build_decoder", "load_decoder", "save_decoder",
    "group_count", "checkpoint_component_kind",
    "register_decoder", "available_decoders", "DecoderRegistrationError",
    "DECODER_SCHEMA_VERSION", "LEGACY_DECODER_SCHEMA_VERSION", "SUPPORTED_DECODER_SCHEMA_VERSIONS",
    "DecoderCompatibilityError", "check_decoder_compatibility", "check_cache_matches_world",
    "check_world_preprocessing", "decoder_identity", "decoder_fingerprint",
    "projection_fingerprint", "encoder_identity", "sampling_identity", "tensor_fingerprint",
    "identity_gaps", "IDENTITY_FIELDS", "TARGET_SEMANTICS",
    "ChunkFrameSource", "CacheAlignmentError", "alignment_record", "check_alignment",
    "decode_latents", "render_latents", "save_frame_png", "to_uint8", "write_artifact",
    "write_records", "safe_name",
    "run_decoder_training", "train_decoder", "evaluate_decoder", "train_mean_frame",
    "gather_decoder_batch", "sample_chunk_indices", "context_chunk_indices",
    "splits_from_world_checkpoint", "prepare_decoder_data", "decoder_loss", "edge_l1", "psnr_db",
    "METRIC_NOTE",
]
