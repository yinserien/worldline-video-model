"""worldline_video_model: a persistent-state world model over frozen video latents.

The public API is re-exported here::

    from wpm_video import RunConfig, WorldState, VideoWorldModel, build_encoder

The model reads video in causal chunks, keeps a persistent state between them,
writes observations through a Gaussian innovation, advances the state by real time
without observations, and predicts distributions over future spatial latents of a
frozen encoder. See README.md and docs/ for usage.

"""
from .config import DataConfig, EncoderConfig, ModelConfig, RunConfig, TrainConfig
from .data import (Chunk, VideoInfo, build_chunks, build_source_manifest, list_videos,
                   load_source_manifest, probe_video, read_frames, resolve_source)
from .dataset import (TIMELINE_SCHEMA, TokenDataset, Window, WindowSampler, cache_video_tokens,
                      check_split_integrity, gather_batch)
from .encoder import NativeEncoder, VJEPA2Encoder, build_encoder, cache_identity, cache_path
from .evaluate import (BASELINES, baseline_summary, compare, evaluate_baselines,
                       fit_baseline_stats, iter_windows)
from .model import (LatentProjection, SpatialAnchors, VideoWorldModel, gaussian_kl_bits,
                    gaussian_nll_bits, grid_coordinates)
from .predict import demo, predict_at, predict_future, query_saved_state, stream_chunks
from .train import (build_model, evaluate_model, fit_projection, rng_restore, rng_snapshot,
                    set_determinism, source_signature, train)
from .world_state import WorldState

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DataConfig", "EncoderConfig", "ModelConfig", "RunConfig", "TrainConfig",
    "Chunk", "VideoInfo", "build_chunks", "list_videos", "probe_video", "read_frames",
    "build_source_manifest", "load_source_manifest", "resolve_source",
    "TIMELINE_SCHEMA", "TokenDataset", "Window", "WindowSampler", "cache_video_tokens",
    "check_split_integrity", "gather_batch",
    "NativeEncoder", "VJEPA2Encoder", "build_encoder", "cache_path", "cache_identity",
    "BASELINES", "baseline_summary", "compare", "evaluate_baselines", "fit_baseline_stats",
    "iter_windows",
    "LatentProjection", "SpatialAnchors", "VideoWorldModel",
    "gaussian_kl_bits", "gaussian_nll_bits", "grid_coordinates",
    "demo", "predict_at", "predict_future", "query_saved_state", "stream_chunks",
    "build_model", "evaluate_model", "fit_projection", "train", "source_signature",
    "rng_snapshot", "rng_restore", "set_determinism",
    "WorldState",
]
