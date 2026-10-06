"""Shared fixtures: tiny synthetic mp4 clips and a small native-encoder config.

Everything here is self-contained so the suite runs without downloads, without a
GPU and without the pretrained encoder.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from wpm_video.config import (DataConfig, DecoderConfig, DecoderTrainConfig, EncoderConfig,
                              ModelConfig, RunConfig, TrainConfig)

FRAMES, SIZE, FPS = 16, 64, 8.0


def write_video(path: Path, seed: int = 0, moving: bool = True, fps: float = FPS,
                frames: int = FRAMES) -> Path:
    """Write a real mp4 whose content moves in space and time."""
    generator = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
    for index in range(frames):
        frame = np.zeros((SIZE, SIZE, 3), np.uint8)
        offset = int((SIZE - 20) * (index / max(1, frames - 1))) if moving else SIZE // 2
        frame[10:30, offset:offset + 20] = (200, 40, 40)
        frame[40:60, 40 - offset // 3:60 - offset // 3] = (40, 40, 200)
        frame = np.clip(frame.astype(np.int16) + generator.integers(0, 12, frame.shape), 0, 255)
        writer.write(frame.astype(np.uint8))
    writer.release()
    return path


def make_clips(directory: Path, count: int = 3) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    return [write_video(directory / f"clip{index}.mp4", seed=index) for index in range(count)]


def tiny_config(train_videos, val_videos, steps: int = 6, seed: int = 0) -> RunConfig:
    """Small native-encoder configuration: 4x4 patches, 16 anchors, CPU."""
    return RunConfig(
        name="native_test",
        data=DataConfig(fps=FPS, chunk_frames=4, chunk_stride_frames=4, image_size=SIZE,
                        context_chunks=2, window_stride_chunks=1, horizon_chunks=[1],
                        max_horizon_chunks=2, train_videos=train_videos, val_videos=val_videos),
        encoder=EncoderConfig(kind="native", device="cpu", batch_clips=2),
        model=ModelConfig(d_world=32, d_hidden=64, slots=16, heads=4, substep_seconds=0.125,
                          max_substeps=64, anchor_bandwidth=1.0),
        train=TrainConfig(seed=seed, batch_windows=2, max_steps=steps, eval_interval=3,
                          eval_batches=1, device="cpu", log_interval=1, max_wall_seconds=600.0),
    )


def tiny_decoder_config(train_videos, val_videos, steps: int = 60, seed: int = 0,
                        image_size: int = SIZE) -> RunConfig:
    """The tiny native configuration plus a small RGB decoder.

    The decoder stays deliberately small (16 base channels, a 4x4 patch grid) so a
    whole train/evaluate/resume cycle runs on CPU in seconds; the *default*
    configuration is the ~1.5M parameter one tested separately.
    """
    config = tiny_config(train_videos, val_videos, seed=seed)
    config.decoder = DecoderConfig(image_size=image_size, base_channels=16,
                                   channel_multipliers=[1, 2], stem_blocks=1, blocks_per_stage=1)
    config.decoder_train = DecoderTrainConfig(seed=seed, batch_windows=2, learning_rate=2e-3,
                                              max_steps=steps, eval_interval=steps, eval_batches=1,
                                              log_interval=1, max_wall_seconds=600.0,
                                              frame_cache_videos=2)
    config.validate()
    return config


def tiny_decoder(config):
    """Decoder module matching :func:`tiny_decoder_config` for a 4x4 patch grid."""
    from wpm_video.decoder import build_decoder
    return build_decoder(config.decoder, patches=16, d_world=config.model.d_world)


def native_encoder(d_model: int = 48, seed: int = 5):
    from wpm_video.encoder import NativeEncoder
    return NativeEncoder(d_model=d_model, image_size=SIZE, patch=16, tubelet=2, seed=seed)


def synthetic_tokens(rows: int, cols: int, pattern: str = "halves", seed: int = 0,
                     d_model: int = 48) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(1, rows * cols, d_model, generator=generator)
    half = (rows * cols) // 2
    if pattern == "halves":
        tokens[0, :half] += 4.0
        tokens[0, half:] -= 4.0
    elif pattern == "row_gradient":
        gradient = torch.linspace(-4.0, 4.0, rows)
        tokens[0] = tokens[0] + gradient.repeat_interleave(cols).unsqueeze(-1)
    return tokens
