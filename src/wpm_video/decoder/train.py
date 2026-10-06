"""Separate decoder training path: frozen world model, cached latents, RGB keyframes.

Encoder, world model and projection stay frozen. The decoder is trained on

    input  = model.project(cached tokens of a chunk)     (the exact latent space the
                                                          world model predicts in)
    target = the last sampled RGB frame of that same chunk, from the source video

so a trained decoder can be pointed at a *predicted* ``mu`` later. Nothing is
fitted on validation: the only statistics involved (the projection's
standardisation) come from the world checkpoint and were fitted on its training
split, and the constant-image reference reported next to the metrics is fitted on
training windows only.

Checks run in two stages, and the difference matters when reading a failure:

- **eagerly, before the first gradient step** -- split provenance, train/val
  overlap by file hash and source identity, the caller's encoder/sampling identity
  against the world checkpoint, and every token cache's recorded metadata
  (encoder, width, patch count, chunk list and token row count);
- **lazily, the first time a source video is used** -- its content hash is compared
  with the cache record, and the re-decoded sampling grid, chunk boundaries and
  timestamps are compared with the cached timeline. Clips are decoded one at a time
  under a bounded per-video LRU cache, so a long run does not hash or decode every
  source up front; a clip that changed since caching fails at that point, before any
  of its pixels can become a target.

Unit contract: ``l1`` and ``mse`` are pixel-space means in ``[0, 1]``, ``psnr_db``
is ``10 * log10(1 / mse)`` because the data range is exactly one.
"""

import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from ..config import RunConfig, decoder_architecture
from ..dataset import TokenDataset, WindowSampler, check_split_integrity
from ..model import VideoWorldModel
from ..performance import (autocast_context, build_optimizer, optimizer_policy, precision_policy,
                           require_matching_policy, should_pin, to_device)
from ..train import materialize_stats, rng_restore, rng_snapshot, set_determinism, source_signature
from .compat import (DecoderCompatibilityError, check_decoder_compatibility, decoder_identity)
from .model import build_decoder, save_decoder
from .targets import ChunkFrameSource, alignment_record

METRIC_NOTE = (
    "pixel metrics on held-out videos: l1/mse are means over RGB values in [0, 1], "
    "psnr_db = 10*log10(1/mse) for a data range of exactly 1"
)


# -- data ---------------------------------------------------------------------
def splits_from_world_checkpoint(payload: dict, path=None) -> dict:
    """Recover the world model's train/val split from its checkpoint provenance."""
    splits = ((payload or {}).get("provenance") or {}).get("splits") or {}
    if not splits.get("train") or not splits.get("val"):
        raise ValueError(
            "this world checkpoint carries no train/val split provenance, so the decoder cannot "
            "know which videos are held out. Train the world model with `wpm-video train` (which "
            "records the split) or point --checkpoint at a checkpoint that has it."
        )
    return {
        "train": sorted(str(name) for name in splits["train"]),
        "val": sorted(str(name) for name in splits["val"]),
        "source": f"world checkpoint provenance{' at ' + str(path) if path else ''}",
    }


def prepare_decoder_data(config: RunConfig, world_payload: dict, image_size: int,
                         world_checkpoint=None, world_model=None) -> tuple:
    """Build both token datasets and the target frame source, verifying alignment.

    Everything that can be checked without decoding a video happens here, before
    training starts: the configuration must describe the same encoder and the same
    sampling as the world checkpoint, the split comes from that checkpoint, the two
    splits may not share a file hash or a source identity, each cache must have been
    written by the encoder the checkpoint expects, and each cache's chunk list must
    match its token rows. The *source videos* are checked lazily, the first time each
    one is decoded (content hash, sampling grid, chunk boundaries, timestamps), so
    this function does not read pixels at all.

    ``image_size`` is the *decoder's* output edge, which is authoritative for the
    targets: on resume it comes from the checkpoint, not from the caller's config.
    """
    from .compat import check_cache_matches_world, check_world_preprocessing

    check_world_preprocessing(world_payload, config, world_checkpoint)
    splits = splits_from_world_checkpoint(world_payload, world_checkpoint)
    config.data.train_videos = splits["train"]
    config.data.val_videos = splits["val"]
    config.validate()
    cache_dir = Path(config.data.cache_dir)
    train_set = TokenDataset(config.data, cache_dir, config.encoder, "train")
    val_set = TokenDataset(config.data, cache_dir, config.encoder, "val")
    check_split_integrity(train_set, val_set)
    if world_model is not None:
        for dataset in (train_set, val_set):
            check_cache_matches_world(dataset, world_model, config)
    source = ChunkFrameSource(config.data.video_dir, config.data, image_size,
                              max_videos=config.decoder_train.frame_cache_videos,
                              target_cache_dir=config.decoder_train.target_cache_dir)
    for dataset in (train_set, val_set):
        for name, cache in dataset.entries.items():
            source.register(name, alignment_record(cache, require_identity=True))
    return train_set, val_set, source, splits


def sample_chunk_indices(dataset: TokenDataset, window, generator=None) -> list:
    """Training sample: one chunk drawn uniformly from the window's video."""
    chunk_count = int(dataset.entries[window.video].tokens.shape[0])
    span = chunk_count - window.start
    offset = int(torch.randint(0, span, (1,), generator=generator))
    return [window.start + offset]


def context_chunk_indices(dataset: TokenDataset, window) -> list:
    """Evaluation sample: the window's context chunks, identical for every checkpoint."""
    chunk_count = int(dataset.entries[window.video].tokens.shape[0])
    count = min(dataset.config.context_chunks, chunk_count - window.start)
    return [window.start + offset for offset in range(count)]


@torch.no_grad()
def gather_decoder_batch(world_model, frame_source: ChunkFrameSource, dataset: TokenDataset,
                         windows: list, device, generator=None, deterministic: bool = False,
                         performance=None):
    """Stack projected latents, their target keyframes and the target timestamps.

    The host side does all the per-example work first -- token rows from the cache,
    target keyframes from the frame source -- and only then transfers once and
    projects once for the whole batch, instead of one transfer and one projection per
    example. With ``performance.pin_memory`` on CUDA the two host stacks are pinned
    and copied asynchronously; the values are identical either way, so pinning is a
    transfer detail, never a numerical one.
    """
    token_rows, frame_rows, records = [], [], []
    for window in windows:
        indices = (context_chunk_indices(dataset, window) if deterministic
                   else sample_chunk_indices(dataset, window, generator))
        for index in indices:
            token_rows.append(dataset.tokens(window.video, index))
            frame, timestamp = frame_source.target_frame(window.video,
                                                         dataset.chunk(window.video, index))
            frame_rows.append(frame)
            records.append({"video": window.video, "chunk": index, "target_time_seconds": timestamp})
    tokens = to_device(torch.stack(token_rows), device, performance).float()
    latents = world_model.project(tokens)              # one batched projection
    targets = to_device(torch.stack(frame_rows), device, performance).float().div_(255.0)
    return latents, targets, records


@torch.no_grad()
def train_mean_frame(frame_source: ChunkFrameSource, dataset: TokenDataset, device,
                     limit: int = 64) -> torch.Tensor:
    """Mean target frame over training windows only: the constant-image reference.

    Fitted on the training split, never on validation, and bounded to ``limit``
    chunks so the reference costs one pass over a handful of clips.
    """
    total, count = None, 0
    for window in dataset.windows:
        for index in context_chunk_indices(dataset, window):
            frame = frame_source.target(window.video, dataset.chunk(window.video, index))
            value = frame.to(device).float().div_(255.0)
            total = value if total is None else total + value
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    if count == 0:
        raise RuntimeError("no training frames available for the constant-image reference")
    return total / count


# -- objective ----------------------------------------------------------------
def edge_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between finite differences of prediction and target (a small edge term).

    Plain pixel L1 over-smooths; comparing horizontal and vertical differences
    penalises lost edges without importing a perceptual network.
    """
    horizontal = (prediction[..., 1:, :] - prediction[..., :-1, :]) - \
                 (target[..., 1:, :] - target[..., :-1, :])
    vertical = (prediction[..., :, 1:] - prediction[..., :, :-1]) - \
               (target[..., :, 1:] - target[..., :, :-1])
    return 0.5 * (horizontal.abs().mean() + vertical.abs().mean())


def decoder_loss(prediction: torch.Tensor, target: torch.Tensor, l1_weight: float,
                 edge_weight: float, stats_mode: str = "float") -> tuple:
    """Weighted L1 (+ optional edge) reconstruction loss and its parts.

    The loss is evaluated in float32 even when the decoder ran under reduced
    precision, so the reported numbers keep their meaning. ``stats_mode="tensor"``
    returns the parts as detached tensors, letting a training loop skip the
    GPU->Python synchronisation on steps that do not log.
    """
    if stats_mode not in ("float", "tensor"):
        raise ValueError(f"stats_mode must be 'float' or 'tensor', got {stats_mode!r}")
    prediction, target = prediction.float(), target.float()
    l1 = (prediction - target).abs().mean()
    edge = edge_l1(prediction, target)
    loss = l1_weight * l1 + edge_weight * edge
    stats = {"l1": l1.detach(), "edge": edge.detach()}
    return loss, (stats if stats_mode == "tensor" else materialize_stats(stats))


def psnr_db(mse: float) -> float:
    """PSNR in dB for images in [0, 1]: ``10*log10(1/mse)``, capped at a finite value."""
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


# -- evaluation ---------------------------------------------------------------
@torch.no_grad()
def evaluate_decoder(world_model, decoder, dataset: TokenDataset, frame_source: ChunkFrameSource,
                     config: RunConfig, device, batches: int, reference: torch.Tensor | None = None,
                     seed: int = 1234) -> dict:
    """Reconstruction metrics on held-out videos; consumes no randomness.

    The decoder's module mode is restored afterwards, so scoring inside a training
    loop cannot silently leave the model in eval mode.
    """
    was_training = decoder.training
    decoder.eval()
    sampler = WindowSampler(dataset, seed=seed)
    per_video, total_square, total_abs, frames_seen = {}, 0.0, 0.0, 0
    reference_totals = {"abs": 0.0, "square": 0.0}
    for _ in range(batches):
        windows = sampler.sample(config.decoder_train.batch_windows)
        latents, targets, records = gather_decoder_batch(
            world_model, frame_source, dataset, windows, device, deterministic=True,
            performance=config.performance
        )
        prediction = decoder(latents)
        difference = prediction - targets
        absolute = difference.abs().flatten(1).mean(dim=1)
        square = difference.pow(2).flatten(1).mean(dim=1)
        for index, record in enumerate(records):
            entry = per_video.setdefault(record["video"], {"abs": 0.0, "square": 0.0, "n": 0})
            entry["abs"] += float(absolute[index])
            entry["square"] += float(square[index])
            entry["n"] += 1
        total_abs += float(absolute.sum())
        total_square += float(square.sum())
        frames_seen += len(records)
        if reference is not None:
            baseline = (reference.unsqueeze(0) - targets).flatten(1)
            reference_totals["abs"] += float(baseline.abs().mean(dim=1).sum())
            reference_totals["square"] += float(baseline.pow(2).mean(dim=1).sum())
    decoder.train(was_training)
    if frames_seen == 0:
        raise RuntimeError(f"no held-out frames to evaluate for split {dataset.split!r}")
    metrics = {
        "l1": total_abs / frames_seen,
        "mse": total_square / frames_seen,
        "psnr_db": psnr_db(total_square / frames_seen),
        "n_frames": frames_seen,
        "videos": sorted(per_video),
        "per_video": {
            name: {"l1": entry["abs"] / entry["n"], "mse": entry["square"] / entry["n"],
                   "psnr_db": psnr_db(entry["square"] / entry["n"]), "n_frames": entry["n"]}
            for name, entry in sorted(per_video.items())
        },
        "units": METRIC_NOTE,
    }
    if reference is not None and frames_seen:
        baseline_mse = reference_totals["square"] / frames_seen
        metrics["constant_frame_reference"] = {
            "l1": reference_totals["abs"] / frames_seen,
            "mse": baseline_mse,
            "psnr_db": psnr_db(baseline_mse),
            "fitted_on": "training split only (mean target frame)",
        }
    return metrics


# -- training -----------------------------------------------------------------
def _checkpoint_extra(decoder, world_model, config: RunConfig, optimizer, step, sampler, generator,
                      provenance, history, best, world_checkpoint, metrics, metrics_step,
                      performance=None) -> dict:
    """Everything a resume needs: weights, optimiser, step, RNG streams and policy.

    Both the window sampler and the chunk-selection generator are saved, plus the
    global torch/CUDA streams: missing one of them would make a resumed run a
    *different* run rather than a continuation. ``performance`` records the
    acceleration policy that actually ran, so resuming cannot silently change it.
    """
    return {
        **decoder_identity(decoder, world_model, config),
        "world_checkpoint": world_checkpoint,
        "config": config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "sampler_rng": sampler.generator.get_state(),
        "chunk_rng": generator.get_state(),
        **rng_snapshot(),
        "provenance": provenance,
        "history": history[-200:],
        "best": best,
        "metrics": metrics,
        "metrics_step": metrics_step,
        "parameters": decoder.parameter_count(),
        "performance": performance if performance is not None else {},
    }


def save_decoder_checkpoint(path, decoder, world_model, config, optimizer, step, sampler, generator,
                            provenance, history, best, world_checkpoint, metrics=None,
                            metrics_step=None, performance=None) -> None:
    save_decoder(path, decoder, _checkpoint_extra(decoder, world_model, config, optimizer, step,
                                                  sampler, generator, provenance, history, best,
                                                  world_checkpoint, metrics, metrics_step,
                                                  performance))


def train_decoder(config: RunConfig, world_model: VideoWorldModel, decoder, frame_source,
                  train_set: TokenDataset, val_set: TokenDataset, out_dir: Path, device,
                  resume: str = "", provenance: dict | None = None,
                  world_checkpoint: dict | None = None, splits: dict | None = None) -> dict:
    """Fit the decoder against frozen world latents and cached source frames.

    Mirrors the world training loop: initial/best/final checkpoints, a JSONL log, a
    summary, and a resume that restores the optimiser, the step, the sampler and
    every RNG stream. ``best`` is the lowest held-out L1.
    """
    settings = config.decoder_train
    check_split_integrity(train_set, val_set)
    set_determinism(settings.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)
    world_model.eval()
    decoder.to(device)
    reference = train_mean_frame(frame_source, train_set, device)
    provenance = dict(provenance or {})
    provenance["source_signature"] = source_signature()
    provenance["train_windows"] = len(train_set)
    provenance["val_windows"] = len(val_set)
    provenance["splits"] = splits or provenance.get("splits") or {}
    provenance["decoder_parameters"] = decoder.parameter_count()
    world_checkpoint = dict(world_checkpoint or {})
    performance = config.performance
    optimizer = build_optimizer(decoder.parameters(), performance, settings.learning_rate,
                                settings.weight_decay, device)
    policy = {
        **precision_policy(performance),
        "optimizer": optimizer_policy(performance, device),
        # the compared policy stays flat (scalars only); the verbose compile status is
        # recorded beside it. Compiling a decoder module would wrap it (and its
        # state_dict), so the hook is deliberately not applied here and the checkpoint
        # says so instead of recording an acceleration that never ran.
        "compile": False,
        "compile_status": {"compile": bool(performance.compile), "applied": False,
                           "reason": "not applied to the decoder",
                           "targets": []},
        # the effective flag, not the request: pinned transfers only exist on CUDA
        "pin_memory": should_pin(performance, device),
    }
    sampler = WindowSampler(train_set, seed=settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    step, history = 0, []
    best = {"val_l1": math.inf, "step": 0}
    metrics, metrics_step = {}, None
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        require_matching_policy(payload.get("performance"), policy, "performance")
        check_decoder_compatibility(payload, world_model, config, decoder)
        decoder.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        sampler.generator.set_state(payload["sampler_rng"])
        generator.set_state(payload["chunk_rng"])
        rng_restore(payload)
        history = list(payload.get("history", []))
        best = dict(payload.get("best", best))
        metrics, metrics_step = dict(payload.get("metrics") or {}), payload.get("metrics_step")
        print(f"resumed decoder from {resume} at step {step}", flush=True)
    else:
        save_decoder_checkpoint(out_dir / "initial.pt", decoder, world_model, config, optimizer, 0,
                                sampler, generator, provenance, history, best, world_checkpoint,
                                performance=policy)
    def record_evaluation(reason: str) -> None:
        """Score the current weights, log them, and keep ``best.pt`` up to date.

        Called both on the periodic schedule and once more after a wall-budget stop:
        the final weights deserve exactly the same best-selection as any other
        evaluated checkpoint, otherwise a run that stops before its first scheduled
        evaluation would report ``best = infinity`` and leave no ``best.pt`` at all.
        """
        nonlocal metrics, metrics_step, best
        metrics = evaluate_decoder(world_model, decoder, val_set, frame_source, config, device,
                                   settings.eval_batches, reference=reference)
        metrics_step = step
        entry = {"step": step, "val": metrics}
        if reason:
            entry["reason"] = reason
        history.append(entry)
        print(f"[step {step}] val l1={metrics['l1']:.4f} mse={metrics['mse']:.5f} "
              f"psnr={metrics['psnr_db']:.2f}dB", flush=True)
        if metrics["l1"] < best["val_l1"]:
            best = {"val_l1": metrics["l1"], "val_mse": metrics["mse"],
                    "val_psnr_db": metrics["psnr_db"], "step": step}
            save_decoder_checkpoint(out_dir / "best.pt", decoder, world_model, config, optimizer,
                                    step, sampler, generator, provenance, history, best,
                                    world_checkpoint, metrics, metrics_step, policy)

    started = time.monotonic()
    log_path = out_dir / "train_log.jsonl"
    decoder.train()
    while step < settings.max_steps:
        step += 1
        windows = sampler.sample(settings.batch_windows)
        latents, targets, _ = gather_decoder_batch(world_model, frame_source, train_set, windows,
                                                   device, generator=generator,
                                                   performance=performance)
        with autocast_context(performance, device):
            prediction = decoder(latents)
        # the reconstruction loss is evaluated in float32 whatever the model computed in
        loss, tensor_parts = decoder_loss(prediction, targets, settings.l1_weight,
                                          settings.edge_weight, stats_mode="tensor")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # the norm stays a tensor; only logging reads it back to the host
        grad_norm = nn.utils.clip_grad_norm_(decoder.parameters(), settings.grad_clip_norm)
        optimizer.step()
        should_log = step % settings.log_interval == 0 or step == 1
        should_eval = step % settings.eval_interval == 0 or step == settings.max_steps
        if should_log or should_eval:
            record = {"step": step, "loss": float(loss.detach()), **materialize_stats(tensor_parts),
                      "grad_norm": float(grad_norm), "elapsed": time.monotonic() - started}
        if should_log:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if should_eval:
            record_evaluation("")
        if time.monotonic() - started > settings.max_wall_seconds:
            history.append({"step": step, "stopped": "wall_budget"})
            break
    if metrics_step != step:
        # the wall-clock budget can stop the run between evaluations: score the
        # checkpoint that is actually written, never report a stale one
        record_evaluation("final checkpoint evaluation")
    save_decoder_checkpoint(out_dir / "final.pt", decoder, world_model, config, optimizer, step,
                            sampler, generator, provenance, history, best, world_checkpoint, metrics,
                            metrics_step, policy)
    summary = {
        "kind": "rgb_decoder",
        "steps": step,
        "wall_seconds": time.monotonic() - started,
        "best": best,
        "final_val": metrics,
        "decoder_config": vars(decoder.config),
        "decoder": {
            "parameters": decoder.parameter_count(),
            "image_size": decoder.config.image_size,
            "base_channels": decoder.config.base_channels,
            "channel_multipliers": list(decoder.config.channel_multipliers),
            "grid": list(decoder.grid),
            "d_world": decoder.d_world,
        },
        "world_checkpoint": world_checkpoint,
        "splits": provenance["splits"],
        "provenance": provenance,
        "performance": policy,
        "decoded_frame_cache": frame_source.counters(),
        "units": METRIC_NOTE,
        "notes": [
            "the decoder reconstructs one keyframe per chunk (the last sampled frame); it is not "
            "a video generator and ordered keyframes are not a continuous clip",
            "reconstruction from a frozen encoder plus a random projection is lossy: expect blur "
            "and missing texture even on true-target reconstruction",
        ],
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                encoding="utf-8")
    return summary


def run_decoder_training(config: RunConfig, world_checkpoint, out_dir, device,
                         resume: str = "") -> dict:
    """End-to-end decoder training from a world checkpoint: the CLI entry point.

    Order of operations, all before the first gradient step:

    1. the world model is loaded and frozen;
    2. the caller's configuration is checked against the world checkpoint's own
       recorded configuration (encoder identity and token preprocessing), so a
       decoder can never be trained on a different token space than the world's;
    3. a resume checkpoint, when given, defines the decoder -- its architecture,
       including the RGB output size, is authoritative, and ``config.decoder`` must
       match it instead of quietly overriding it;
    4. the split comes from the world provenance, the caches are checked against the
       world model, and every source video is checked against its cache.

    ``config.decoder_train.seed`` is applied *before* the decoder is constructed, so
    an initial checkpoint and a resumed run reproduce the same trajectory.
    """
    from .model import load_decoder

    world_checkpoint = Path(world_checkpoint)
    set_determinism(config.decoder_train.seed)
    world_model, world_payload = VideoWorldModel.from_checkpoint(world_checkpoint,
                                                                 map_location="cpu")
    world_model = world_model.to(device).eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)
    if resume:
        decoder, payload = load_decoder(resume, map_location="cpu")
        check_decoder_compatibility(payload, world_model, config, decoder)
        stored = decoder_architecture(decoder.config)
        if stored != decoder_architecture(config.decoder):
            raise DecoderCompatibilityError(
                "config.decoder does not match the decoder being resumed:\n"
                f"  - checkpoint architecture: {stored}\n"
                f"  - config architecture:     {decoder_architecture(config.decoder)}\n"
                "The checkpoint is authoritative, so either align the configuration or start a new "
                "decoder run; a checkpoint and a configuration must not disagree about what was "
                "trained."
            )
        print(f"continuing decoder from {resume} (step {payload.get('step')}, output "
              f"{decoder.config.image_size}px)", flush=True)
    else:
        decoder = build_decoder(config.decoder, world_model.patches, world_model.config.d_world)
    decoder = decoder.to(device)
    train_set, val_set, frame_source, splits = prepare_decoder_data(
        config, world_payload, decoder.config.image_size, world_checkpoint, world_model
    )
    print(f"decoder: {decoder.parameter_count():,} parameters, output "
          f"{decoder.config.image_size}x{decoder.config.image_size} for a "
          f"{decoder.grid[0]}x{decoder.grid[1]} patch grid, d_world={decoder.d_world}", flush=True)
    record = {
        "path": str(world_checkpoint),
        "step": world_payload.get("step"),
        "patches": int(world_model.patches),
        "d_world": int(world_model.config.d_world),
        "chunk_seconds": float(world_model.chunk_seconds),
        "source_signature": (world_payload.get("provenance") or {}).get("source_signature"),
    }
    provenance = {
        "source": "wpm-video train-decoder",
        "splits": splits,
        "world_checkpoint": record,
        "target_frames": {"semantics": decoder_identity(decoder, world_model, config)["frame_target"]},
        "train_cache": {key: value for key, value in train_set.provenance().items()
                        if key != "window_keys"},
        "val_cache": {key: value for key, value in val_set.provenance().items()
                      if key != "window_keys"},
        "config": config.to_dict(),
    }
    return train_decoder(config, world_model, decoder, frame_source, train_set, val_set, out_dir,
                         device, resume=resume or "", provenance=provenance,
                         world_checkpoint=record, splits=splits)
