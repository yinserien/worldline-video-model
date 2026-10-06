"""Streaming inference: persistent state across chunks, future prediction, demo artifacts.

The demo reads a real video prefix chunk by chunk with real source timestamps.
Each chunk is encoded, enters the state only through the innovation writer, and
after the prefix ends the state is advanced by real time with no further
observation. Prediction depends only on the state and a time delta, so

- a video without any future ground truth can still be predicted,
- a saved state can be queried directly with no video and no encoder at all.

Ground-truth future tokens are loaded only to score and to plot.
"""

import json
from pathlib import Path

import torch

from .config import RunConfig
from .data import Chunk, build_chunks, probe_video, read_frames
from .model import gaussian_nll_bits
from .world_state import WorldState


@torch.no_grad()
def stream_chunks(model, encoder, frames: torch.Tensor, chunks: list[Chunk], device,
                  state: WorldState | None = None) -> tuple[WorldState, list[dict]]:
    """Observe chunks in order with real end timestamps, carrying one persistent state.

    A chunk that ends at or before the state time was already observed by that
    state and is skipped, so continuing a saved stream never moves time backwards.
    """
    if state is None:
        state = model.initial_state(1, device, next(model.parameters()).dtype)
    log = []
    for chunk in chunks:
        if chunk.end_seconds <= float(state.time) + 1e-9:
            log.append({"chunk": chunk.index, "skipped": "already covered by the state",
                        "end_seconds": chunk.end_seconds, "state_time_seconds": float(state.time)})
            continue
        clip = frames[chunk.start_frame:chunk.end_frame]
        tokens = encoder.encode_clip(clip).to(device).unsqueeze(0)
        state, diagnostics = model.observe(state, tokens, chunk.end_seconds, sample=False)
        log.append({
            "chunk": chunk.index, "skipped": False,
            "start_seconds": chunk.start_seconds, "end_seconds": chunk.end_seconds,
            "state_time_seconds": float(state.time),
            "innovation_rms": float(diagnostics["innovation_rms"].mean()),
            "kl_bits_per_dim": float(diagnostics["kl_bits_per_dim"].mean()),
            "sampled": diagnostics["sampled"],
        })
    return state, log


@torch.no_grad()
def predict_at(model, state: WorldState, deltas_seconds) -> dict:
    """Predict future spatial latent distributions for explicit time deltas.

    This is the whole inference API: a state and deltas. No encoder, no video and
    no future ground truth are involved.
    """
    deltas = torch.as_tensor(deltas_seconds, dtype=torch.float32).reshape(-1)
    outputs = {}
    for delta in deltas.tolist():
        mu, logvar, advanced, attention = model.predict(state, float(delta))
        outputs[float(delta)] = {
            "delta_seconds": float(delta),
            "target_time_seconds": float(state.time) + float(delta),
            "mu": mu[0].detach().cpu(),
            "sigma": logvar[0].detach().exp().sqrt().cpu(),
            "scored": False,
        }
    return outputs


@torch.no_grad()
def predict_future(model, encoder, frames: torch.Tensor, state: WorldState, target_chunks: list[Chunk],
                   device, horizons: list[int], chunk_seconds: float) -> dict:
    """Predict future chunks from the state, scoring against ground truth when present.

    The query delta is taken from the target chunk's real end time minus the state
    time, not from an assumed constant chunk spacing.
    """
    outputs = {}
    base_time = float(state.time)
    for horizon in horizons:
        target_chunk = target_chunks[horizon - 1] if horizon - 1 < len(target_chunks) else None
        if target_chunk is not None:
            delta = float(target_chunk.end_seconds) - base_time
        else:
            delta = horizon * chunk_seconds
        mu, logvar, advanced, attention = model.predict(state, delta)
        entry = {
            "horizon_chunks": horizon,
            "delta_seconds": delta,
            "target_time_seconds": base_time + delta,
            "mu": mu[0].detach().cpu(),
            "sigma": logvar[0].detach().exp().sqrt().cpu(),
            "scored": False,
        }
        if target_chunk is not None:
            clip = frames[target_chunk.start_frame:target_chunk.end_frame]
            target = model.project(encoder.encode_clip(clip).to(device).unsqueeze(0))
            entry.update({
                "target": target[0].detach().cpu(),
                "target_start_seconds": target_chunk.start_seconds,
                "target_end_seconds": target_chunk.end_seconds,
                "scored": True,
                "nll_bits_per_dim": float(gaussian_nll_bits(target, mu, logvar).mean()),
                "mse": float((target - mu).pow(2).mean()),
            })
        outputs[horizon] = entry
    return outputs


def load_state(path, device, dtype=torch.float32) -> WorldState:
    """Load a saved persistent state; accepts str or Path."""
    return WorldState.load(Path(path), device=device, dtype=dtype)


def demo(model, encoder, config: RunConfig, video, device, out_dir,
         prefix_chunks: int = 3, horizons: list[int] | None = None,
         state_path: Path | None = None) -> dict:
    """Full streaming demo on one real video: state, futures, uncertainties, timestamps."""
    horizons = horizons or list(config.data.horizon_chunks)
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    info = probe_video(video)
    frames, timestamps = read_frames(video, config.data.fps, config.data.image_size)
    chunks = build_chunks(info, config.data, timestamps)
    if not chunks:
        raise ValueError(
            f"{video.name} is too short for one chunk at fps={config.data.fps} with "
            f"chunk_frames={config.data.chunk_frames}; use a longer clip or lower fps"
        )
    # a short clip is still predictable: observe what exists and say so explicitly
    effective_prefix = min(prefix_chunks, len(chunks))
    prefix_was_clamped = effective_prefix != prefix_chunks
    if prefix_was_clamped:
        print(f"prefix clamped to {effective_prefix} chunk(s): the video only has {len(chunks)}",
              flush=True)
    prefix, future = chunks[:effective_prefix], chunks[effective_prefix:]
    initial = load_state(state_path, device) if state_path else None
    continued = initial is not None
    state, log = stream_chunks(model, encoder, frames, prefix, device, state=initial)
    saved_state = out_dir / "world_state.pt"
    state.save(saved_state)
    reloaded = WorldState.load(saved_state, device=device, dtype=state.slots.dtype)
    predictions = predict_future(model, encoder, frames, state, future, device, horizons,
                                 config.data.chunk_seconds)
    reloaded_predictions = predict_future(model, encoder, frames, reloaded, future, device, horizons,
                                          config.data.chunk_seconds)
    reload_max_diff = max(
        float((predictions[h]["mu"] - reloaded_predictions[h]["mu"]).abs().max()) for h in horizons
    )
    payload = {
        "video": info.path, "video_name": info.name, "video_sha256": info.sha256,
        "source_id": info.source_id, "source_fps": info.fps,
        "chunk_seconds": config.data.chunk_seconds,
        "prefix_chunks_requested": prefix_chunks, "prefix_chunks": effective_prefix,
        "prefix_clamped": prefix_was_clamped, "chunks_in_video": len(chunks),
        "prefix_end_seconds": float(state.time), "stream_log": log,
        "continued_from_saved_state": continued,
    }
    torch.save({
        "predictions": {h: {k: v for k, v in entry.items() if torch.is_tensor(v)}
                        for h, entry in predictions.items()},
        "meta": payload,
    }, out_dir / "future_latents.pt")
    summary = {
        **payload,
        "state_path": saved_state.as_posix(),
        "state_time_seconds": float(state.time),
        "state_reload_max_abs_mu_difference": reload_max_diff,
        "horizons": {
            str(h): {k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in entry.items() if not torch.is_tensor(v)}
            for h, entry in predictions.items()
        },
    }
    (out_dir / "predict_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                  encoding="utf-8")
    return {"summary": summary, "frames": frames, "timestamps": timestamps, "prefix": prefix,
            "future": future, "predictions": predictions, "state": state, "info": info}


def query_saved_state(model, state_path, deltas_seconds, device, out_dir=None) -> dict:
    """State-only future query: no video, no encoder, no ground truth."""
    state_path = Path(state_path)
    state = load_state(state_path, device)
    outputs = predict_at(model, state, deltas_seconds)
    summary = {
        "state_path": state_path.as_posix(),
        "state_time_seconds": float(state.time),
        "state_step": int(state.step),
        "queries": {
            str(delta): {"target_time_seconds": entry["target_time_seconds"],
                         "mean_sigma": float(entry["sigma"].mean()),
                         "mu_std": float(entry["mu"].std())}
            for delta, entry in outputs.items()
        },
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({str(delta): {"mu": entry["mu"], "sigma": entry["sigma"]}
                    for delta, entry in outputs.items()}, out_dir / "state_query_latents.pt")
        (out_dir / "state_query.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                  encoding="utf-8")
    return {"summary": summary, "outputs": outputs}
