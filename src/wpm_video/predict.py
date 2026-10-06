"""Streaming inference: persistent state across chunks, future prediction, demo artifacts.

The demo reads a real video prefix chunk by chunk with real source timestamps.
Each chunk is encoded, enters the state only through the innovation writer, and
after the prefix ends the state is advanced by real time with no further
observation. Prediction depends only on the state and a time delta, so

- a video without any future ground truth can still be predicted,
- a saved state can be queried directly with no video and no encoder at all.

Ground-truth future tokens are loaded only to score and to plot.

The optional decoder is applied last and only to already-computed latents: a
predicted ``mu`` becomes one decoded keyframe per horizon, and a true future
latent becomes one reconstruction used to judge the decoder on its own. Future
ground truth never influences the state or any predicted pixel -- it is decoded
separately, into its own files, purely as a reference; no decoded image of any
kind is fed back into the state, the prediction or another image.
"""

import json
from pathlib import Path

import torch

from .config import RunConfig
from .data import Chunk, build_chunks, probe_video, read_frames
from .model import gaussian_nll_bits
from .world_state import WorldState


def frame_time_metadata(chunk_end_seconds: float, source_fps: float | None) -> dict:
    """Where a decoded keyframe sits on the source timeline.

    ``target_time_seconds`` is the *chunk end*: the decoded image is the chunk's last
    sampled frame, which was captured one source frame period earlier. With a known
    source fps both times are reported; without one (a state-only query, which has no
    video) the pixel time is explicitly unknown rather than silently equated with the
    chunk end.
    """
    chunk_end = float(chunk_end_seconds)
    if source_fps and source_fps > 0:
        return {
            "chunk_end_seconds": chunk_end,
            "target_frame_timestamp_seconds": chunk_end - 1.0 / float(source_fps),
            "source_fps": float(source_fps),
            "timestamp_basis": ("last sampled frame of the chunk: the chunk end minus one source "
                                "frame period"),
        }
    return {
        "chunk_end_seconds": chunk_end,
        "target_frame_timestamp_seconds": None,
        "source_fps": None,
        "timestamp_basis": ("unknown: no video was read, so the source fps (and with it the "
                            "capture time of the last sampled frame) is not known; the chunk end "
                            "is the query anchor, not the pixel time"),
    }


def _decode_entries(predictions: dict, source: str, source_fps: float | None = None) -> dict:
    """Latents to render for one set of predictions, with their time metadata.

    ``source`` is ``"predicted mu"`` for the model's own future estimate or
    ``"true future latent"`` for the decoder-only reconstruction; the two are never
    mixed in one artifact so a decoder error cannot be read as a prediction error.
    """
    entries = {}
    for horizon, entry in sorted(predictions.items()):
        key = "target" if source == "true future latent" else "mu"
        if key not in entry:
            continue
        # a scored target knows its own chunk end; a decoded reconstruction must use it
        anchor = entry.get("target_end_seconds", entry["target_time_seconds"])
        entries[f"h{horizon}"] = (entry[key], {
            "horizon_chunks": horizon,
            "delta_seconds": float(entry["delta_seconds"]),
            "target_time_seconds": float(entry["target_time_seconds"]),
            "latent_source": source,
            "frame_role": "last sampled frame of that future chunk",
            **frame_time_metadata(anchor, source_fps),
        })
    return entries


@torch.no_grad()
def decode_predictions(model, config: RunConfig, predictions: dict, decoder, decoder_payload: dict,
                       device, out_dir: Path, source_fps: float | None = None) -> dict:
    """Decode predicted and (when scored) true future latents into keyframe PNGs.

    Reused by ``predict``; ``query_saved_state`` uses the same helper through
    ``decode_query``. Both write sparse keyframes: the horizon order is a reading
    order, not a frame order of a continuous video.
    """
    from .decoder.compat import check_decoder_compatibility
    from .decoder.render import render_latents, write_artifact, write_records

    check_decoder_compatibility(decoder_payload, model, config, decoder)
    out_dir = Path(out_dir)
    meta = {
        "decoder_checkpoint": decoder_payload.get("path"),
        "images": {
            "decoded_h<horizon>.png": "decoder output for the PREDICTED future latent",
            "target_reconstruction_h<horizon>.png": "decoder output for the TRUE future latent",
        },
        "note": (
            "decoded keyframes: independent images ordered by horizon. They are not a continuous "
            "video and not ground truth; ground-truth pixels are shown separately."
        ),
    }
    predicted = render_latents(decoder, _decode_entries(predictions, "predicted mu", source_fps),
                              out_dir, device, payload=decoder_payload, prefix="decoded",
                              name="predicted")
    write_artifact(out_dir / "decoded_predictions.pt", predicted, meta)
    targets = render_latents(decoder,
                             _decode_entries(predictions, "true future latent", source_fps),
                             out_dir, device, payload=decoder_payload,
                             prefix="target_reconstruction", name="true_target_reconstruction")
    if targets["frames"]:
        write_artifact(out_dir / "decoded_targets.pt", targets, {
            **meta,
            "note": (
                "the decoder applied to the TRUE future latent: this measures the decoder alone, "
                "with prediction error removed."
            ),
        })
    write_records(out_dir / "decoded_summary.json", predicted, meta)
    return {"predictions": predicted, "targets": targets, "meta": meta}


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

    Deltas stay float64, like the state clock: the keys, the reported delta and the
    target time are the caller's exact times. Rounding them to float32 first would
    merge two distinct query times (1.0 and 1.000000001) into one entry before any
    artifact could tell them apart. Only the activations are float32.
    """
    deltas = torch.as_tensor(deltas_seconds, dtype=torch.float64).reshape(-1)
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
         state_path: Path | None = None, decoder=None, decoder_payload: dict | None = None) -> dict:
    """Full streaming demo on one real video: state, futures, uncertainties, timestamps.

    With ``decoder`` the already-computed latents are additionally decoded into
    keyframe PNGs and tensor artifacts. The decoder runs after the state, the
    predictions and every latent artifact are finished, so it cannot influence any
    of them.
    """
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
    # the ground-truth keyframe of each scored horizon, for the decoded comparison
    # figure only: it is real video, never an input to the state or the decoder
    target_frames = {
        horizon: frames[future[horizon - 1].end_frame - 1]
        for horizon, entry in predictions.items()
        if entry.get("scored") and horizon - 1 < len(future)
    }
    decoded = None
    if decoder is not None:
        decoded = decode_predictions(model, config, predictions, decoder, decoder_payload or {},
                                     device, out_dir, source_fps=info.fps)
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
        "decoded_rgb": None if decoded is None else {
            "enabled": True,
            "decoder_checkpoint": (decoder_payload or {}).get("path"),
            "image_size": (decoder_payload or {}).get("decoder_config", {}).get("image_size"),
            "projection_sha256": ((decoder_payload or {}).get("world_projection") or {}).get("sha256"),
            "predicted_keyframes": decoded["predictions"]["records"],
            "true_target_reconstructions": decoded["targets"]["records"],
            "note": ("decoded keyframes are sparse (one per horizon) and are not a continuous video; "
                     "they are reconstructions from latents, not ground truth"),
        },
    }
    (out_dir / "predict_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                  encoding="utf-8")
    return {"summary": summary, "frames": frames, "timestamps": timestamps, "prefix": prefix,
            "future": future, "predictions": predictions, "state": state, "info": info,
            "target_frames": target_frames, "decoded": decoded}


@torch.no_grad()
def decode_query(model, config: RunConfig, outputs: dict, decoder, decoder_payload: dict, device,
                 out_dir: Path) -> dict:
    """Decode state-only query latents into keyframe PNGs: no video, no encoder."""
    from .decoder.compat import check_decoder_compatibility
    from .decoder.render import render_latents, safe_name, write_artifact, write_records

    check_decoder_compatibility(decoder_payload, model, config, decoder)
    out_dir = Path(out_dir)
    entries = {
        # the full float repr keeps two near-identical deltas apart: ':g' would round
        # 1.0000001 and 1.0 onto the same file name
        f"d{safe_name(repr(float(delta)))}s": (entry["mu"], {
            "delta_seconds": float(delta),
            "target_time_seconds": float(entry["target_time_seconds"]),
            "latent_source": "predicted mu",
            "frame_role": "last sampled frame of the chunk ending at this time",
            # a state-only query has no video, so no source fps and no pixel time
            **frame_time_metadata(entry["target_time_seconds"], None),
        })
        for delta, entry in sorted(outputs.items())
    }
    meta = {
        "decoder_checkpoint": decoder_payload.get("path"),
        "note": ("state-only query: decoded from predicted latents with no video and no encoder. "
                 "Keyframes are ordered by delta, not a continuous video."),
    }
    artifact = render_latents(decoder, entries, out_dir, device, payload=decoder_payload,
                              prefix="decoded", name="state_query")
    write_artifact(out_dir / "state_query_decoded.pt", artifact, meta)
    write_records(out_dir / "decoded_summary.json", artifact, meta)
    return artifact


def query_saved_state(model, state_path, deltas_seconds, device, out_dir=None, decoder=None,
                      decoder_payload: dict | None = None, config: RunConfig | None = None) -> dict:
    """State-only future query: no video, no encoder, no ground truth.

    ``decoder`` additionally renders each queried latent as a keyframe image; the
    query itself stays latent-only and video-free either way. Decoding writes files,
    so it requires ``out_dir``: with no directory to write into the request is
    rejected rather than silently ignored.
    """
    if decoder is not None and out_dir is None:
        raise ValueError(
            "decoding a state query writes keyframe PNGs, so it needs out_dir; pass a directory "
            "or call predict_at() for latents only"
        )
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
        "decoded_rgb": None,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({str(delta): {"mu": entry["mu"], "sigma": entry["sigma"]}
                    for delta, entry in outputs.items()}, out_dir / "state_query_latents.pt")
        if decoder is not None:
            if config is None:
                raise ValueError("decoding a state query needs the run configuration for the "
                                 "compatibility check")
            artifact = decode_query(model, config, outputs, decoder, decoder_payload or {}, device,
                                    out_dir)
            summary["decoded_rgb"] = {
                "enabled": True,
                "decoder_checkpoint": (decoder_payload or {}).get("path"),
                "image_size": (decoder_payload or {}).get("decoder_config", {}).get("image_size"),
                "keyframes": artifact["records"],
                "note": "decoded keyframes from predicted latents; not a continuous video",
            }
        (out_dir / "state_query.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                  encoding="utf-8")
    return {"summary": summary, "outputs": outputs}
