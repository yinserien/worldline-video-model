"""Command line interface: ``wpm-video <command>`` or ``python -m wpm_video``.

The CLI contains no repository-relative imports and no sys.path manipulation; it
works the same from a source checkout and from an installed wheel.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from . import __version__
from .config import RunConfig
from .data import PUBLIC_DATASET, build_source_manifest, list_videos, load_source_manifest
from .dataset import TokenDataset, cache_video_tokens, prepare_split_lists
from .encoder import cache_path
from .encoder import build_encoder
from .evaluate import baseline_summary, compare, evaluate_baselines, fit_baseline_stats
from .model import VideoWorldModel
from .predict import demo, query_saved_state
from .train import build_model, evaluate_model, fit_projection, source_signature, train
from .viz import fit_pca, plot_frames, plot_future_pca, plot_uncertainty


def load_config(path: Path) -> RunConfig:
    config = RunConfig.load(path)
    Path(config.data.cache_dir).mkdir(parents=True, exist_ok=True)
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


def _load_model(path: Path, device):
    model, payload = VideoWorldModel.from_checkpoint(path, map_location="cpu")
    return model.to(device), payload


def _splits(config: RunConfig, args) -> dict:
    if getattr(args, "train_videos", None) and getattr(args, "val_videos", None):
        split = {"train": sorted(args.train_videos), "val": sorted(args.val_videos),
                 "source": "command line"}
    elif config.data.train_videos and config.data.val_videos:
        split = {"train": sorted(config.data.train_videos), "val": sorted(config.data.val_videos),
                 "source": "config"}
    else:
        split = prepare_split_lists(Path(config.data.video_dir), val_count=args.val_count)
        split["source"] = "auto by filename order"
    config.data.train_videos = split["train"]
    config.data.val_videos = split["val"]
    config.validate()
    return split


def command_cache(args, config: RunConfig) -> int:
    device = resolve_device(config.encoder.device)
    encoder = build_encoder(config.encoder, device, allow_native=args.allow_native)
    if not encoder.is_pretrained and not args.allow_native:
        raise SystemExit("refusing to cache with the randomly initialized native encoder "
                         "without --allow-native")
    videos = list_videos(Path(config.data.video_dir))
    if not videos:
        raise SystemExit(f"no .mp4 files in {config.data.video_dir}")
    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": {"kind": encoder.kind, "model_id": encoder.model_id,
                    "revision": encoder.revision, "d_model": encoder.d_model,
                    "patches": encoder.patches, "pretrained": bool(encoder.is_pretrained)},
        "data": {"fps": config.data.fps, "chunk_frames": config.data.chunk_frames,
                 "chunk_stride_frames": config.data.chunk_stride_frames,
                 "image_size": config.data.image_size},
        "source_signature": source_signature(),
        "videos": [],
    }
    started = time.monotonic()
    source_manifest = load_source_manifest(Path(config.data.video_dir))
    manifest["source_manifest_entries"] = len(source_manifest)
    for video in videos:
        meta = cache_video_tokens(video, config.data, encoder, Path(config.data.cache_dir),
                                  batch_clips=config.encoder.batch_clips,
                                  source_manifest=source_manifest)
        manifest["videos"].append(meta)
        print(f"cached {meta['video']}: {len(meta['chunks'])} chunks", flush=True)
    manifest["wall_seconds"] = time.monotonic() - started
    out = Path(args.out or Path(config.data.cache_dir) / "manifest.json")
    write_json(out, manifest)
    print(f"cache manifest: {out} ({manifest['wall_seconds']:.1f}s)")
    return 0


def command_train(args, config: RunConfig) -> int:
    device = resolve_device(config.train.device)
    split = _splits(config, args)
    out_dir = Path(args.out or "runs/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.json")
    write_json(out_dir / "splits.json", split)
    if args.resume:
        model, _ = _load_model(Path(args.resume), device)
    else:
        encoder = build_encoder(config.encoder, device, allow_native=args.allow_native)
        if not encoder.is_pretrained and not args.allow_native:
            raise SystemExit("refusing to train against the randomly initialized native encoder "
                             "without --allow-native")
        model = build_model(config, encoder, device)
    cache_dir = Path(config.data.cache_dir)
    train_set = TokenDataset(config.data, cache_dir, config.encoder, "train")
    val_set = TokenDataset(config.data, cache_dir, config.encoder, "val")
    if not args.resume:
        used = fit_projection(model, train_set)
        print(f"fitted projection standardisation on {used} train tokens", flush=True)
    provenance = {
        "splits": split,
        "train": {k: v for k, v in train_set.provenance().items() if k != "window_keys"},
        "val": {k: v for k, v in val_set.provenance().items() if k != "window_keys"},
        "config": config.to_dict(),
        "source": "wpm-video train",
        "source_signature": source_signature(),
    }
    summary = train(config, model, train_set, val_set, out_dir, device, resume=args.resume or "",
                    provenance=provenance)
    print(json.dumps(summary, indent=2, default=float))
    return 0


def command_eval(args, config: RunConfig) -> int:
    device = resolve_device(config.train.device)
    out_dir = Path(args.out or "runs/eval")
    if args.checkpoints:
        checkpoints = {path.stem: path for path in args.checkpoints}
    else:
        run_dir = Path(args.checkpoint).parent
        checkpoints = {name: run_dir / f"{name}.pt" for name in ("initial", "best", "final")}
        checkpoints = {name: path for name, path in checkpoints.items() if path.is_file()}
    if not checkpoints:
        raise SystemExit("no checkpoints to evaluate")
    model, payload = _load_model(next(iter(checkpoints.values())), device)
    splits = (payload.get("provenance") or {}).get("splits") or {}
    if not splits.get("train") or not splits.get("val"):
        raise SystemExit(
            "wpm-video eval needs a checkpoint trained by this package (it scores the "
            "registered validation split); this checkpoint carries no split provenance. "
            "Use `wpm-video predict` for inference from an arbitrary checkpoint."
        )
    config.data.train_videos = list(splits["train"])
    config.data.val_videos = list(splits["val"])
    config.validate()
    cache_dir = Path(config.data.cache_dir)
    val_set = TokenDataset(config.data, cache_dir, config.encoder, "val")
    train_set = TokenDataset(config.data, cache_dir, config.encoder, "train")
    started = time.monotonic()
    stats = fit_baseline_stats(model, train_set, config, device, args.batches)
    baseline_metrics = evaluate_baselines(model, val_set, config, device, args.batches, stats)
    results = {}
    for name, path in sorted(checkpoints.items()):
        loaded, checkpoint_payload = _load_model(path, device)
        model_metrics = evaluate_model(loaded, val_set, config, device, args.batches)
        table = compare(model_metrics, baseline_metrics, config)
        results[name] = {
            "checkpoint": str(path),
            "step": checkpoint_payload.get("step"),
            "per_horizon": model_metrics["per_horizon"],
            "per_anchor_horizon": table["per_anchor_horizon"],
            "summary": table["summary"],
        }
        print(f"[{name}] step {results[name]['step']} "
              f"nll={table['summary']['model_mean_nll_bits_per_dim']:.4f} "
              f"mse={table['summary']['model_mean_mse']:.4f} "
              f"beats_best_baseline_nll={table['summary']['model_beats_best_baseline_nll_on']}",
              flush=True)
    initial = results.get("initial", {}).get("summary")
    best = results.get("best", {}).get("summary")
    improvement = None
    if initial and best:
        improvement = {
            "best_minus_initial_nll_bits_per_dim": (best["model_mean_nll_bits_per_dim"]
                                                    - initial["model_mean_nll_bits_per_dim"]),
            "best_minus_initial_mse": best["model_mean_mse"] - initial["model_mean_mse"],
            "note": "negative means the trained checkpoint improves on the untrained one",
        }
    write_json(out_dir / "metrics.json", {
        "checkpoints": results,
        "baselines": baseline_metrics,
        "baseline_summary": baseline_summary(baseline_metrics),
        "learning_check_initial_vs_best": improvement,
        "validation_provenance": {k: v for k, v in val_set.provenance().items()
                                  if k != "window_keys"},
        "wall_seconds": time.monotonic() - started,
    })
    print(json.dumps(baseline_summary(baseline_metrics), indent=2, default=float))
    if improvement:
        print(json.dumps(improvement, indent=2, default=float))
    return 0


def optional_train_latents(model, config: RunConfig, device, limit: int = 64):
    """Latents for the optional PCA plot, or ``(None, reason)`` when unavailable.

    This is the only part of prediction that touches the training split. It is a
    visualisation aid, so a missing checkpoint provenance, a missing training cache
    or a cache without a usable split degrades to "skipped" with a stated reason
    instead of failing the run. Prediction itself never needs the training data,
    and PCA is never fitted on the inference video.
    """
    provenance = getattr(model, "checkpoint_provenance", None) or {}
    splits = (provenance.get("splits") or {})
    if not splits.get("train"):
        return None, "checkpoint carries no training split provenance"
    data = config.data
    data.train_videos = list(splits["train"])
    data.val_videos = list(splits.get("val", []))
    cache_dir = Path(data.cache_dir)
    missing = [name for name in data.train_videos
               if not cache_path(cache_dir, name, data, config.encoder).is_file()]
    if missing:
        return None, (f"training token cache is not available here "
                      f"({len(missing)} of {len(data.train_videos)} clips missing)")
    try:
        train_set = TokenDataset(data, cache_dir, config.encoder, "train")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return None, f"training split unusable for PCA: {error}"
    with torch.no_grad():
        latents = []
        for window in train_set.windows[:limit]:
            block = train_set.entries[window.video].tokens[
                window.start:window.start + data.context_chunks]
            latents.append(model.project(block.float().to(device)).cpu())
    return torch.cat(latents), None


def command_predict(args, config: RunConfig) -> int:
    """Stream a video with a checkpoint; requires only the checkpoint, the encoder
    config and the video. No training data or split provenance is needed."""
    device = resolve_device(config.train.device)
    model, payload = _load_model(Path(args.checkpoint), device)
    model.checkpoint_provenance = payload.get("provenance") or {}
    encoder = build_encoder(config.encoder, device, allow_native=args.allow_native)
    out_dir = Path(args.out or "runs/predict")
    result = demo(model, encoder, config, Path(args.video), device, out_dir,
                  prefix_chunks=args.prefix_chunks,
                  state_path=Path(args.state) if args.state else None)
    plot_uncertainty(result["predictions"], out_dir / "uncertainty.png")
    plot_frames(result["frames"], result["prefix"][-1].end_frame,
                result["future"][0].start_frame if result["future"] else None,
                out_dir / "frames.png", [float(t) for t in result["timestamps"]])
    latents, skipped_reason = optional_train_latents(model, config, device)
    pca_meta = {"fitted_on": "train split only", "optional": True,
                "checkpoint_has_provenance": bool(model.checkpoint_provenance)}
    if latents is None:
        pca_meta.update({"status": "skipped", "reason": skipped_reason})
        print(f"PCA visualisation skipped: {skipped_reason}", flush=True)
    else:
        pca = fit_pca(latents)
        plot_future_pca(pca, result["predictions"], out_dir / "latent_pca.png",
                        title=result["info"].name)
        pca_meta.update({"status": "written", "n_fit_latents": pca["n_fit"],
                         "path": (out_dir / "latent_pca.png").name})
    write_json(out_dir / "pca_meta.json", pca_meta)
    print(json.dumps(result["summary"]["horizons"], indent=2, default=float))
    return 0


def command_query(args, config: RunConfig) -> int:
    """State-only future query: no video, no encoder, no ground truth."""
    device = resolve_device(config.train.device)
    model, _ = _load_model(Path(args.checkpoint), device)
    out_dir = Path(args.out or "runs/query")
    result = query_saved_state(model, Path(args.state), args.deltas, device, out_dir)
    print(json.dumps(result["summary"], indent=2, default=float))
    return 0


def command_provenance(args, config: RunConfig) -> int:
    """Write the local-file -> immutable public-dataset mapping, verifying bytes first."""
    revision = args.revision or PUBLIC_DATASET["revision"]
    snapshot = Path(args.snapshot) if args.snapshot else None
    if snapshot is None:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_root / f"datasets--{PUBLIC_DATASET['repo_id'].replace('/', '--')}"
        candidates = [p for p in (repo_dir / "snapshots").iterdir() if p.is_dir()] \
            if (repo_dir / "snapshots").is_dir() else []
        snapshot = next((p for p in candidates if p.name == revision), None)
        if snapshot is None:
            raise SystemExit(
                f"no local snapshot of {PUBLIC_DATASET['repo_id']} at revision {revision}; "
                "download it first (see examples/prepare_sample_dataset.md) or pass --snapshot"
            )
    manifest = build_source_manifest(Path(config.data.video_dir), snapshot, revision,
                                     repo_id=PUBLIC_DATASET["repo_id"])
    out = Path(args.out or (Path(config.data.video_dir) / "sources.json"))
    write_json(out, manifest)
    mapped = len(manifest["clips"])
    skipped = manifest.get("skipped", [])
    print(f"wrote {out}: {mapped} clips verified against the snapshot bytes")
    if skipped:
        print(f"not declared public (recorded as local_file): {skipped}")
    return 0


def command_selfcheck(args, config: RunConfig) -> int:
    from .selfcheck import run_selfcheck
    return run_selfcheck(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wpm-video", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"wpm-video {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "train", "eval", "predict", "query", "provenance", "selfcheck"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--out", type=Path, default=None)
        item.add_argument("--allow-native", action="store_true",
                          help="permit the randomly initialized native encoder (tests/examples only)")
        if name in ("train", "eval", "predict"):
            item.add_argument("--val-count", type=int, default=1)
        if name == "train":
            item.add_argument("--resume", type=Path, default=None)
            item.add_argument("--train-videos", nargs="*", default=None)
            item.add_argument("--val-videos", nargs="*", default=None)
        if name == "eval":
            item.add_argument("--checkpoint", required=True, type=Path)
            item.add_argument("--checkpoints", nargs="*", type=Path, default=None)
            item.add_argument("--batches", type=int, default=8)
        if name == "predict":
            item.add_argument("--checkpoint", required=True, type=Path)
            item.add_argument("--video", required=True, type=Path)
            item.add_argument("--prefix-chunks", type=int, default=3)
            item.add_argument("--state", type=Path, default=None,
                              help="continue from a saved persistent state")
        if name == "query":
            item.add_argument("--checkpoint", required=True, type=Path)
            item.add_argument("--state", required=True, type=Path)
            item.add_argument("--deltas", nargs="*", type=float, default=[1.0, 2.0, 4.0])
        if name == "provenance":
            item.add_argument("--revision", type=str, default=None)
            item.add_argument("--snapshot", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    commands = {
        "cache": command_cache, "train": command_train, "eval": command_eval,
        "predict": command_predict, "query": command_query,
        "provenance": command_provenance, "selfcheck": command_selfcheck,
    }
    try:
        return commands[args.command](args, config)
    except KeyboardInterrupt:
        print("interrupted; partial outputs were kept", file=sys.stderr)
        return 130
