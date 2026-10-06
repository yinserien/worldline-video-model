"""Measure synthetic world/decoder training steps using the installed package.

Requires --out; keep that directory outside the package. No video or encoder weights
are downloaded. Use an external --config to measure your model dimensions and
performance policy. This measures compute, not data loading or model quality.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
from wpm_video import RunConfig, VideoWorldModel, __version__, build_decoder
from wpm_video.dataset import Window, gather_batch
from wpm_video.decoder.train import decoder_loss
from wpm_video.performance import apply_compile, autocast_context, build_optimizer
from wpm_video.train import forward_window, source_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--decoder-batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--encoder-width", type=int, default=1024)
    parser.add_argument("--grid-side", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    for name in ("batch", "decoder_batch", "steps", "warmup", "repeats", "encoder_width", "grid_side"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    config = RunConfig.load(args.config) if args.config else RunConfig()
    config.validate()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(config.train.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    patches = args.grid_side ** 2
    record = {
        "package_version": __version__, "source_signature": source_signature(),
        "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "seed": config.train.seed, "config": config.to_dict(), "threads": torch.get_num_threads(),
        "batch_windows": args.batch, "decoder_batch": args.decoder_batch,
        "patches": patches, "encoder_width": args.encoder_width,
        "warmup": args.warmup, "steps_per_repeat": args.steps, "repeats": args.repeats,
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn": torch.backends.cudnn.allow_tf32},
        "scope": "Synthetic GPU/CPU resident input, forward/backward/gradient clip/AdamW. "
                 "Excludes pretrained encoder, video IO, target cache creation, batch preparation, "
                 "evaluation and checkpoints. Measures performance, not trained model quality.",
        "cases": {},
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(name, model, step, items):
        for _ in range(args.warmup):
            loss = step()
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.repeats):
            synchronize()
            started = time.perf_counter()
            for _ in range(args.steps):
                loss = step()
            synchronize()
            times.append((time.perf_counter() - started) * 1000 / args.steps)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError(f"{name} produced a non-finite final loss")
        median = statistics.median(times)
        result = {"median_ms_per_step": median, "items_per_second": items * 1000 / median,
                  "repetitions_ms": times, "items_per_step": items, "last_loss": value,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        record["cases"][name] = result
        (args.out / "benchmark.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({name: result}), flush=True)
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
                step()
                synchronize()
            profiler.export_chrome_trace(str(args.out / f"{name}_trace.json"))
            key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
            (args.out / f"{name}_operators.txt").write_text(
                profiler.key_averages().table(sort_by=key, row_limit=30), encoding="utf-8")

    def world_case():
        model = VideoWorldModel(config.model, args.encoder_width, patches, config.data.chunk_seconds,
                                anchor_attention=config.performance.anchor_attention).to(device).train()
        record["world_compile"] = apply_compile(model, config.performance)
        count = config.data.context_chunks + max(config.data.horizon_chunks)
        tokens = torch.randn(count, patches, args.encoder_width)

        class SyntheticDataset:
            def __init__(self):
                self.config = config.data

            def chunk(self, video, index):
                start = index * config.data.chunk_stride_frames / config.data.fps
                return {"index": index, "start_seconds": start,
                        "end_seconds": start + config.data.chunk_seconds}

            def tokens(self, video, index):
                return tokens[index]

        windows = [Window("synthetic", "train", 0, count, count * config.data.chunk_seconds)
                   for _ in range(args.batch)]
        batch = gather_batch(SyntheticDataset(), windows, device, model_config=config.model,
                             performance=config.performance)
        optimizer = build_optimizer(model.parameters(), config.performance, config.train.learning_rate,
                                    config.train.weight_decay, device)

        def step():
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config.performance, device):
                loss, _ = forward_window(model, batch, config, sample=True, stats_mode="tensor")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            optimizer.step()
            return loss

        measure("world", model, step, args.batch)

    world_case()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    decoder = build_decoder(config.decoder, patches, config.model.d_world).to(device).train()
    latents = torch.randn(args.decoder_batch, patches, config.model.d_world, device=device)
    targets = torch.rand(args.decoder_batch, 3, config.decoder.image_size, config.decoder.image_size, device=device)
    optimizer = build_optimizer(decoder.parameters(), config.performance, config.decoder_train.learning_rate,
                                config.decoder_train.weight_decay, device)

    def decoder_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config.performance, device):
            prediction = decoder(latents)
        loss, _ = decoder_loss(prediction, targets, config.decoder_train.l1_weight,
                               config.decoder_train.edge_weight, stats_mode="tensor")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.decoder_train.grad_clip_norm)
        optimizer.step()
        return loss

    measure("decoder", decoder, decoder_step, args.decoder_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
