"""Training loop with validation, checkpoints and resume.

Unit contract used everywhere in this module:

- every reported NLL and KL is **bits per dimension**, averaged over the batch,
  over the spatial patches and over the latent dimensions;
- ``bits_per_event`` multiplies that by patches x dimensions when a per-chunk
  total is wanted.

Checkpoints keep the model, the frozen projection and standardisation buffers
(part of the state dict), the optimizer, the step counter, the sampler generator
and both the CPU and every CUDA RNG stream, so a resumed run continues the same
trajectory instead of an approximation of it. Validation uses the deterministic
posterior mean and consumes no randomness, so scoring cannot perturb training.
"""

from dataclasses import asdict
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .config import RunConfig
from .dataset import TokenDataset, WindowSampler, check_split_integrity, gather_batch
from .model import VideoWorldModel, gaussian_kl_bits, gaussian_nll_bits
from .world_state import WorldState


def source_signature(package: Path | None = None) -> str:
    """Hash of the installed package sources, so a run records which code produced it.

    Recursive, so subpackages (``wpm_video/decoder``) are part of the signature and
    a decoder training run records the decoder code that produced it. Defaults to the
    directory this module was imported from, which works for a wheel install where no
    ``src/`` tree exists.
    """
    directory = Path(package) if package is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_snapshot() -> dict:
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def rng_restore(snapshot: dict) -> None:
    torch.set_rng_state(snapshot["torch_rng"])
    if snapshot.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["cuda_rng"])


def build_model(config: RunConfig, encoder, device) -> VideoWorldModel:
    model = VideoWorldModel(config.model, encoder.d_model, encoder.patches, config.data.chunk_seconds)
    return model.to(device)


@torch.no_grad()
def fit_projection(model: VideoWorldModel, dataset: TokenDataset, max_tokens: int = 200_000) -> int:
    """Fit the frozen projection's standardisation on training tokens only."""
    collected, used = [], 0
    for window in dataset.windows:
        tokens = dataset.entries[window.video].tokens
        block = tokens[window.start:window.start + dataset.config.context_chunks]
        collected.append(block.reshape(-1, block.shape[-1]))
        used += block.shape[0] * block.shape[1]
        if used >= max_tokens:
            break
    model.projection.fit_standardisation(torch.cat(collected).float())
    return used


def stream_observed(model: VideoWorldModel, batch: dict, sample: bool, generator=None):
    """Observe every context chunk in order; returns the per-chunk states and diagnostics."""
    observed = batch["observed"]
    state = model.initial_state(observed.shape[0], observed.device, observed.dtype)
    states, diagnostics = [], []
    for index in range(observed.shape[1]):
        state, record = model.observe(
            state, observed[:, index], batch["end_seconds"][:, index], sample=sample, generator=generator
        )
        states.append(state)
        diagnostics.append(record)
    return states, diagnostics


def forward_window(model: VideoWorldModel, batch: dict, config: RunConfig, sample: bool = True,
                   generator=None):
    """Stream the observed context and score the multi-horizon objective."""
    states, diagnostics = stream_observed(model, batch, sample, generator)
    model_cfg = config.model
    horizon_sum, horizon_items, squared_sum = None, 0, 0.0
    per_horizon = {}
    for (offset, horizon), entry in batch["targets"].items():
        valid = entry["valid"]
        index = valid.nonzero().flatten()
        if index.numel() == 0:
            continue
        mu, logvar, _, _ = model.predict(states[offset][index], entry["delta_seconds"][index])
        target = model.project(entry["tokens"][index])
        # bits per dimension: mean over patches and dims, then summed over the batch
        bits = gaussian_nll_bits(target, mu, logvar).mean(dim=(1, 2))
        horizon_sum = bits.sum() if horizon_sum is None else horizon_sum + bits.sum()
        squared_sum += float((target - mu).detach().pow(2).mean(dim=(1, 2)).sum())
        horizon_items += int(index.numel())
        per_horizon.setdefault(horizon, []).append(float(bits.mean().detach()))
    horizon_nll = horizon_sum / max(horizon_items, 1)
    present_mu, present_logvar = model.present_estimate(states[-1])
    present_bits = gaussian_nll_bits(model.project(batch["observed"][:, -1]), present_mu,
                                     present_logvar).mean()
    prior_bits = torch.stack([record["prior_nll_bits_per_dim"] for record in diagnostics]).mean()
    kl_bits = torch.stack([record["kl_bits_per_dim"] for record in diagnostics]).mean()
    loss = (
        model_cfg.w_horizon * horizon_nll
        + model_cfg.w_prior * prior_bits
        + model_cfg.w_present * present_bits
        + model_cfg.kl_beta * kl_bits
    )
    stats = {
        "loss": float(loss.detach()),
        "nll_bits_per_dim_horizon": float(horizon_nll.detach()),
        "mse": squared_sum / max(horizon_items, 1),
        "prior_nll_bits_per_dim": float(prior_bits.detach()),
        "present_nll_bits_per_dim": float(present_bits.detach()),
        "kl_bits_per_dim": float(kl_bits.detach()),
        "kl_bits_per_event": float(kl_bits.detach()) * model.patches * model.config.d_world,
        "n_items": horizon_items,
    }
    for horizon, values in per_horizon.items():
        stats[f"nll_h{horizon}_bits_per_dim"] = sum(values) / len(values)
    return loss, stats


@torch.no_grad()
def evaluate_model(model: VideoWorldModel, dataset: TokenDataset, config: RunConfig, device,
                   batches: int, sampler: WindowSampler | None = None) -> dict:
    """Deterministic validation: posterior mean, per anchor/horizon, no RNG use."""
    model.eval()
    sampler = sampler or WindowSampler(dataset, seed=1234)
    totals = {}
    for _ in range(batches):
        windows = sampler.sample(config.train.batch_windows)
        batch = gather_batch(dataset, windows, device)
        states, _ = stream_observed(model, batch, sample=False)
        for (offset, horizon), entry in batch["targets"].items():
            valid = entry["valid"]
            index = valid.nonzero().flatten()
            if index.numel() == 0:
                continue
            mu, logvar, _, _ = model.predict(states[offset][index], entry["delta_seconds"][index])
            target = model.project(entry["tokens"][index])
            bits = gaussian_nll_bits(target, mu, logvar).mean(dim=(1, 2))
            key = (offset, horizon)
            record = totals.setdefault(key, {"nll": 0.0, "mse": 0.0, "n": 0, "delta": 0.0})
            record["nll"] += float(bits.sum())
            record["mse"] += float((target - mu).pow(2).mean(dim=(1, 2)).sum())
            record["n"] += int(index.numel())
            # sum of the real per-sample deltas; dividing by n gives their true mean
            record["delta"] += float(entry["delta_seconds"][index].sum())
    model.train()
    metrics = {}
    for (offset, horizon), record in sorted(totals.items()):
        metrics[f"offset_{offset}_horizon_{horizon}"] = {
            "nll_bits_per_dim": record["nll"] / record["n"],
            "mse": record["mse"] / record["n"],
            "n": record["n"],
            "delta_seconds": record["delta"] / record["n"],
        }
    grouped = {}
    for (offset, horizon), record in sorted(totals.items()):
        entry = grouped.setdefault(horizon, {"nll": 0.0, "mse": 0.0, "n": 0, "delta": 0.0})
        entry["nll"] += record["nll"]
        entry["mse"] += record["mse"]
        entry["n"] += record["n"]
        entry["delta"] += record["delta"]
    return {
        "per_anchor_horizon": metrics,
        "per_horizon": {
            f"horizon_{horizon}": {
                "nll_bits_per_dim": entry["nll"] / entry["n"],
                "mse": entry["mse"] / entry["n"],
                "n": entry["n"],
                "delta_seconds": entry["delta"] / entry["n"],
            }
            for horizon, entry in sorted(grouped.items())
        },
    }


def save_checkpoint(path: Path, model, optimizer, step: int, config: RunConfig, sampler, provenance: dict,
                    history: list, best: dict) -> None:
    torch.save({
        "schema_version": 2,
        "model_config": asdict(config.model),
        "patches": model.patches,
        "chunk_seconds": model.chunk_seconds,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "sampler_rng": sampler.generator.get_state(),
        **rng_snapshot(),
        "provenance": provenance,
        "history": history[-200:],
        "best": best,
    }, path)


def train(config: RunConfig, model: VideoWorldModel, train_set: TokenDataset, val_set: TokenDataset,
          out_dir: Path, device, resume: str = "", provenance: dict | None = None) -> dict:
    check_split_integrity(train_set, val_set)
    set_determinism(config.train.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = dict(provenance or {})
    provenance["source_signature"] = source_signature()
    provenance["train_windows"] = len(train_set)
    provenance["val_windows"] = len(val_set)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    sampler = WindowSampler(train_set, seed=config.train.seed)
    step, history = 0, []
    best = {"val_mean_nll_bits_per_dim": math.inf, "step": 0}
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        sampler.generator.set_state(payload["sampler_rng"])
        rng_restore(payload)
        history = list(payload.get("history", []))
        best = dict(payload.get("best", best))
    else:
        save_checkpoint(out_dir / "initial.pt", model, optimizer, 0, config, sampler, provenance,
                        history, best)
    started = time.monotonic()
    log_path = out_dir / "train_log.jsonl"
    model.train()
    while step < config.train.max_steps:
        step += 1
        windows = sampler.sample(config.train.batch_windows)
        batch = gather_batch(train_set, windows, device)
        loss, stats = forward_window(model, batch, config, sample=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.train.grad_clip_norm
        ))
        optimizer.step()
        stats.update({"step": step, "grad_norm": grad_norm, "elapsed": time.monotonic() - started})
        if step % config.train.log_interval == 0 or step == 1:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(stats) + "\n")
        if step % config.train.eval_interval == 0 or step == config.train.max_steps:
            metrics = evaluate_model(model, val_set, config, device, config.train.eval_batches)
            per_horizon = metrics["per_horizon"]
            mean_nll = sum(v["nll_bits_per_dim"] for v in per_horizon.values()) / len(per_horizon)
            history.append({"step": step, "val_mean_nll_bits_per_dim": mean_nll, "per_horizon": per_horizon})
            if mean_nll < best["val_mean_nll_bits_per_dim"]:
                best = {"val_mean_nll_bits_per_dim": mean_nll, "step": step}
                save_checkpoint(out_dir / "best.pt", model, optimizer, step, config, sampler, provenance,
                                history, best)
        if time.monotonic() - started > config.train.max_wall_seconds:
            history.append({"step": step, "stopped": "wall_budget"})
            break
    save_checkpoint(out_dir / "final.pt", model, optimizer, step, config, sampler, provenance, history, best)
    summary = {
        "steps": step,
        "wall_seconds": time.monotonic() - started,
        "best": best,
        "final_val": history[-1].get("per_horizon") if history else {},
        "provenance": provenance,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                encoding="utf-8")
    return summary
