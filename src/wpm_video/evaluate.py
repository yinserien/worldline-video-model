"""Held-out baselines and the like-for-like comparison table.

Baselines use the same validation windows, the same anchor/horizon pairs and the
same Gaussian NLL in bits per dimension as the model. Each baseline gets its own
constant variance fitted on training residuals, which is the strongest form of
that baseline; the model predicts its own variance instead. The comparison table
is per (anchor, horizon) so an aggregate is never matched against a single
hand-picked baseline entry.
"""

import json
import math
from pathlib import Path

import torch

from .config import RunConfig
from .dataset import TokenDataset, WindowSampler, gather_batch
from .model import gaussian_nll_bits

BASELINES = ("persistence", "linear_extrapolation", "train_mean")


def iter_windows(dataset: TokenDataset, config: RunConfig, device, batches: int, seed: int = 1234):
    sampler = WindowSampler(dataset, seed=seed)
    for _ in range(batches):
        windows = sampler.sample(config.train.batch_windows)
        yield windows, gather_batch(dataset, windows, device)


def baseline_prediction(kind: str, observed, offset: int, horizon: int, train_mean):
    """Baseline mean in the projected latent space of shape (n, P, D)."""
    if kind == "persistence":
        return observed[:, offset]
    if kind == "linear_extrapolation":
        if offset >= 1 and horizon >= 1:
            velocity = observed[:, offset] - observed[:, offset - 1]
            return observed[:, offset] + velocity * float(horizon)
        return observed[:, offset]
    if kind == "train_mean":
        return train_mean.unsqueeze(0).expand_as(observed[:, offset])
    raise ValueError(f"Unknown baseline: {kind}")


@torch.no_grad()
def fit_baseline_stats(model, train_set: TokenDataset, config: RunConfig, device, batches: int) -> dict:
    """Train-only statistics: the mean projected latent and per-baseline residual variance.

    The variance is taken around the fitted train mean for ``train_mean`` and
    around that baseline's own prediction for the others, so no baseline is
    charged for a reference point it does not use.
    """
    total, count = None, 0
    for _, batch in iter_windows(train_set, config, device, batches, seed=99):
        projected = model.project(batch["observed"]).reshape(-1, model.config.d_world)
        current = projected.sum(dim=0)
        total = current if total is None else total + current
        count += projected.shape[0]
    train_mean = (total / count).detach()
    residuals = {kind: {} for kind in BASELINES}
    for _, batch in iter_windows(train_set, config, device, batches, seed=98):
        observed = model.project(batch["observed"])
        for (offset, horizon), entry in batch["targets"].items():
            valid = entry["valid"]
            index = valid.nonzero().flatten()
            if index.numel() == 0:
                continue
            target = model.project(entry["tokens"])[index]
            for kind in BASELINES:
                mu = baseline_prediction(kind, observed, offset, horizon, train_mean)[index]
                squared = (target - mu).pow(2).sum(dim=0)
                record = residuals[kind].setdefault((offset, horizon), {"sum": torch.zeros_like(squared), "n": 0})
                record["sum"] += squared
                record["n"] += int(index.numel())
    return {
        "train_mean": train_mean,
        "variance": {
            kind: {key: (record["sum"] / record["n"]).clamp_min(1e-6) for key, record in entries.items()}
            for kind, entries in residuals.items()
        },
    }


@torch.no_grad()
def evaluate_baselines(model, dataset: TokenDataset, config: RunConfig, device, batches: int,
                       stats: dict) -> dict:
    totals = {kind: {} for kind in BASELINES}
    for _, batch in iter_windows(dataset, config, device, batches):
        observed = model.project(batch["observed"])
        for (offset, horizon), entry in batch["targets"].items():
            valid = entry["valid"]
            index = valid.nonzero().flatten()
            if index.numel() == 0:
                continue
            target = model.project(entry["tokens"])[index]
            for kind in BASELINES:
                variance = stats["variance"][kind].get((offset, horizon))
                if variance is None:
                    continue
                mu = baseline_prediction(kind, observed, offset, horizon, stats["train_mean"])[index]
                bits = gaussian_nll_bits(target, mu, variance.clamp_min(1e-6).log()).mean(dim=(1, 2))
                record = totals[kind].setdefault((offset, horizon),
                                                 {"nll": 0.0, "mse": 0.0, "n": 0, "delta": 0.0})
                record["nll"] += float(bits.sum())
                record["mse"] += float((target - mu).pow(2).mean(dim=(1, 2)).sum())
                record["n"] += int(index.numel())
                record["delta"] += float(entry["delta_seconds"][index].sum())
    return {
        kind: {
            f"offset_{offset}_horizon_{horizon}": {
                "nll_bits_per_dim": record["nll"] / record["n"],
                "mse": record["mse"] / record["n"],
                "n": record["n"],
                "delta_seconds": record["delta"] / record["n"],
            }
            for (offset, horizon), record in sorted(entries.items())
        }
        for kind, entries in totals.items()
    }


def compare(model_metrics: dict, baseline_metrics: dict, config: RunConfig) -> dict:
    """Per (anchor, horizon) comparison on identical samples, plus equal-weight means.

    Every method is averaged over exactly the same pairs, and both NLL and MSE are
    reported so a mean-squared-error improvement cannot be confused with a
    likelihood improvement. An untrained model can already beat these baselines on
    NLL, so this table is a capability check, not evidence of learning; learning is
    established by comparing the initial, best and final checkpoints (see
    ``evaluate_checkpoints``).
    """
    per_key = {}
    for key, entry in model_metrics["per_anchor_horizon"].items():
        offset, horizon = (int(part) for part in key.replace("offset_", "").split("_horizon_"))
        row = {
            "offset": offset,
            "horizon": horizon,
            "delta_seconds": entry["delta_seconds"],
            "n": entry["n"],
            "model_nll_bits_per_dim": entry["nll_bits_per_dim"],
            "model_mse": entry["mse"],
        }
        for kind in BASELINES:
            baseline = baseline_metrics.get(kind, {}).get(key)
            row[f"{kind}_nll_bits_per_dim"] = baseline["nll_bits_per_dim"] if baseline else None
            row[f"{kind}_mse"] = baseline["mse"] if baseline else None
        available = [row[f"{kind}_nll_bits_per_dim"] for kind in BASELINES
                     if row[f"{kind}_nll_bits_per_dim"] is not None]
        row["best_baseline_nll_bits_per_dim"] = min(available) if available else None
        row["model_minus_best_baseline_bits"] = (
            row["model_nll_bits_per_dim"] - row["best_baseline_nll_bits_per_dim"]
            if available else None
        )
        best_mse = min((row[f"{kind}_mse"] for kind in BASELINES
                        if row[f"{kind}_mse"] is not None), default=None)
        row["best_baseline_mse"] = best_mse
        row["model_minus_best_baseline_mse"] = (
            row["model_mse"] - best_mse if best_mse is not None else None
        )
        per_key[key] = row
    count = max(len(per_key), 1)
    model_wins = sum(1 for row in per_key.values() if (row["model_minus_best_baseline_bits"] or 0) < 0)
    model_mse_wins = sum(1 for row in per_key.values()
                         if (row["model_minus_best_baseline_mse"] or 0) < 0)
    return {
        "per_anchor_horizon": per_key,
        "summary": {
            "n_pairs": len(per_key),
            "weighting": "equal weight over the same (anchor, horizon) pairs for every method",
            "model_mean_nll_bits_per_dim": sum(r["model_nll_bits_per_dim"] for r in per_key.values()) / count,
            "model_mean_mse": sum(r["model_mse"] for r in per_key.values()) / count,
            "baseline_mean_nll_bits_per_dim": {
                kind: sum(r[f"{kind}_nll_bits_per_dim"] for r in per_key.values()
                          if r[f"{kind}_nll_bits_per_dim"] is not None)
                / max(1, sum(1 for r in per_key.values() if r[f"{kind}_nll_bits_per_dim"] is not None))
                for kind in BASELINES
            },
            "baseline_mean_mse": {
                kind: sum(r[f"{kind}_mse"] for r in per_key.values() if r[f"{kind}_mse"] is not None)
                / max(1, sum(1 for r in per_key.values() if r[f"{kind}_mse"] is not None))
                for kind in BASELINES
            },
            "model_beats_best_baseline_nll_on": f"{model_wins}/{len(per_key)} pairs",
            "model_beats_best_baseline_mse_on": f"{model_mse_wins}/{len(per_key)} pairs",
            "interpretation": (
                "baseline wins alone are not learning evidence: an untrained model already beats "
                "these baselines on this validation set; compare initial/best/final checkpoints"
            ),
        },
    }


def baseline_summary(baseline_metrics: dict) -> dict:
    """Equal-weight per-baseline means over the shared (anchor, horizon) pairs."""
    summary = {}
    for kind, entries in baseline_metrics.items():
        values = list(entries.values())
        if not values:
            continue
        summary[kind] = {
            "n_pairs": len(values),
            "mean_nll_bits_per_dim": sum(v["nll_bits_per_dim"] for v in values) / len(values),
            "mean_mse": sum(v["mse"] for v in values) / len(values),
            "mean_delta_seconds": sum(v["delta_seconds"] for v in values) / len(values),
            "weighting": "equal weight over the same (anchor, horizon) pairs as the model",
        }
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
