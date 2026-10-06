"""Train-only PCA visualisation and demo figures.

The PCA basis is fitted on training latents only, so a validation or demo point
projected into it is not fitted on itself. The figures show latent geometry and
prediction error; they are not generated pixels and are labelled as such.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


@torch.no_grad()
def fit_pca(latents: torch.Tensor, components: int = 2) -> dict:
    """Fit PCA on (N, D) training latents; returns the mean and the projection matrix."""
    flat = latents.reshape(-1, latents.shape[-1]).float()
    mean = flat.mean(dim=0)
    centered = flat - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return {"mean": mean, "components": vh[:components].transpose(0, 1).contiguous(),
            "n_fit": int(flat.shape[0])}


@torch.no_grad()
def project(pca: dict, latents: torch.Tensor) -> torch.Tensor:
    flat = latents.reshape(-1, latents.shape[-1]).float()
    return (flat - pca["mean"]) @ pca["components"]


def plot_future_pca(pca: dict, predictions: dict, path: Path, title: str) -> None:
    """Spatial spread of present, predicted and (if available) true future latents."""
    figure, axes = plt.subplots(1, len(predictions), figsize=(4.2 * len(predictions), 4.2), squeeze=False)
    for axis, (horizon, entry) in zip(axes[0], sorted(predictions.items())):
        predicted = project(pca, entry["mu"])
        axis.scatter(predicted[:, 0], predicted[:, 1], s=14, alpha=0.7, label="predicted mu")
        if "target" in entry:
            target = project(pca, entry["target"])
            axis.scatter(target[:, 0], target[:, 1], s=14, alpha=0.5, marker="x", label="true future")
            for index in range(0, predicted.shape[0], 16):
                axis.plot([predicted[index, 0], target[index, 0]], [predicted[index, 1], target[index, 1]],
                          color="grey", linewidth=0.4, alpha=0.5)
        axis.set_title(f"horizon {horizon} ({entry['delta_seconds']:.1f}s)")
        axis.set_xlabel("PC1 (fit on train)")
        axis.set_ylabel("PC2 (fit on train)")
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(f"{title}\nlatent PCA of frozen-encoder targets; not generated RGB")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_uncertainty(predictions: dict, path: Path) -> None:
    """Predicted sigma against realised error, per horizon and per spatial patch."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    horizons, sigmas, errors = [], [], []
    for horizon, entry in sorted(predictions.items()):
        horizons.append(f"h{horizon}\n{entry['delta_seconds']:.0f}s")
        sigma = entry["sigma"].mean().item()
        sigmas.append(sigma)
        errors.append(entry.get("mse", float("nan")) ** 0.5)
    axes[0].bar(horizons, sigmas, color="#4477aa")
    axes[0].set_title("mean predicted sigma (latent units)")
    axes[0].set_ylabel("sigma")
    axes[1].bar(horizons, errors, color="#cc6677")
    axes[1].set_title("realised RMSE (latent units)")
    axes[1].set_ylabel("RMSE")
    figure.suptitle("uncertainty vs realised error per horizon")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_frames(frames: torch.Tensor, prefix_end_frame: int, future_frame: int | None, path: Path,
                times: list[float]) -> None:
    """Observed prefix frames and the true future frame used as the prediction target."""
    count = min(6, frames.shape[0])
    indices = torch.linspace(0, max(prefix_end_frame - 1, 0), count).long().tolist()
    columns = count + (1 if future_frame is not None else 0)
    figure, axes = plt.subplots(1, columns, figsize=(2.1 * columns, 2.6), squeeze=False)
    for axis, index in zip(axes[0], indices):
        axis.imshow(frames[index].permute(1, 2, 0).numpy())
        axis.set_title(f"t={times[index]:.1f}s", fontsize=8)
        axis.axis("off")
    if future_frame is not None:
        axis = axes[0][-1]
        axis.imshow(frames[future_frame].permute(1, 2, 0).numpy())
        axis.set_title("true future frame\n(ground truth reference)", fontsize=8)
        axis.axis("off")
    figure.suptitle("observed prefix (left) and ground-truth target frame (right); no RGB is generated")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
