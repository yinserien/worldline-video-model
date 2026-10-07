r"""Example: a custom RGB decoder component, registered and trained end to end.

The decoder is a replaceable component. This example implements a second
architecture -- per-cell MLP + pixel shuffle, **no convolutions at all** -- registers
it under its own kind, and runs the shared pipeline (train -> save -> load ->
render) with it on CPU, offline.

    # offline smoke run: synthesises three tiny clips, trains a tiny native world
    # model and then the custom decoder, and renders one keyframe. --out must be a
    # working directory outside the repository, e.g. E:\work\wpm_run\custom_demo or
    # ~/wpm_run/custom_demo
    python examples/custom_decoder.py --out E:\work\wpm_run\custom_demo

    # against an existing world checkpoint (still offline, uses local videos)
    python examples/custom_decoder.py --out E:\work\wpm_run\custom_demo \
        --config E:\work\wpm_run\config.json --checkpoint E:\work\wpm_run\runs\world\best.pt

    # drive the stock CLI in-process, after this script registered the component
    python examples/custom_decoder.py --out E:\work\wpm_run\custom_demo --via-cli

    # same data, same splits, same projection, same output size as the built-in
    # conv decoder: print both held-out metric blocks side by side. The output size
    # must be one the conv architecture accepts (a power-of-two multiple of the grid
    # side), so 64 rather than the 48 default.
    python examples/custom_decoder.py --out E:\work\wpm_run\custom_demo \
        --image-size 64 --compare-conv

``--out`` is required and is the only place this script writes (plus the system
temporary directory for the offline fixtures); it refuses a path inside this source
checkout or the installed package. The numbers it prints are a smoke test of the
component API on synthetic clips, not evidence that one architecture reconstructs
better than another: that is a separate, measured experiment.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn

import wpm_video
from wpm_video.config import (DataConfig, DecoderConfig, DecoderTrainConfig, EncoderConfig,
                              ModelConfig, RunConfig, TrainConfig)
from wpm_video.dataset import TokenDataset, cache_video_tokens
from wpm_video.decoder import RGBDecoder, load_decoder, register_decoder, run_decoder_training
from wpm_video.model import VideoWorldModel
from wpm_video.train import build_model, fit_projection, train

# The name identifies the implementation *including its semantics*, so it carries a
# version: if this architecture ever changes what it computes, it must be registered
# under a new name ("pixelshuffle_v2") instead of redefining what "pixelshuffle_v1"
# means, otherwise a checkpoint of v1 would silently load into a different network.
KIND = "pixelshuffle_v1"
OPTIONS = ("hidden",)                        # every other option name is a typo
CLIP_FRAMES, CLIP_SIZE, CLIP_FPS = 16, 64, 8.0


class PixelShuffleDecoder(RGBDecoder):
    """Learned per-cell projection followed by a pixel shuffle.

    Every latent cell is mapped independently to a ``3 * r * r`` block (``r =
    image_size / grid_side``) and the blocks are rearranged into the output image, so
    the architecture has no spatial convolution and no upsampling stage. It does need
    ``r`` to be a whole number -- its own constraint, checked here, not part of the
    decoder contract: an architecture that resizes differently need not have it.

    Options: ``{"hidden": int}`` (width of the per-cell MLP, default 64). Unknown
    option names are refused: a typo must not silently fall back to a default.
    """

    def __init__(self, config: DecoderConfig, patches: int, d_world: int):
        super().__init__(config, patches, d_world)
        unknown = sorted(set(config.options) - set(OPTIONS))
        if unknown:
            raise ValueError(
                f"the {KIND!r} decoder takes only options={{'hidden': int}}, got unknown "
                f"{unknown}; a mistyped option is refused rather than ignored"
            )
        hidden = config.options.get("hidden", 64)
        if type(hidden) is not int or not 4 <= hidden <= 4096:
            raise ValueError(
                f"the {KIND!r} decoder takes options={{'hidden': int}} with 4 <= hidden <= 4096, "
                f"got {hidden!r}"
            )
        if self.output_size % self.grid_side:
            raise ValueError(
                f"the {KIND!r} decoder maps every latent cell to a {self.grid_side}x"
                f"{self.grid_side} block, so decoder.image_size ({self.output_size}) must be a "
                f"whole multiple of the patch grid side ({self.grid_side}); the conv "
                "architecture additionally requires that factor to be a power of two"
            )
        rows, cols = self.grid
        self.scale = self.output_size // self.grid_side
        self.position = nn.Parameter(torch.randn(1, d_world, rows, cols) * 0.02)
        self.to_hidden = nn.Linear(d_world, hidden)
        self.activation = nn.SiLU()
        self.to_pixels = nn.Linear(hidden, 3 * self.scale * self.scale)
        self.shuffle = nn.PixelShuffle(self.scale)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.check_input(latents)
        rows, cols = self.grid
        cells = latents.transpose(1, 2).reshape(latents.shape[0], self.d_world, rows, cols)
        cells = (cells + self.position).permute(0, 2, 3, 1)          # (B, gh, gw, d_world)
        blocks = self.to_pixels(self.activation(self.to_hidden(cells)))
        return torch.sigmoid(self.shuffle(blocks.permute(0, 3, 1, 2)))


def build_pixel_shuffle_decoder(config: DecoderConfig, patches: int, d_world: int):
    """Factory registered under :data:`KIND`; returns the component for this world."""
    return PixelShuffleDecoder(config, patches, d_world)


# Registration is process-local: importing this module makes this kind available to
# build_decoder/load_decoder/run_decoder_training in this process.
register_decoder(KIND, build_pixel_shuffle_decoder)


def source_checkout_root() -> "Path | None":
    """The repository root when this example runs from a source checkout."""
    package = Path(wpm_video.__file__).resolve().parent          # .../src/wpm_video
    if package.parent.name == "src" and (package.parent.parent / "pyproject.toml").is_file():
        return package.parent.parent                            # the checkout root
    return None


def reject_repository_output(out_dir: Path) -> None:
    """Refuse an output directory inside the package tree or the source checkout.

    ``<checkout>/src/wpm_video`` alone would miss ``<checkout>/runs``: the whole
    checkout is read-only working material, and a run belongs next to the data.
    """
    package_dir = Path(wpm_video.__file__).resolve().parent
    forbidden = [("the package directory", package_dir)]
    checkout = source_checkout_root()
    if checkout is not None:
        forbidden.append(("this source checkout", checkout))
    for what, directory in forbidden:
        if out_dir == directory or directory in out_dir.parents:
            raise SystemExit(f"--out must be outside {what} ({directory}), got {out_dir}")


# -- offline fixtures ---------------------------------------------------------
def write_clip(path: Path, seed: int) -> Path:
    """A tiny moving-shape clip, written with OpenCV so no download is involved."""
    import cv2

    generator = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS,
                             (CLIP_SIZE, CLIP_SIZE))
    for index in range(CLIP_FRAMES):
        frame = np.zeros((CLIP_SIZE, CLIP_SIZE, 3), np.uint8)
        offset = int((CLIP_SIZE - 20) * (index / max(1, CLIP_FRAMES - 1)))
        frame[10:30, offset:offset + 20] = (200, 40, 40)
        frame[40:60, 40 - offset // 3:60 - offset // 3] = (40, 40, 200)
        frame = np.clip(frame.astype(np.int16) + generator.integers(0, 12, frame.shape), 0, 255)
        writer.write(frame.astype(np.uint8))
    writer.release()
    return path


def demo_config(image_size: int, hidden: int, steps: int, seed: int = 0) -> RunConfig:
    """A tiny native-encoder configuration carrying the custom decoder component."""
    return RunConfig(
        name="custom_decoder_demo",
        data=DataConfig(fps=CLIP_FPS, chunk_frames=4, chunk_stride_frames=4, image_size=CLIP_SIZE,
                        context_chunks=2, window_stride_chunks=1, horizon_chunks=[1],
                        max_horizon_chunks=2),
        encoder=EncoderConfig(kind="native", device="cpu", batch_clips=2),
        model=ModelConfig(d_world=32, d_hidden=64, slots=16, heads=4, substep_seconds=0.125,
                          max_substeps=64, anchor_bandwidth=1.0),
        train=TrainConfig(seed=seed, batch_windows=2, max_steps=6, eval_interval=3,
                          eval_batches=1, device="cpu", log_interval=1, max_wall_seconds=600.0),
        decoder=DecoderConfig(kind=KIND, image_size=image_size, options={"hidden": hidden}),
        decoder_train=DecoderTrainConfig(seed=seed, batch_windows=2, learning_rate=2e-3,
                                         max_steps=steps, eval_interval=steps, eval_batches=1,
                                         log_interval=1, max_wall_seconds=600.0,
                                         frame_cache_videos=2),
    )


def prepare_offline_world(config: RunConfig, work_dir: Path, device) -> Path:
    """Synthesise clips, train a tiny native world model and return its checkpoint."""
    from wpm_video.encoder import NativeEncoder

    work_dir = Path(work_dir)
    video_dir, cache_dir = work_dir / "videos", work_dir / "cache"
    names = [write_clip(video_dir / f"clip{index}.mp4", index).stem for index in range(3)]
    config.data.video_dir = str(video_dir)
    config.data.cache_dir = str(cache_dir)
    config.data.train_videos = names[:2]
    config.data.val_videos = names[2:]
    config.validate()
    encoder = NativeEncoder(d_model=48, image_size=CLIP_SIZE, patch=16, tubelet=2, seed=5)
    for name in names:
        cache_video_tokens(video_dir / f"{name}.mp4", config.data, encoder, cache_dir,
                           batch_clips=2)
    train_set = TokenDataset(config.data, cache_dir, config.encoder, "train")
    val_set = TokenDataset(config.data, cache_dir, config.encoder, "val")
    torch.manual_seed(0)
    world = build_model(config, encoder, device)
    fit_projection(world, train_set)
    train(config, world, train_set, val_set, work_dir / "world", device,
          provenance={"splits": {"train": names[:2], "val": names[2:]}})
    return work_dir / "world" / "final.pt"


# -- the shared pipeline, with a custom component -----------------------------
def train_custom_decoder(config: RunConfig, world_checkpoint, out_dir: Path, device,
                         via_cli: bool = False, config_path: Path | None = None) -> dict:
    """Train the registered custom decoder through the pipeline or through the CLI.

    Both paths do the same thing; ``via_cli`` shows the documented way to use the
    stock command line with a component that lives outside the package: register it
    in this process, then call ``wpm_video.cli.main`` from here.
    """
    out_dir = Path(out_dir)
    if not via_cli:
        return run_decoder_training(config, world_checkpoint, out_dir, device)
    from wpm_video.cli import main as cli_main

    if config_path is None:
        raise ValueError("--via-cli needs a config.json on disk (the CLI reads a file)")
    code = cli_main(["train-decoder", "--config", str(config_path),
                     "--checkpoint", str(world_checkpoint), "--out", str(out_dir)])
    if code != 0:
        raise SystemExit(f"wpm-video train-decoder exited with {code}")
    return json.loads((out_dir / "train_summary.json").read_text(encoding="utf-8"))


def reference_latents(config: RunConfig, world_checkpoint, device, index: int = 0) -> torch.Tensor:
    """Project one cached chunk into the latent space the decoder was trained on."""
    split = "val" if config.data.val_videos else "train"
    dataset = TokenDataset(config.data, Path(config.data.cache_dir), config.encoder, split)
    window = dataset.windows[min(index, len(dataset.windows) - 1)]
    rows = dataset.tokens(window.video, window.start)
    world, _ = VideoWorldModel.from_checkpoint(world_checkpoint, map_location="cpu")
    with torch.no_grad():
        return world.to(device).project(rows.unsqueeze(0).to(device).float())[0].cpu()


def render_one(decoder_checkpoint, latents: torch.Tensor, out_dir: Path, device) -> Path:
    """Load a decoder checkpoint and write one decoded keyframe PNG."""
    from wpm_video.decoder import save_frame_png

    decoder, _ = load_decoder(decoder_checkpoint, map_location="cpu")
    decoder = decoder.to(device).eval()
    with torch.no_grad():
        image = decoder(latents.unsqueeze(0).to(device).float())[0]
    return save_frame_png(image, Path(out_dir) / "decoded_keyframe.png")


def report(label: str, summary: dict) -> None:
    """Print the held-out metrics of one decoder run, with their units."""
    metrics = summary.get("final_val") or {}
    reference = metrics.get("constant_frame_reference") or {}
    print(f"\n[{label}] kind={summary['decoder']['kind']} "
          f"parameters={summary['decoder']['parameters']:,} "
          f"output={summary['decoder']['image_size']}px")
    print(f"  held-out l1={metrics.get('l1'):.4f} mse={metrics.get('mse'):.5f} "
          f"psnr={metrics.get('psnr_db'):.2f} dB  (constant-frame reference: "
          f"l1={reference.get('l1'):.4f}, psnr={reference.get('psnr_db'):.2f} dB)")
    print(f"  videos: {', '.join(metrics.get('videos', []))}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path,
                        help="run directory (required); the only place this script writes")
    parser.add_argument("--config", type=Path, default=None,
                        help="existing configuration (with --checkpoint); default: build a demo")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="existing world checkpoint to train the decoder against")
    parser.add_argument("--image-size", type=int, default=48,
                        help="decoder output edge; 48 is not a power-of-two multiple of the "
                             "4x4 grid, which is fine for this architecture and refused by conv")
    parser.add_argument("--hidden", type=int, default=64, help="per-cell MLP width")
    parser.add_argument("--steps", type=int, default=60, help="decoder training steps")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--via-cli", action="store_true",
                        help="run the stock wpm-video train-decoder command in this process")
    parser.add_argument("--compare-conv", action="store_true",
                        help="also train the built-in conv decoder on identical data, splits, "
                             "projection and output size, and print both metric blocks")
    args = parser.parse_args(argv)
    if (args.config is None) != (args.checkpoint is None):
        parser.error("--config and --checkpoint go together")
    device = torch.device(args.device)

    out_dir = Path(args.out).resolve()
    reject_repository_output(out_dir)         # before anything is written anywhere

    temporary = None
    if args.checkpoint is not None:
        config = RunConfig.load(args.config)
        config.decoder = DecoderConfig(kind=KIND, image_size=args.image_size,
                                       options={"hidden": args.hidden})
        config.decoder_train.max_steps = args.steps
        config.decoder_train.eval_interval = args.steps
        config.validate()
        world_checkpoint = Path(args.checkpoint)
    else:
        # offline demo: everything is synthesised under the system temporary directory
        temporary = tempfile.TemporaryDirectory(prefix="wpm_custom_decoder_")
        config = demo_config(args.image_size, args.hidden, args.steps)
        world_checkpoint = prepare_offline_world(config, Path(temporary.name), device)

    out_dir.mkdir(parents=True, exist_ok=True)
    # the device applies to every path, including --via-cli, which resolves it from
    # the configuration file rather than from this process
    config.train.device = args.device
    config_path = out_dir / "config.json"   # never over the caller's own config file
    config.save(config_path)
    print(f"registered decoder kinds: {list(wpm_video.available_decoders())}")
    print(f"training kind={KIND!r} against {world_checkpoint} -> {out_dir}")
    summary = train_custom_decoder(config, world_checkpoint, out_dir, device,
                                   via_cli=args.via_cli, config_path=config_path)
    report(KIND, summary)
    if args.compare_conv:
        # Identical data, splits, projection and output size: only the architecture
        # differs, which is what a component comparison has to hold fixed.
        factor = args.image_size // summary["decoder"]["grid"][0]
        if args.image_size % summary["decoder"]["grid"][0] or factor & (factor - 1):
            # a skip is not a result: say so and fail, rather than print one metric
            # block where a side-by-side comparison was asked for
            grid_side = summary["decoder"]["grid"][0]
            factor = 1
            while grid_side * factor < max(args.image_size, 32):   # round up to a valid size
                factor *= 2
            print(f"\n--compare-conv: NO comparison was run. The conv architecture needs "
                  f"image_size / grid_side to be a power of two, and this run used "
                  f"image_size={args.image_size} on a {grid_side}x{grid_side} grid. Rerun with "
                  f"a power-of-two multiple of the grid side (for example --image-size "
                  f"{grid_side * factor}) to compare both architectures at the same output "
                  "size.")
            if temporary is not None:
                temporary.cleanup()
            return 1
        else:
            conv_config = RunConfig.load(config_path)
            conv_config.decoder = DecoderConfig(kind="conv", image_size=args.image_size,
                                                base_channels=16, channel_multipliers=[1, 2],
                                                stem_blocks=1, blocks_per_stage=1)
            conv_config.validate()
            conv_summary = run_decoder_training(conv_config, world_checkpoint, out_dir / "conv",
                                                device)
            report("conv (built-in)", conv_summary)
            print("\nBoth numbers come from the same clips, splits, projection and output size; "
                  "they are a\nprotocol demonstration on synthetic data, not a claim that either "
                  "architecture reconstructs better.")

    (out_dir / "custom_decoder_summary.json").write_text(
        json.dumps({"component": KIND, "options": config.decoder.options, "summary": summary},
                   indent=2, default=float) + "\n", encoding="utf-8")
    splits_path = out_dir / "splits.json"
    if splits_path.is_file():          # written by the CLI path
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        config.data.train_videos = config.data.train_videos or list(splits.get("train") or [])
        config.data.val_videos = config.data.val_videos or list(splits.get("val") or [])
    if Path(config.data.cache_dir).is_dir() and (config.data.train_videos or config.data.val_videos):
        latents = reference_latents(config, world_checkpoint, device)
        print(f"rendered {render_one(out_dir / 'final.pt', latents, out_dir, device)}")
    else:
        print("no token cache next to this configuration: skipping the render step")
    print(f"\nwrote {out_dir}; see docs/decoder_components.md for the component contract")
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
