"""Stage the public sample clips and write a byte-verified source manifest.

    python examples/prepare_samples.py --clips 20
    python examples/prepare_samples.py --clips 20 --check-only

The dataset revision is pinned. The script downloads the selected files through
``huggingface_hub`` (reusing the local cache when present), copies them into
``videos/`` keeping the original file stem so the source identity survives, and
then writes ``videos/sources.json``. A clip is only listed as public after the
staged bytes and the snapshot bytes hash identically; anything else stays a
``local_file``. No training is performed.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

# imports the installed package; run this after `pip install -e .` from the repo root
from wpm_video.data import build_source_manifest, list_videos, probe_video

DATASET_REPO = "nateraw/kinetics-mini"
DATASET_REVISION = "9f4ed38128a355c352527209101be3e326471816"
# only these class directories are considered: a cached snapshot may hold many
# other classes, and they must never be pulled in silently
PATTERNS = ("train/archery", "train/bowling")


def download_snapshot(repo: str, revision: str, patterns) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency hint
        raise SystemExit(
            "huggingface_hub is required for the sample helper: "
            f"pip install -e \".[vjepa]\" ({error})"
        )
    return Path(snapshot_download(repo, repo_type="dataset", revision=revision,
                                  allow_patterns=[f"{pattern.rstrip('/')}/*" for pattern in patterns]))


def stage(snapshot: Path, out_dir: Path, limit: int, patterns=PATTERNS) -> list:
    """Copy clips from the requested class directories only, keeping the original stem.

    Only ``patterns`` (``train/<class>`` directories, relative to the snapshot) are
    scanned, so other classes that happen to be cached are never staged by accident.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for pattern in patterns:
        directory = Path(snapshot) / pattern.rstrip("*").rstrip("/")
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a directory in the snapshot")
        selected.extend(sorted(directory.glob("*.mp4")))
    staged = []
    for clip in selected:
        if len(staged) >= limit:
            break
        # the class directory is the label; the file stem carries the source id
        destination = out_dir / f"{clip.parent.name}_{clip.stem}.mp4"
        if not destination.is_file():
            shutil.copy2(clip, destination)
        staged.append(destination)
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("videos"))
    parser.add_argument("--clips", type=int, default=20, help="how many clips to stage")
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="use an already downloaded snapshot directory instead of downloading")
    parser.add_argument("--check-only", action="store_true",
                        help="verify the staged clips against an existing snapshot, do not copy")
    args = parser.parse_args()

    snapshot = args.snapshot or download_snapshot(DATASET_REPO, args.revision, PATTERNS)
    if not (snapshot / "train").is_dir():
        raise SystemExit(f"{snapshot} does not look like a {DATASET_REPO} snapshot")
    if not args.check_only:
        staged = stage(snapshot, args.out, args.clips)
        print(f"staged {len(staged)} clips into {args.out} (revision {args.revision[:12]})")

    manifest = build_source_manifest(args.out, snapshot, args.revision, repo_id=DATASET_REPO)
    manifest_path = args.out / "sources.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    mapped = len(manifest["clips"])
    skipped = manifest.get("skipped", [])
    print(f"{mapped} clips verified byte-for-byte against the snapshot")
    if skipped:
        print(f"not declared public (kept as local_file): {skipped}")

    infos = [probe_video(path, with_hash=False) for path in list_videos(args.out)]
    classes = sorted({path.stem.partition("_")[0] for path in list_videos(args.out)})
    print(f"staged collection: {len(infos)} clips, classes {classes}")
    unexpected = [name for name in classes if name not in {p.split("/")[-1] for p in PATTERNS}]
    if unexpected:
        print(f"note: classes outside the requested patterns: {unexpected}")
    # Run the rest from your own working directory (see examples/prepare_sample_dataset.md);
    # the package directory itself stays clean.
    print("\nnext steps, run from your work directory with the interpreter that has the "
          "package installed:")
    print("  python -m wpm_video cache --config config.json")
    print("  python -m wpm_video train --config config.json --out runs/run1 --val-count 4")
    print("  python -m wpm_video eval  --config config.json \\")
    print("      --checkpoint runs/run1/best.pt --out runs/eval")
    print("  # --val-count N holds out whole clips; the split actually used is written")
    print("  # to runs/run1/splits.json, e.g. its val[0] is the clip to predict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
