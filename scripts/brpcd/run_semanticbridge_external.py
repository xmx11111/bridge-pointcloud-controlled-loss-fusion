"""Run the SemanticBridge external validation: 2D training + fusion sweep.

Mirrors the BrPCD protocol exactly (same scripts, same flags) so that the
only difference between the two datasets is the data itself.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\肖\Documents\Codex\2026-09-16\ni")
PY = r"D:\Codex\envs\pointcloud\Scripts\python.exe"

SB_PREPARED = Path(r"D:\Codex\datasets\SemanticBridge\prepared_v1")
SB_PROJECTION = Path(r"D:\Codex\datasets\SemanticBridge\projection_geom_512_radius0")
PRIOR_DIR = REPO / "work" / "brpcd" / "sb_prior_3d_seed42"
LOG_DIR = REPO / "work" / "sb_external"

SEEDS = (42, 43, 44)
VIEW_CONFIGS = {
    "side": "side",
    "side_left": "side,side_left",
}
RATES = (0.0, 0.25, 0.5, 0.75)
ALIGN_WEIGHT = 0.3


def run(name: str, command: list[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    start = time.time()
    print(f"[start] {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    status = "ok" if process.returncode == 0 else f"FAILED({process.returncode})"
    print(f"[{status}] {name} in {elapsed / 60:.1f} min", flush=True)
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        print(tail, flush=True)
        raise SystemExit(1)


def train_2d(seed: int) -> None:
    run(
        f"train_2d_seed{seed}",
        [
            PY,
            "work/brpcd/train_projection_unet_geometry.py",
            "--projection-root",
            str(SB_PROJECTION),
            "--output-dir",
            str(REPO / "runs" / f"semanticbridge_projection_unet_seed{seed}"),
            "--epochs",
            "80",
            "--batch-size",
            "4",
            "--seed",
            str(seed),
        ],
    )


def export_prior(split: str) -> None:
    run(
        f"prior_{split}",
        [
            PY,
            "work/brpcd/extract_3d_prior.py",
            "--prepared-root",
            str(SB_PREPARED),
            "--checkpoint",
            str(REPO / "runs" / "semanticbridge_baseline_seed42" / "best_model.pt"),
            "--output-dir",
            str(PRIOR_DIR),
            "--split",
            split,
            "--seed",
            "42",
        ],
    )


def train_aligned(seed: int) -> None:
    run(
        f"train_aligned_seed{seed}",
        [
            PY,
            "work/brpcd/train_projection_unet_aligned.py",
            "--projection-root",
            str(SB_PROJECTION),
            "--prior-dir",
            str(PRIOR_DIR),
            "--output-dir",
            str(REPO / "runs" / f"semanticbridge_projection_aligned_unet_seed{seed}"),
            "--epochs",
            "80",
            "--batch-size",
            "4",
            "--align-weight",
            str(ALIGN_WEIGHT),
            "--seed",
            str(seed),
        ],
    )


def fusion_runs(method: str, seed: int) -> None:
    if method == "vanilla":
        projection_checkpoint = (
            REPO / "runs" / f"semanticbridge_projection_unet_seed{seed}" / "best_model.pt"
        )
    else:
        projection_checkpoint = (
            REPO
            / "runs"
            / f"semanticbridge_projection_aligned_unet_seed{seed}"
            / "best_model.pt"
        )
    for view_name, side_views in VIEW_CONFIGS.items():
        for rate in RATES:
            out = (
                LOG_DIR
                / "fusion"
                / f"sb_{method}_{view_name}_seed{seed}_r{rate:.2f}.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            run(
                f"fusion_{method}_{view_name}_seed{seed}_r{rate:.2f}",
                [
                    PY,
                    "work/brpcd/evaluate_geometry_2d3d_fusion.py",
                    "--prepared-root",
                    str(SB_PREPARED),
                    "--projection-root",
                    str(SB_PROJECTION),
                    "--pointnet-checkpoint",
                    str(
                        REPO
                        / "runs"
                        / f"semanticbridge_baseline_seed{seed}"
                        / "best_model.pt"
                    ),
                    "--projection-checkpoint",
                    str(projection_checkpoint),
                    "--side-view-names",
                    side_views,
                    "--missing-rate",
                    f"{rate:.2f}",
                    "--output",
                    str(out),
                ],
            )


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "train2d"):
        for seed in SEEDS:
            train_2d(seed)
    if only in ("all", "prior"):
        for split in ("train", "val", "test"):
            export_prior(split)
    if only in ("all", "aligned"):
        for seed in SEEDS:
            train_aligned(seed)
    if only in ("all", "fusion"):
        for method in ("vanilla", "aligned"):
            for seed in SEEDS:
                fusion_runs(method, seed)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
