"""Export the frozen 3D branch's per-point class probabilities.

The alignment constraint needs a cross-modal target: for every pixel that has a
visible point, the 3D branch's class distribution on that point. This script
runs a trained PointNet checkpoint over nearest-neighbour blocks that cover each
prepared scene and averages the softmax over the blocks a point appears in.

Output: one ``<scene>.npz`` per prepared scene with ``probabilities [N, 3]``
float16 and ``counts [N]`` int32.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bridge_seg.classes import CLASS_NAMES  # noqa: E402
from bridge_seg.data import build_block_features, load_scene_npz  # noqa: E402
from bridge_seg.models import PointNetSegmentation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--points-per-block", type=int, default=1024)
    parser.add_argument("--blocks-per-scene", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PointNetSegmentation(
        input_channels=9,
        num_classes=len(CLASS_NAMES),
        use_cross_side=False,
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((Path(args.prepared_root) / args.split).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no prepared scenes under {args.prepared_root}/{args.split}")

    for path in paths:
        scene = load_scene_npz(path)
        point_count = len(scene.xyz)
        probabilities = np.zeros((point_count, len(CLASS_NAMES)), dtype=np.float64)
        counts = np.zeros(point_count, dtype=np.int64)
        rng = np.random.default_rng(args.seed + 1000)
        block_size = min(args.points_per_block, point_count)

        for _ in range(args.blocks_per_scene):
            seed_index = int(rng.integers(0, point_count))
            distances = np.sum((scene.xyz - scene.xyz[seed_index]) ** 2, axis=1)
            indices = np.argpartition(distances, block_size - 1)[:block_size]
            features = build_block_features(
                scene.xyz[indices],
                scene.rgb[indices],
                scene,
            )
            with torch.no_grad():
                logits = model(
                    torch.from_numpy(features).transpose(0, 1).unsqueeze(0).to(device)
                )
                softmax = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            probabilities[indices] += softmax.T
            counts[indices] += 1

        covered = counts > 0
        probabilities[covered] /= counts[covered, None]
        np.savez_compressed(
            output_dir / f"{path.stem}.npz",
            probabilities=probabilities.astype(np.float16),
            counts=counts.astype(np.int32),
        )
        print(
            f"{path.stem}: points={point_count} covered={int(covered.sum())} "
            f"({100.0 * covered.mean():.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
