"""Export paired per-block metrics for the manuscript's main fusion claims.

This script replays the P1 evaluation protocol (1024 points per block,
32 blocks per test scene, 11 test scenes) for the original and force-aligned
2D branches.  It writes one record per seed, variant, missing rate and block
so that paired block-level bootstrap confidence intervals can be computed
without changing the existing aggregate result files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (_HERE, _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evaluate_geometry_2d3d_fusion import (  # noqa: E402
    IndexedBlockDataset,
    _aggregate_metrics,
    _load_2d_point_probabilities,
)

from bridge_seg.classes import CLASS_NAMES  # noqa: E402
from bridge_seg.data import load_scene_npz  # noqa: E402
from bridge_seg.models import PointNetSegmentation  # noqa: E402
from bridge_seg.occlusion import occlude_points  # noqa: E402
from bridge_seg.projection import ProjectionUNet  # noqa: E402


RATES = (0.0, 0.25, 0.5, 0.75)
SIDE_VIEWS = ("side", "side_left")


def _load_pointnet(checkpoint: Path, device: torch.device) -> PointNetSegmentation:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PointNetSegmentation(
        input_channels=9,
        num_classes=len(CLASS_NAMES),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _load_projection(
    checkpoint: Path,
    device: torch.device,
) -> ProjectionUNet:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ProjectionUNet().to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _side_probabilities(
    scenes: list[object],
    projection_root: Path,
    projection: ProjectionUNet,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    probabilities: dict[str, np.ndarray] = {}
    visibility: dict[str, np.ndarray] = {}
    view_names: list[str] | None = None
    for scene in scenes:
        scene_probabilities, scene_visibility, current_names = (
            _load_2d_point_probabilities(
                projection,
                projection_root / "test" / f"{scene.name}.npz",
                device,
            )
        )
        if view_names is None:
            view_names = current_names
        elif view_names != current_names:
            raise ValueError("Projection view order changed between scenes")
        missing = [name for name in SIDE_VIEWS if name not in current_names]
        if missing:
            raise ValueError(f"Missing side views {missing} in {scene.name}")
        side_indices = [current_names.index(name) for name in SIDE_VIEWS]
        probabilities[scene.name] = scene_probabilities[side_indices]
        visibility[scene.name] = scene_visibility[side_indices]
    if view_names is None:
        raise ValueError("No test scenes were loaded")
    return probabilities, visibility, view_names


def _checkpoint_for(
    variant: str,
    seed: int,
) -> tuple[Path, Path]:
    pointnet = (
        Path("runs")
        / "topology_v1"
        / f"topology_order_balanced_focal_seed{seed}"
        / "best_model.pt"
    )
    if variant == "original":
        # The archived main-table fusion runs pair every 3D seed with the
        # seed-42 original projection checkpoint.
        projection = (
            Path("runs")
            / "projection_geom_512_radius0_unet_seed42"
            / "best_model.pt"
        )
    elif variant == "aligned":
        projection = (
            Path("runs")
            / f"projection_geom_512_aligned_unet_seed{seed}"
            / "best_model.pt"
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    for path in (pointnet, projection):
        if not path.exists():
            raise FileNotFoundError(path)
    return pointnet, projection


def _export_variant(
    variant: str,
    seed: int,
    scenes: list[object],
    projection_root: Path,
    pointnet_checkpoint: Path,
    projection_checkpoint: Path,
    device: torch.device,
    sampling_seed: int,
    blocks_per_scene: int,
    points_per_block: int,
    batch_size: int,
    fusion_alpha: float,
) -> list[dict[str, object]]:
    pointnet = _load_pointnet(pointnet_checkpoint, device)
    projection = _load_projection(projection_checkpoint, device)
    point_2d, side_visible, _ = _side_probabilities(
        scenes,
        projection_root,
        projection,
        device,
    )

    dataset = IndexedBlockDataset(
        scenes,
        blocks_per_scene=blocks_per_scene,
        points_per_block=points_per_block,
        seed=sampling_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    records: list[dict[str, object]] = []
    dataset_index = 0
    with torch.no_grad():
        for features, labels, indices, scene_indices in loader:
            for batch_index in range(len(features)):
                scene = scenes[int(scene_indices[batch_index])]
                block_labels = labels[batch_index].numpy()
                xyz = (
                    features[batch_index]
                    .transpose(0, 1)
                    .numpy()[:, :3]
                )
                block_indices = indices[batch_index].numpy()
                side_probabilities = point_2d[scene.name][:, block_indices, :]
                side_visibility = side_visible[
                    scene.name
                ][:, block_indices]
                side_count = side_visibility.sum(axis=0)
                side_evidence = np.zeros_like(side_probabilities[0])
                has_side = side_count > 0
                side_evidence[has_side] = (
                    side_probabilities[:, has_side, :].sum(axis=0)
                    / side_count[has_side, None]
                )

                for rate in RATES:
                    occlusion = occlude_points(
                        xyz,
                        xyz,
                        block_labels,
                        missing_rate=rate,
                        strategy="viewpoint",
                        seed=sampling_seed + dataset_index,
                    )
                    kept = occlusion.kept_indices
                    block_features = features[
                        batch_index : batch_index + 1
                    ][:, :, kept]
                    logits = pointnet(block_features.to(device))
                    pointnet_probabilities = (
                        torch.softmax(logits, dim=1)[0]
                        .transpose(0, 1)
                        .cpu()
                        .numpy()
                    )
                    kept_labels = block_labels[kept]
                    kept_side = side_evidence[kept]
                    kept_side_visible = has_side[kept]
                    fused = (
                        pointnet_probabilities
                        + fusion_alpha * kept_side
                    )
                    fused[~kept_side_visible] = pointnet_probabilities[
                        ~kept_side_visible
                    ]
                    records.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "rate": rate,
                            "scene": scene.name,
                            "block_index": dataset_index % len(scenes),
                            "dataset_index": dataset_index,
                            "points": int(len(kept)),
                            "pointnet": _aggregate_metrics(
                                kept_labels,
                                pointnet_probabilities,
                            ),
                            "fusion": _aggregate_metrics(
                                kept_labels,
                                fused,
                            ),
                        }
                    )
                dataset_index += 1
            print(
                f"{variant} seed {seed}: {dataset_index}/"
                f"{len(dataset)} blocks",
                flush=True,
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(r"D:\Codex\datasets\BrPCD\official\prepared_20k_v2"),
    )
    parser.add_argument(
        "--projection-root",
        type=Path,
        default=Path(
            r"D:\Codex\datasets\BrPCD\official\projection_geom_512_v6_radius0"
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("original", "aligned"),
        default=("original", "aligned"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--blocks-per-scene", type=int, default=32)
    parser.add_argument("--points-per-block", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=2042)
    parser.add_argument("--fusion-alpha", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/block_statistics_v1.json"),
    )
    args = parser.parse_args()

    test_paths = sorted((args.prepared_root / "test").glob("*.npz"))
    scenes = [load_scene_npz(path) for path in test_paths]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, object]] = []
    for variant in args.variants:
        for seed in args.seeds:
            pointnet_checkpoint, projection_checkpoint = _checkpoint_for(
                variant,
                seed,
            )
            print(
                f"evaluating {variant} seed {seed} on {device}",
                flush=True,
            )
            records.extend(
                _export_variant(
                    variant=variant,
                    seed=seed,
                    scenes=scenes,
                    projection_root=args.projection_root,
                    pointnet_checkpoint=pointnet_checkpoint,
                    projection_checkpoint=projection_checkpoint,
                    device=device,
                    sampling_seed=args.sampling_seed,
                    blocks_per_scene=args.blocks_per_scene,
                    points_per_block=args.points_per_block,
                    batch_size=args.batch_size,
                    fusion_alpha=args.fusion_alpha,
                )
            )
    output = {
        "protocol": {
            "prepared_root": str(args.prepared_root),
            "projection_root": str(args.projection_root),
            "variants": list(args.variants),
            "seeds": args.seeds,
            "blocks_per_scene": args.blocks_per_scene,
            "points_per_block": args.points_per_block,
            "sampling_seed": args.sampling_seed,
            "fusion_alpha": args.fusion_alpha,
            "side_views": list(SIDE_VIEWS),
            "rates": list(RATES),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {args.output} ({len(records)} records)")


if __name__ == "__main__":
    main()
