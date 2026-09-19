from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bridge_seg.classes import CLASS_NAMES
from bridge_seg.data import build_block_features, load_scene_npz
from bridge_seg.metrics import segmentation_metrics
from bridge_seg.models import PointNetSegmentation
from bridge_seg.occlusion import OCCLUSION_STRATEGIES, occlude_points
from bridge_seg.projection import ProjectionUNet


class IndexedBlockDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, np.ndarray, int]]
):
    def __init__(
        self,
        scenes: list[object],
        blocks_per_scene: int,
        points_per_block: int,
        seed: int,
    ) -> None:
        self.scenes = scenes
        self.blocks_per_scene = blocks_per_scene
        self.points_per_block = points_per_block
        self.seed = seed

    def __len__(self) -> int:
        return len(self.scenes) * self.blocks_per_scene

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
        scene_index = index % len(self.scenes)
        scene = self.scenes[scene_index]
        rng = np.random.default_rng(self.seed + index * 97)
        seed_index = int(rng.integers(0, len(scene.xyz)))
        distances = np.sum((scene.xyz - scene.xyz[seed_index]) ** 2, axis=1)
        indices = np.argpartition(
            distances,
            self.points_per_block - 1,
        )[: self.points_per_block].astype(np.int64)
        rng.shuffle(indices)
        xyz = scene.xyz[indices]
        rgb = scene.rgb[indices]
        labels = scene.labels[indices]
        features = build_block_features(xyz, rgb, scene)
        return (
            torch.from_numpy(features).transpose(0, 1).contiguous(),
            torch.from_numpy(labels),
            indices,
            scene_index,
        )


def _load_2d_point_probabilities(
    model: ProjectionUNet,
    projection_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(projection_path, allow_pickle=False) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        view_indices = np.asarray(data["view_indices"], dtype=np.int64)
        point_depths = np.asarray(data["point_depths"], dtype=np.float32)
        depth_maps = np.asarray(data["depth_maps"], dtype=np.float32)

    image_tensor = torch.from_numpy(images.astype(np.float32) / 255.0).to(
        device
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)
    probabilities = probabilities.cpu().numpy()

    with np.load(projection_path, allow_pickle=False) as data:
        view_names = [str(name) for name in data["view_names"].tolist()]
    view_count, point_count, _ = view_indices.shape
    point_probabilities = np.zeros(
        (view_count, point_count, len(CLASS_NAMES)),
        dtype=np.float64,
    )
    visible = np.zeros((view_count, point_count), dtype=bool)
    for view_index in range(view_count):
        coordinates = view_indices[view_index]
        vertical = np.clip(coordinates[:, 0], 0, depth_maps.shape[1] - 1)
        horizontal = np.clip(coordinates[:, 1], 0, depth_maps.shape[2] - 1)
        depth_at_pixel = depth_maps[view_index, vertical, horizontal]
        finite = np.isfinite(depth_at_pixel) & np.isfinite(
            point_depths[view_index]
        )
        visible[view_index] = finite & np.isclose(
            point_depths[view_index],
            depth_at_pixel,
            rtol=0.0,
            atol=1e-3,
        )
        point_probabilities[view_index] = probabilities[
            view_index,
            :,
            vertical,
            horizontal,
        ]

    return point_probabilities, visible, view_names


def _aggregate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    if mask is not None:
        labels = labels[mask]
        probabilities = probabilities[mask]
    prediction = probabilities.argmax(axis=1)
    result = segmentation_metrics(
        labels.reshape(-1),
        prediction.reshape(-1),
        len(CLASS_NAMES),
    )
    return {
        **result,
        "points": int(labels.size),
    }


def _coverage_metrics(
    full_labels: np.ndarray,
    probabilities: np.ndarray,
    target_index: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    """IoU over every reference-block point, with removed points as errors.

    ``probabilities`` holds one row per surviving point, in the same order as
    ``target_index`` (its position inside the concatenated reference blocks).
    Points that occlusion removed carry no prediction and therefore count as
    false negatives for their true class, which is the coverage-penalised
    metric used in Section 3.6 of the manuscript.
    """

    prediction = np.full(len(full_labels), -1, dtype=np.int64)
    prediction[target_index] = probabilities.argmax(axis=1)
    per_class = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        in_true = full_labels == cls
        in_pred = prediction == cls
        true_positive = int(np.count_nonzero(in_true & in_pred))
        false_positive = int(np.count_nonzero(in_pred & ~in_true))
        false_negative = int(np.count_nonzero(in_true & ~in_pred))
        per_class[cls] = true_positive / max(
            true_positive + false_positive + false_negative,
            1,
        )
    correct = int(
        np.count_nonzero((full_labels == prediction) & (prediction != -1))
    )
    return {
        "mean_iou": float(per_class.mean()),
        "per_class_iou": per_class.tolist(),
        "accuracy": correct / max(len(full_labels), 1),
        "points": int(len(full_labels)),
        "points_scored": int(np.count_nonzero(prediction != -1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(r"D:\Codex\datasets\BrPCD\official\prepared_20k_v2"),
    )
    parser.add_argument(
        "--projection-root",
        type=Path,
        default=Path(
            r"D:\Codex\datasets\BrPCD\official\projection_geom_192_v2"
        ),
    )
    parser.add_argument(
        "--pointnet-checkpoint",
        type=Path,
        default=Path(r"runs\seg_tuning_topo05_seed42\best_model.pt"),
    )
    parser.add_argument(
        "--projection-checkpoint",
        type=Path,
        default=Path(r"runs\projection_geom_unet_seed42\best_model.pt"),
    )
    parser.add_argument("--blocks-per-scene", type=int, default=32)
    parser.add_argument("--points-per-block", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=2042)
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.0,
        help="Fraction of block points removed before evaluation.",
    )
    parser.add_argument(
        "--occlusion-strategy",
        choices=OCCLUSION_STRATEGIES,
        default="viewpoint",
        help=(
            "Occlusion strategy; viewpoint is the project's evaluation "
            "protocol and keeps the removed sets nested across rates."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/geometry_2d3d_fusion_results.json"),
    )
    parser.add_argument(
        "--side-view-names",
        default=None,
        help=(
            "Comma-separated projection views used as side evidence. "
            "Defaults to all views whose names start with 'side'."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_paths = sorted((args.prepared_root / "test").glob("*.npz"))
    scenes = [load_scene_npz(path) for path in test_paths]

    pointnet_checkpoint = torch.load(
        args.pointnet_checkpoint,
        map_location=device,
        weights_only=False,
    )
    pointnet = PointNetSegmentation(
        input_channels=9,
        num_classes=len(CLASS_NAMES),
    ).to(device)
    pointnet.load_state_dict(pointnet_checkpoint["model_state"])
    pointnet.eval()

    projection_checkpoint = torch.load(
        args.projection_checkpoint,
        map_location=device,
        weights_only=False,
    )
    projection = ProjectionUNet().to(device)
    projection.load_state_dict(projection_checkpoint["model_state"])
    projection.eval()

    point_2d_probabilities: dict[str, np.ndarray] = {}
    point_visibility: dict[str, np.ndarray] = {}
    view_names: list[str] | None = None
    for scene in scenes:
        probabilities, visibility, current_view_names = (
            _load_2d_point_probabilities(
            projection,
            args.projection_root / "test" / f"{scene.name}.npz",
            device,
            )
        )
        if view_names is None:
            view_names = current_view_names
        elif view_names != current_view_names:
            raise ValueError("Projection view order changed between scenes")
        point_2d_probabilities[scene.name] = probabilities
        point_visibility[scene.name] = visibility
    if view_names is None or "side" not in view_names:
        raise ValueError("The side projection is required for fusion")
    if args.side_view_names is None:
        side_indices = [
            index
            for index, name in enumerate(view_names)
            if name == "side" or name.startswith("side_")
        ]
    else:
        requested = [
            name.strip()
            for name in args.side_view_names.split(",")
            if name.strip()
        ]
        missing = [name for name in requested if name not in view_names]
        if missing:
            raise ValueError(f"Unknown side projection views: {missing}")
        side_indices = [view_names.index(name) for name in requested]

    dataset = IndexedBlockDataset(
        scenes,
        blocks_per_scene=args.blocks_per_scene,
        points_per_block=args.points_per_block,
        seed=args.sampling_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    label_chunks: list[np.ndarray] = []
    pointnet_chunks: list[np.ndarray] = []
    evidence_chunks: list[np.ndarray] = []
    visibility_chunks: list[np.ndarray] = []
    side_evidence_chunks: list[np.ndarray] = []
    side_visibility_chunks: list[np.ndarray] = []
    scene_chunks: list[np.ndarray] = []
    full_label_chunks: list[np.ndarray] = []
    coverage_target_chunks: list[np.ndarray] = []
    block_offset = 0
    block_counter = 0
    with torch.no_grad():
        for features, labels, indices, scene_indices in loader:
            label_np = labels.numpy()
            indices_np = indices.numpy()
            full_labels_batch = label_np.copy()
            block_size = full_labels_batch.shape[1]
            if args.missing_rate > 0.0:
                # Occlusion must be applied before the forward pass: dropping
                # points from an already-computed prediction would only score
                # the surviving points and inflate the metric.
                kept_features: list[torch.Tensor] = []
                kept_labels: list[np.ndarray] = []
                kept_indices: list[np.ndarray] = []
                for batch_index in range(len(features)):
                    xyz = (
                        features[batch_index].transpose(0, 1).numpy()[:, :3]
                    )
                    occlusion = occlude_points(
                        xyz,
                        xyz,
                        label_np[batch_index],
                        missing_rate=args.missing_rate,
                        strategy=args.occlusion_strategy,
                        # The seed does not depend on the rate, so the removed
                        # sets are nested across rates.
                        seed=args.sampling_seed + block_counter,
                    )
                    block_counter += 1
                    kept = occlusion.kept_indices
                    kept_features.append(features[batch_index][:, kept])
                    kept_labels.append(label_np[batch_index][kept])
                    kept_indices.append(indices_np[batch_index][kept])
                    coverage_target_chunks.append(
                        block_offset + batch_index * block_size + kept
                    )
                features = torch.stack(kept_features)
                label_np = np.stack(kept_labels)
                indices_np = np.stack(kept_indices)
            else:
                block_counter += len(features)
                for batch_index in range(len(features)):
                    coverage_target_chunks.append(
                        block_offset
                        + batch_index * block_size
                        + np.arange(block_size, dtype=np.int64)
                    )
            full_label_chunks.append(full_labels_batch.reshape(-1))
            block_offset += len(features) * block_size
            logits = pointnet(features.to(device))
            pointnet_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for batch_index in range(len(features)):
                batch_scene = dataset.scenes[int(scene_indices[batch_index])]
                block_labels = label_np[batch_index]
                block_indices = indices_np[batch_index]
                block_probabilities = pointnet_probabilities[batch_index].T
                probabilities = point_2d_probabilities[batch_scene.name][
                    :, block_indices, :
                ]
                visibility = point_visibility[batch_scene.name][
                    :, block_indices
                ]
                visibility_count = visibility.sum(axis=0)
                aggregated = np.zeros_like(probabilities[0])
                has_evidence = visibility_count > 0
                aggregated[has_evidence] = (
                    probabilities[:, has_evidence, :].sum(axis=0)
                    / visibility_count[has_evidence, None]
                )
                label_chunks.append(block_labels)
                pointnet_chunks.append(block_probabilities)
                evidence_chunks.append(aggregated)
                visibility_chunks.append(visibility_count)
                side_probabilities = probabilities[side_indices]
                side_visibility = visibility[side_indices]
                side_count = side_visibility.sum(axis=0)
                side_evidence = np.zeros_like(side_probabilities[0])
                has_side_evidence = side_count > 0
                side_evidence[has_side_evidence] = (
                    side_probabilities[:, has_side_evidence, :].sum(axis=0)
                    / side_count[has_side_evidence, None]
                )
                side_evidence_chunks.append(side_evidence)
                side_visibility_chunks.append(side_count > 0)
                scene_chunks.append(
                    np.full(len(block_labels), batch_scene.name)
                )

    labels = np.concatenate(label_chunks)
    pointnet_probabilities = np.concatenate(pointnet_chunks).astype(
        np.float64
    )
    evidence = np.concatenate(evidence_chunks)
    visibility = np.concatenate(visibility_chunks)
    side_evidence = np.concatenate(side_evidence_chunks)
    side_visibility = np.concatenate(side_visibility_chunks)
    scene_names = np.concatenate(scene_chunks)
    records: list[dict[str, object]] = []
    records.append(
        {
            "method": "pointnet",
            **(_aggregate_metrics(labels, pointnet_probabilities)),
        }
    )
    records.append(
        {
            "method": "geometry_2d_visible",
            **(
                _aggregate_metrics(
                    labels,
                    evidence,
                    visibility > 0,
                )
            ),
        }
    )
    records.append(
        {
            "method": "geometry_2d_side_visible",
            **(
                _aggregate_metrics(
                    labels,
                    side_evidence,
                    side_visibility > 0,
                )
            ),
        }
    )
    records.append(
        {
            "method": "geometry_2d_side_all_points",
            **(_aggregate_metrics(labels, side_evidence)),
        }
    )

    for alpha in (0.25, 0.5, 1.0, 2.0):
        fused = pointnet_probabilities + alpha * evidence
        fused[visibility == 0] = pointnet_probabilities[visibility == 0]
        records.append(
            {
                "method": f"fusion_alpha_{alpha:g}",
                **(_aggregate_metrics(labels, fused)),
            }
        )
        side_fused = pointnet_probabilities + alpha * side_evidence
        side_fused[side_visibility == 0] = pointnet_probabilities[
            side_visibility == 0
        ]
        records.append(
            {
                "method": f"fusion_side_alpha_{alpha:g}",
                **(_aggregate_metrics(labels, side_fused)),
            }
        )
        side_all_fused = pointnet_probabilities + alpha * side_evidence
        records.append(
            {
                "method": f"fusion_side_all_alpha_{alpha:g}",
                **(_aggregate_metrics(labels, side_all_fused)),
            }
        )

    full_labels = np.concatenate(full_label_chunks)
    coverage_target = np.concatenate(coverage_target_chunks)
    coverage_records: list[dict[str, object]] = [
        {
            "method": "pointnet",
            **(
                _coverage_metrics(
                    full_labels,
                    pointnet_probabilities,
                    coverage_target,
                    len(CLASS_NAMES),
                )
            ),
        }
    ]
    for alpha in (0.25, 0.5, 1.0, 2.0):
        fused = pointnet_probabilities + alpha * evidence
        fused[visibility == 0] = pointnet_probabilities[visibility == 0]
        coverage_records.append(
            {
                "method": f"fusion_alpha_{alpha:g}",
                **(
                    _coverage_metrics(
                        full_labels,
                        fused,
                        coverage_target,
                        len(CLASS_NAMES),
                    )
                ),
            }
        )
        side_fused = pointnet_probabilities + alpha * side_evidence
        side_fused[side_visibility == 0] = pointnet_probabilities[
            side_visibility == 0
        ]
        coverage_records.append(
            {
                "method": f"fusion_side_alpha_{alpha:g}",
                **(
                    _coverage_metrics(
                        full_labels,
                        side_fused,
                        coverage_target,
                        len(CLASS_NAMES),
                    )
                ),
            }
        )
        side_all_fused = pointnet_probabilities + alpha * side_evidence
        coverage_records.append(
            {
                "method": f"fusion_side_all_alpha_{alpha:g}",
                **(
                    _coverage_metrics(
                        full_labels,
                        side_all_fused,
                        coverage_target,
                        len(CLASS_NAMES),
                    )
                ),
            }
        )

    for scene_name in sorted(set(scene_names.tolist())):
        scene_mask = scene_names == scene_name
        scene_records = []
        for method in ("pointnet",):
            scene_records.append(
                {
                    "method": method,
                    **_aggregate_metrics(
                        labels[scene_mask],
                        pointnet_probabilities[scene_mask],
                    ),
                }
            )
        for alpha in (0.5, 1.0):
            fused = pointnet_probabilities[scene_mask] + alpha * evidence[
                scene_mask
            ]
            scene_visibility = visibility[scene_mask]
            fused[scene_visibility == 0] = pointnet_probabilities[scene_mask][
                scene_visibility == 0
            ]
            scene_records.append(
                {
                    "method": f"fusion_alpha_{alpha:g}",
                    **_aggregate_metrics(labels[scene_mask], fused),
                }
            )
            side_fused = pointnet_probabilities[
                scene_mask
            ] + alpha * side_evidence[scene_mask]
            scene_side_visibility = side_visibility[scene_mask]
            side_fused[scene_side_visibility == 0] = (
                pointnet_probabilities[scene_mask][
                    scene_side_visibility == 0
                ]
            )
            scene_records.append(
                {
                    "method": f"fusion_side_alpha_{alpha:g}",
                    **_aggregate_metrics(labels[scene_mask], side_fused),
                }
            )
            side_all_fused = pointnet_probabilities[
                scene_mask
            ] + alpha * side_evidence[scene_mask]
            scene_records.append(
                {
                    "method": f"fusion_side_all_alpha_{alpha:g}",
                    **_aggregate_metrics(
                        labels[scene_mask],
                        side_all_fused,
                    ),
                }
            )
        records.append(
            {
                "scene": scene_name,
                "visibility_coverage": float(
                    np.mean(visibility[scene_mask] > 0)
                ),
                "side_visibility_coverage": float(
                    np.mean(side_visibility[scene_mask] > 0)
                ),
                "methods": scene_records,
            }
        )

    visible_fraction = float(np.mean(visibility > 0))
    output = {
        "configuration": {
            "prepared_root": str(args.prepared_root),
            "projection_root": str(args.projection_root),
            "pointnet_checkpoint": str(args.pointnet_checkpoint),
            "projection_checkpoint": str(args.projection_checkpoint),
            "blocks_per_scene": args.blocks_per_scene,
            "points_per_block": args.points_per_block,
            "sampling_seed": args.sampling_seed,
            "missing_rate": float(args.missing_rate),
            "occlusion_strategy": args.occlusion_strategy,
            "evaluated_points": int(len(labels)),
            "visible_2d_fraction": visible_fraction,
            "visible_side_fraction": float(np.mean(side_visibility > 0)),
            "side_views": [view_names[index] for index in side_indices],
            "fusion_rule": (
                "argmax(P_pointnet + alpha * P_geometry_2d); "
                "invisible 2D points fall back to the 3D prediction"
            ),
        },
        "overall": records,
        "overall_coverage": coverage_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["configuration"], indent=2))
    for row in records[:6]:
        print(
            row["method"],
            "mIoU=",
            round(float(row["mean_iou"]), 5),
            "IoU=",
            [round(float(value), 5) for value in row["per_class_iou"]],
        )
    print("--- coverage-penalised (removed points count as errors) ---")
    for row in coverage_records[:4]:
        print(
            row["method"],
            "mIoU=",
            round(float(row["mean_iou"]), 5),
            "IoU=",
            [round(float(value), 5) for value in row["per_class_iou"]],
            f"scored={row['points_scored']}/{row['points']}",
        )
    print(args.output)


if __name__ == "__main__":
    main()
