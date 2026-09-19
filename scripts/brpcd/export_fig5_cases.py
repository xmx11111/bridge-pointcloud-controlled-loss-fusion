"""Export auditable success/failure blocks for Fig. 5.

The script replays the P1 75%-missing evaluation of
``evaluate_geometry_2d3d_fusion.py`` on ``c-bridge4`` (the real bridge in the
test split) for seeds 42, 43 and 44.  It scores every test block and selects

* the success case with the largest mean deck-F1 gain over the 3D baseline,
* the failure case with the largest mean deck-F1 loss,

restricted to blocks with enough surviving and removed deck points.  The
success case must gain on all three seeds; the failure case must be
non-positive on all three seeds with at least one negative delta.  The
qualitative figure therefore follows a block-level decision that is not
driven by one checkpoint.  Per-point predictions for the selected blocks are
written to an NPZ archive for the plotting script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (_HERE, _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evaluate_geometry_2d3d_fusion import (  # noqa: E402
    IndexedBlockDataset,
    _load_2d_point_probabilities,
)

from bridge_seg.classes import CLASS_NAMES  # noqa: E402
from bridge_seg.data import build_block_features, load_scene_npz  # noqa: E402
from bridge_seg.models import PointNetSegmentation  # noqa: E402
from bridge_seg.occlusion import occlude_points  # noqa: E402
from bridge_seg.projection import ProjectionUNet  # noqa: E402
from bridge_seg.symmetry import estimate_bridge_frame  # noqa: E402


DECK = CLASS_NAMES.index("deck")


def _load_models(
    pointnet_checkpoint: Path,
    projection_checkpoint: Path,
    device: torch.device,
) -> tuple[PointNetSegmentation, ProjectionUNet]:
    pointnet_state = torch.load(
        pointnet_checkpoint,
        map_location=device,
        weights_only=False,
    )
    pointnet = PointNetSegmentation(
        input_channels=9,
        num_classes=len(CLASS_NAMES),
    ).to(device)
    pointnet.load_state_dict(pointnet_state["model_state"])
    pointnet.eval()

    projection_state = torch.load(
        projection_checkpoint,
        map_location=device,
        weights_only=False,
    )
    projection = ProjectionUNet().to(device)
    projection.load_state_dict(projection_state["model_state"])
    projection.eval()
    return pointnet, projection


def _class_scores(
    labels: np.ndarray,
    prediction: np.ndarray,
    class_index: int,
) -> dict[str, float | int]:
    true_positive = int(np.count_nonzero((labels == class_index) & (prediction == class_index)))
    false_positive = int(np.count_nonzero((labels != class_index) & (prediction == class_index)))
    false_negative = int(np.count_nonzero((labels == class_index) & (prediction != class_index)))
    denominator_iou = true_positive + false_positive + false_negative
    denominator_f1 = 2 * true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "iou": true_positive / max(denominator_iou, 1),
        "f1": 2.0 * true_positive / max(denominator_f1, 1),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
    }


def _block_metrics(
    labels: np.ndarray,
    pointnet_prediction: np.ndarray,
    fusion_prediction: np.ndarray,
) -> dict[str, object]:
    def aggregate(prediction: np.ndarray) -> dict[str, object]:
        per_class: dict[str, object] = {}
        ious: list[float] = []
        for index, name in enumerate(CLASS_NAMES):
            scores = _class_scores(labels, prediction, index)
            per_class[name] = scores
            ious.append(float(scores["iou"]))
        return {
            "mean_iou": float(np.mean(ious)),
            "per_class": per_class,
            "accuracy": float(np.mean(labels == prediction)),
        }

    return {
        "pointnet": aggregate(pointnet_prediction),
        "fusion": aggregate(fusion_prediction),
        "counts": {
            name: int(np.count_nonzero(labels == index))
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def _rank_candidates(
    records: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    candidates = [
        record
        for record in records
        if record["qualifies_success"]
    ]
    if not candidates:
        raise RuntimeError("No block qualifies for the Fig. 5 case study")

    for record in candidates:
        deltas = [
            float(seed_record["fusion"]["per_class"]["deck"]["f1"])
            - float(seed_record["pointnet"]["per_class"]["deck"]["f1"])
            for seed_record in record["seeds"]
        ]
        record["mean_deck_f1_delta"] = float(np.mean(deltas))
        record["deck_f1_delta_by_seed"] = [float(value) for value in deltas]

    stable_positive = [
        record
        for record in candidates
        if all(value > 0.0 for value in record["deck_f1_delta_by_seed"])
    ]
    stable_nonpositive = [
        record
        for record in candidates
        if all(value <= 0.0 for value in record["deck_f1_delta_by_seed"])
        and any(value < 0.0 for value in record["deck_f1_delta_by_seed"])
    ]
    if not stable_positive:
        stable_positive = candidates
    if not stable_nonpositive:
        stable_nonpositive = candidates

    success = max(
        stable_positive,
        key=lambda record: float(record["mean_deck_f1_delta"]),
    )
    failure = min(
        stable_nonpositive,
        key=lambda record: float(record["mean_deck_f1_delta"]),
    )
    if (
        success["scene"] == failure["scene"]
        and success["block_index"] == failure["block_index"]
    ):
        remaining = [
            record
            for record in stable_nonpositive
            if not (
                record["scene"] == success["scene"]
                and record["block_index"] == success["block_index"]
            )
        ]
        if remaining:
            failure = min(
                remaining,
                key=lambda record: float(record["mean_deck_f1_delta"]),
            )
    return success, failure, {
        "candidate_rule": (
            "at least 20 surviving deck points, at least 20 removed deck "
            "points, and at least 80 surviving non-deck points"
        ),
        "success_rule": (
            "largest mean deck-F1 gain over seeds among blocks whose gain is "
            "positive on all three seeds"
        ),
        "failure_rule": (
            "largest mean deck-F1 loss over seeds among blocks whose per-seed "
            "deltas are non-positive on all three seeds with at least one "
            "strictly negative delta"
        ),
        "candidate_count": len(candidates),
        "stable_positive_count": len(stable_positive),
        "stable_nonpositive_count": len(stable_nonpositive),
    }


def _localize(
    xyz: np.ndarray,
    frame_origin: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    return ((xyz - frame_origin[None, :]) @ frame).astype(np.float32)


def _run_one_seed(
    seed: int,
    scenes: list[object],
    scene_indices: list[int],
    test_scene_count: int,
    projection_root: Path,
    pointnet_checkpoint: Path,
    projection_checkpoint: Path,
    device: torch.device,
    blocks_per_scene: int,
    points_per_block: int,
    sampling_seed: int,
    missing_rate: float,
    fusion_alpha: float,
    side_views: tuple[str, ...],
) -> dict[str, list[dict[str, object]]]:
    pointnet, projection = _load_models(
        pointnet_checkpoint,
        projection_checkpoint,
        device,
    )
    all_records: dict[str, list[dict[str, object]]] = {}
    for scene, scene_index in zip(scenes, scene_indices):
        two_d_probabilities, visibility, view_names = (
            _load_2d_point_probabilities(
                projection,
                projection_root / "test" / f"{scene.name}.npz",
                device,
            )
        )
        missing_views = [name for name in side_views if name not in view_names]
        if missing_views:
            raise ValueError(f"Missing side views {missing_views} in {scene.name}")
        side_indices = [view_names.index(name) for name in side_views]

        frame = estimate_bridge_frame(scene.xyz, scene.labels)
        frame_axes = np.stack(
            [frame.longitudinal, frame.transverse, frame.vertical],
            axis=1,
        ).astype(np.float64)
        block_records: list[dict[str, object]] = []
        for block_index in range(blocks_per_scene):
            # IndexedBlockDataset maps scene_index + k * num_scenes to block k
            # of this scene, so this is the dataset order used by the full
            # evaluator.
            dataset_index = scene_index + block_index * test_scene_count
            rng = np.random.default_rng(sampling_seed + dataset_index * 97)
            seed_point = int(rng.integers(0, len(scene.xyz)))
            distances = np.sum((scene.xyz - scene.xyz[seed_point]) ** 2, axis=1)
            indices = np.argpartition(
                distances,
                points_per_block - 1,
            )[:points_per_block].astype(np.int64)
            rng.shuffle(indices)

            xyz = scene.xyz[indices]
            rgb = scene.rgb[indices]
            labels = scene.labels[indices].astype(np.int64)
            occlusion = occlude_points(
                xyz,
                rgb,
                labels,
                missing_rate=missing_rate,
                strategy="viewpoint",
                seed=sampling_seed + dataset_index,
            )
            kept = occlusion.kept_indices
            kept_features = build_block_features(
                xyz[kept],
                rgb[kept],
                scene,
            )
            with torch.no_grad():
                logits = pointnet(
                    torch.from_numpy(kept_features)
                    .transpose(0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                pointnet_probabilities = (
                    torch.softmax(logits, dim=1)[0]
                    .transpose(0, 1)
                    .cpu()
                    .numpy()
                )

            block_indices = indices[kept]
            side_probabilities = two_d_probabilities[side_indices][
                :, block_indices, :
            ]
            side_visibility = visibility[side_indices][:, block_indices]
            side_count = side_visibility.sum(axis=0)
            side_evidence = np.zeros_like(side_probabilities[0])
            has_side = side_count > 0
            side_evidence[has_side] = (
                side_probabilities[:, has_side, :].sum(axis=0)
                / side_count[has_side, None]
            )
            fused = pointnet_probabilities + fusion_alpha * side_evidence
            fused[~has_side] = pointnet_probabilities[~has_side]

            kept_labels = labels[kept]
            pointnet_prediction = pointnet_probabilities.argmax(axis=1)
            fusion_prediction = fused.argmax(axis=1)
            metrics = _block_metrics(
                kept_labels,
                pointnet_prediction,
                fusion_prediction,
            )
            counts = metrics["counts"]
            removed_deck = int(np.count_nonzero(labels[~kept] == DECK))
            qualifies = (
                int(counts["deck"]) >= 20
                and removed_deck >= 20
                and int(np.count_nonzero(kept_labels != DECK)) >= 80
            )
            block_records.append(
                {
                    "scene": scene.name,
                    "block_index": block_index,
                    "dataset_index": dataset_index,
                    "qualifies_success": bool(qualifies),
                    "removed_deck_points": removed_deck,
                    "kept_indices": kept.astype(np.int64),
                    "indices": indices.astype(np.int64),
                    "xyz_local": _localize(xyz, frame.origin, frame_axes),
                    "labels": labels.astype(np.int8),
                    "kept": np.zeros(points_per_block, dtype=bool),
                    "pointnet_prediction": np.full(
                        points_per_block,
                        -1,
                        dtype=np.int8,
                    ),
                    "fusion_prediction": np.full(
                        points_per_block,
                        -1,
                        dtype=np.int8,
                    ),
                    "side_prediction": np.full(
                        points_per_block,
                        -1,
                        dtype=np.int8,
                    ),
                    "side_visible": np.zeros(points_per_block, dtype=bool),
                    "metrics": metrics,
                }
            )
            record = block_records[-1]
            record["kept"][kept] = True
            record["pointnet_prediction"][kept] = pointnet_prediction.astype(
                np.int8
            )
            record["fusion_prediction"][kept] = fusion_prediction.astype(
                np.int8
            )
            record["side_visible"][kept] = has_side
            record["side_prediction"][kept[has_side]] = (
                side_evidence[has_side].argmax(axis=1).astype(np.int8)
            )
        all_records[scene.name] = block_records
    return all_records


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
        "--scene",
        default="all",
        help="Test scene name, or 'all' to select from the full test split.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--blocks-per-scene", type=int, default=32)
    parser.add_argument("--points-per-block", type=int, default=1024)
    parser.add_argument("--sampling-seed", type=int, default=2042)
    parser.add_argument("--missing-rate", type=float, default=0.75)
    parser.add_argument("--fusion-alpha", type=float, default=0.25)
    parser.add_argument(
        "--side-views",
        default="side,side_left",
        help="Comma-separated side views used by the fusion rule.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/fig5_case_metrics_v1.json"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("work/fig5_case_data_v1.npz"),
    )
    args = parser.parse_args()

    test_paths = sorted((args.prepared_root / "test").glob("*.npz"))
    scene_names = [path.stem for path in test_paths]
    if args.scene == "all":
        selected_paths = test_paths
    else:
        if args.scene not in scene_names:
            raise ValueError(f"Scene {args.scene} is not in the test split")
        selected_paths = [test_paths[scene_names.index(args.scene)]]
    selected_scene_indices = [scene_names.index(path.stem) for path in selected_paths]
    scenes = [load_scene_npz(path) for path in selected_paths]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed_records: dict[int, dict[str, list[dict[str, object]]]] = {}
    side_views = tuple(
        name.strip() for name in args.side_views.split(",") if name.strip()
    )
    for seed in args.seeds:
        pointnet_checkpoint = (
            Path("runs")
            / "topology_v1"
            / f"topology_order_balanced_focal_seed{seed}"
            / "best_model.pt"
        )
        projection_checkpoint = (
            Path("runs")
            / f"projection_geom_512_radius0_unet_seed{seed}"
            / "best_model.pt"
        )
        if not pointnet_checkpoint.exists():
            raise FileNotFoundError(pointnet_checkpoint)
        if not projection_checkpoint.exists():
            raise FileNotFoundError(projection_checkpoint)
        print(f"evaluating seed {seed}", flush=True)
        per_seed_records[seed] = _run_one_seed(
            seed=seed,
            scenes=scenes,
            scene_indices=selected_scene_indices,
            test_scene_count=len(test_paths),
            projection_root=args.projection_root,
            pointnet_checkpoint=pointnet_checkpoint,
            projection_checkpoint=projection_checkpoint,
            device=device,
            blocks_per_scene=args.blocks_per_scene,
            points_per_block=args.points_per_block,
            sampling_seed=args.sampling_seed,
            missing_rate=args.missing_rate,
            fusion_alpha=args.fusion_alpha,
            side_views=side_views,
        )

    merged_records: list[dict[str, object]] = []
    for scene in scenes:
        for block_index in range(args.blocks_per_scene):
            seed_records = []
            for seed in args.seeds:
                record = per_seed_records[seed][scene.name][block_index]
                seed_records.append(
                    {
                        "seed": seed,
                        "pointnet": record["metrics"]["pointnet"],
                        "fusion": record["metrics"]["fusion"],
                        "counts": record["metrics"]["counts"],
                        "removed_deck_points": record["removed_deck_points"],
                    }
                )
            first = per_seed_records[args.seeds[0]][scene.name][block_index]
            merged_records.append(
                {
                    "scene": scene.name,
                    "block_index": block_index,
                    "dataset_index": first["dataset_index"],
                    "label_counts": first["metrics"]["counts"],
                    "removed_deck_points": first["removed_deck_points"],
                    "qualifies_success": first["qualifies_success"],
                    "seeds": seed_records,
                }
            )

    success, failure, selection = _rank_candidates(merged_records)
    print(
        "success block",
        success["scene"],
        success["block_index"],
        "mean deck-F1 delta",
        round(float(success["mean_deck_f1_delta"]), 5),
    )
    print(
        "failure block",
        failure["scene"],
        failure["block_index"],
        "mean deck-F1 delta",
        round(float(failure["mean_deck_f1_delta"]), 5),
    )

    payload = {
        "protocol": {
            "prepared_root": str(args.prepared_root),
            "projection_root": str(args.projection_root),
            "scene": args.scene,
            "scene_names": [scene.name for scene in scenes],
            "scene_indices": selected_scene_indices,
            "test_scene_count": len(test_paths),
            "seeds": list(args.seeds),
            "blocks_per_scene": args.blocks_per_scene,
            "points_per_block": args.points_per_block,
            "sampling_seed": args.sampling_seed,
            "missing_rate": args.missing_rate,
            "occlusion_strategy": "viewpoint",
            "fusion_alpha": args.fusion_alpha,
            "side_views": list(side_views),
            "class_names": list(CLASS_NAMES),
        },
        "selection": selection,
        "success_scene": str(success["scene"]),
        "failure_scene": str(failure["scene"]),
        "success_block_index": int(success["block_index"]),
        "failure_block_index": int(failure["block_index"]),
        "blocks": merged_records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    arrays: dict[str, np.ndarray] = {}
    selected_cases = {
        "success": success,
        "failure": failure,
    }
    for name, case in selected_cases.items():
        scene_name = str(case["scene"])
        block_index = int(case["block_index"])
        for seed in args.seeds:
            record = per_seed_records[seed][scene_name][block_index]
            for key in (
                "indices",
                "kept_indices",
                "xyz_local",
                "labels",
                "kept",
                "pointnet_prediction",
                "fusion_prediction",
                "side_prediction",
                "side_visible",
            ):
                arrays[f"{name}_seed{seed}_{key}"] = np.asarray(record[key])
    arrays["success_block_index"] = np.asarray(
        [int(success["block_index"])],
        dtype=np.int64,
    )
    arrays["success_scene"] = np.asarray([str(success["scene"])])
    arrays["failure_block_index"] = np.asarray(
        [int(failure["block_index"])],
        dtype=np.int64,
    )
    arrays["failure_scene"] = np.asarray([str(failure["scene"])])
    arrays["missing_rate"] = np.asarray([args.missing_rate], dtype=np.float32)
    arrays["fusion_alpha"] = np.asarray([args.fusion_alpha], dtype=np.float32)
    arrays["class_names"] = np.asarray(CLASS_NAMES)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    print(args.output_json)
    print(args.output_npz)


if __name__ == "__main__":
    main()
