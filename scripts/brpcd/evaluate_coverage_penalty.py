"""Diagnose the occlusion-evaluation survivorship bias.

Runs an existing checkpoint on the test split at 0/25/50/75 % missing points
and reports two metrics per rate:

* ``surviving`` - IoU over the points that survived occlusion (the metric the
  training-time evaluation reports today);
* ``coverage``  - IoU over every point of the reference block, with removed
  points counted as errors.

The deterministic block sampling of ``PreparedSceneBlockDataset`` is replayed
(same seed, same epoch) so the surviving-only numbers can be checked against the
recorded run metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bridge_seg.classes import CLASS_NAMES  # noqa: E402
from bridge_seg.data import discover_prepared_splits  # noqa: E402
from bridge_seg.dataset import PreparedSceneBlockDataset  # noqa: E402
from bridge_seg.metrics import segmentation_metrics  # noqa: E402
from bridge_seg.models import PointNetSegmentation  # noqa: E402
from bridge_seg.occlusion import occlude_points  # noqa: E402

MISSING = -1


def class_iou_with_missing(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    missing: int = MISSING,
) -> np.ndarray:
    """IoU where a ``missing`` prediction counts as a false negative."""

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    ious = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        in_true = y_true == cls
        in_pred = y_pred == cls
        true_positive = int(np.count_nonzero(in_true & in_pred))
        false_positive = int(np.count_nonzero(in_pred & ~in_true))
        false_negative = int(np.count_nonzero(in_true & ~in_pred))
        ious[cls] = true_positive / max(
            true_positive + false_positive + false_negative,
            1,
        )
    return ious


def metrics_with_missing(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    ious = class_iou_with_missing(y_true, y_pred, num_classes)
    correct = int(np.count_nonzero((y_true == y_pred) & (y_pred != MISSING)))
    return {
        "mean_iou": float(ious.mean()),
        "per_class_iou": ious.tolist(),
        "accuracy": correct / max(len(y_true), 1),
        "points_scored": int(len(y_true)),
        "predictions_missing": int(np.count_nonzero(y_pred == MISSING)),
    }


def evaluate_rate(
    model: torch.nn.Module,
    device: torch.device,
    test_paths: list[Path],
    points_per_block: int,
    blocks_per_scene: int,
    seed: int,
    rate: float,
    strategy: str,
) -> dict[str, object]:
    full = PreparedSceneBlockDataset(
        test_paths,
        points_per_block=points_per_block,
        blocks_per_scene=blocks_per_scene,
        seed=seed + 2000,
        sampling_mode="uniform",
        occlusion_rate=0.0,
    )
    occluded = PreparedSceneBlockDataset(
        test_paths,
        points_per_block=points_per_block,
        blocks_per_scene=blocks_per_scene,
        seed=seed + 2000,
        sampling_mode="uniform",
        occlusion_rate=rate,
        occlusion_strategy=strategy,
    )

    surviving_true: list[np.ndarray] = []
    surviving_pred: list[np.ndarray] = []
    coverage_true: list[np.ndarray] = []
    coverage_pred: list[np.ndarray] = []

    for index in range(len(occluded)):
        features_full, labels_full = full[index]
        features_occluded, labels_occluded = occluded[index]
        xyz = features_full[:3].T.numpy()
        reference = labels_full.numpy()
        kept = occlude_points(
            xyz,
            xyz,
            reference,
            missing_rate=rate,
            strategy=strategy,
            seed=seed + 2000 + index,
        ).kept_indices
        if not np.array_equal(labels_occluded.numpy(), reference[kept]):
            raise RuntimeError("replayed occlusion does not match the dataset")

        with torch.no_grad():
            logits = model(features_occluded.unsqueeze(0).to(device))
            predicted = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        full_prediction = np.full(len(reference), MISSING, dtype=np.int64)
        full_prediction[kept] = predicted

        surviving_true.append(labels_occluded.numpy())
        surviving_pred.append(predicted)
        coverage_true.append(reference)
        coverage_pred.append(full_prediction)

    surviving_true_array = np.concatenate(surviving_true)
    surviving_pred_array = np.concatenate(surviving_pred)
    coverage_true_array = np.concatenate(coverage_true)
    coverage_pred_array = np.concatenate(coverage_pred)

    surviving = segmentation_metrics(
        surviving_true_array,
        surviving_pred_array,
        len(CLASS_NAMES),
    )
    surviving.pop("confusion_matrix", None)
    coverage = metrics_with_missing(
        coverage_true_array,
        coverage_pred_array,
        len(CLASS_NAMES),
    )
    return {
        "requested_rate": rate,
        "kept_fraction": float(
            np.count_nonzero(coverage_pred_array != MISSING)
            / len(coverage_pred_array)
        ),
        "surviving": surviving,
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points-per-block", type=int, default=1024)
    parser.add_argument("--blocks-per-scene", type=int, default=32)
    parser.add_argument("--rates", default="0,0.25,0.5,0.75")
    parser.add_argument("--strategy", default="viewpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    splits = discover_prepared_splits(Path(args.prepared_root))
    model = PointNetSegmentation(
        input_channels=9,
        num_classes=len(CLASS_NAMES),
        use_cross_side=False,
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    rates = [float(item) for item in str(args.rates).split(",") if item.strip()]
    results: dict[str, object] = {
        "prepared_root": args.prepared_root,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "points_per_block": args.points_per_block,
        "blocks_per_scene": args.blocks_per_scene,
        "strategy": args.strategy,
        "test_scenes": [path.name for path in splits["test"]],
        "rates": {},
    }
    for rate in rates:
        record = evaluate_rate(
            model,
            device,
            splits["test"],
            args.points_per_block,
            args.blocks_per_scene,
            args.seed,
            rate,
            args.strategy,
        )
        results["rates"][f"{rate:.2f}"] = record
        print(
            f"rate={rate:.2f} "
            f"surviving={record['surviving']['mean_iou']:.4f} "
            f"coverage={record['coverage']['mean_iou']:.4f} "
            f"kept={record['kept_fraction']:.3f}",
            flush=True,
        )

    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
