from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "runs" / "pointnetpp_pilot_v1"
OUTPUT_JSON = ROOT / "work" / "pointnetpp_pilot_analysis.json"
OUTPUT_MD = ROOT / "refine-logs" / "POINTNETPP_PILOT_V1.md"
TTA_DIAGNOSTIC = ROOT / "refine-logs" / "topology_tta_diagnostic.json"

RATES = ("0.00", "0.25", "0.50", "0.75")
VARIANT_LABELS = {
    "baseline": "PointNet++ baseline",
    "topology": "PointNet++ topology order",
    "topology_mirror_tta": "PointNet++ topology + mirror TTA",
}


def _mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _aggregate_metrics(metrics: list[dict[str, object]]) -> dict[str, object]:
    by_rate: dict[str, object] = {}
    for rate in RATES:
        rate_metrics = [item["test_by_occlusion"][rate] for item in metrics]
        mean_iou, std_iou = _mean_std(
            [float(item["mean_iou"]) for item in rate_metrics]
        )
        per_class = []
        for class_index in range(3):
            class_mean, class_std = _mean_std(
                [
                    float(item["per_class_iou"][class_index])
                    for item in rate_metrics
                ]
            )
            per_class.append(
                {
                    "mean": class_mean,
                    "std": class_std,
                }
            )
        by_rate[rate] = {
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "per_class_iou": per_class,
            "points": int(rate_metrics[0]["points"]),
        }
    return by_rate


def _load_pilot() -> dict[str, object]:
    summary_path = PILOT_ROOT / "pilot_summary.csv"
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8-sig")))
    if len(rows) != 9:
        raise ValueError(f"Expected 9 pilot jobs, found {len(rows)}")

    variants: dict[str, object] = {}
    for variant in VARIANT_LABELS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if len(variant_rows) != 3:
            raise ValueError(
                f"Expected 3 seeds for {variant}, found {len(variant_rows)}"
            )
        metrics = [
            json.loads(
                (Path(row["output_dir"]) / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            for row in sorted(variant_rows, key=lambda item: int(item["seed"]))
        ]
        variants[variant] = {
            "seeds": [int(item["seed"]) for item in metrics],
            "aggregate": _aggregate_metrics(metrics),
            "per_seed": {
                str(item["seed"]): {
                    rate: {
                        "mean_iou": float(
                            item["test_by_occlusion"][rate]["mean_iou"]
                        ),
                        "per_class_iou": [
                            float(value)
                            for value in item["test_by_occlusion"][rate][
                                "per_class_iou"
                            ]
                        ],
                    }
                    for rate in RATES
                }
                for item in metrics
            },
        }
    return {
        "jobs": rows,
        "variants": variants,
    }


def _aggregate_run_dirs(paths: list[Path]) -> dict[str, object]:
    metrics = [
        json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for path in paths
    ]
    return _aggregate_metrics(metrics)


def _load_pointnet_comparators() -> dict[str, object]:
    baseline_paths = sorted(
        (ROOT / "runs" / "brpcd_formal_v1").glob(
            "clean_balanced_focal_seed*/metrics.json"
        )
    )
    topology_paths = sorted(
        (ROOT / "runs" / "topology_v1").glob(
            "topology_order_balanced_focal_seed*/metrics.json"
        )
    )
    if len(baseline_paths) != 3 or len(topology_paths) != 3:
        raise ValueError("Expected 3 PointNet baseline/topology runs")

    tta_payload = json.loads(TTA_DIAGNOSTIC.read_text(encoding="utf-8"))
    pointnet_topology_tta = tta_payload["summary"]["topology"]
    return {
        "pointnet_baseline": _aggregate_run_dirs(
            [path.parent for path in baseline_paths]
        ),
        "pointnet_topology": _aggregate_run_dirs(
            [path.parent for path in topology_paths]
        ),
        "pointnet_topology_mirror_tta": {
            rate: {
                "mean_iou": float(values["mIoU_mean"]),
                "std_iou": float(values["mIoU_std"]),
                "per_class_iou": [
                    {"mean": float(value), "std": None}
                    for value in values["per_class_mean"]
                ],
            }
            for rate, values in pointnet_topology_tta.items()
        },
    }


def _format_mean_std(value: dict[str, object]) -> str:
    std = value["std_iou"]
    if std is None:
        return f"{float(value['mean_iou']):.4f}"
    return f"{float(value['mean_iou']):.4f} +/- {float(std):.4f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}"


def _build_markdown(
    pilot: dict[str, object],
    comparators: dict[str, object],
) -> str:
    variants = pilot["variants"]
    lines = [
        "# PointNet++ Pilot V1",
        "",
        "Date: 2026-09-17",
        "",
        "## Protocol",
        "",
        "- Dataset: BrPCD `prepared_20k_v2`",
        "- Split: P1 random-family scene split, test family `c-bridge4`",
        "- Seeds: 42, 43, 44",
        "- Training: 20 epochs, 1024 points per block, batch 16,",
        "  class-balanced sampling, weighted focal loss",
        "- Backbone: pure-PyTorch PointNet++-style hierarchical encoder",
        "- Methods: baseline; `deck > girder > pier` topology order;",
        "  topology order with geometry-only transverse mirror TTA",
        "- TTA uses no test labels and reuses the same block and occlusion seeds",
        "",
        "## PointNet++ Three-Seed Results",
        "",
        "| Method | 0% mIoU | 25% mIoU | 50% mIoU | 75% mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, label in VARIANT_LABELS.items():
        aggregate = variants[variant]["aggregate"]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(_format_mean_std(aggregate[rate]) for rate in RATES)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Class IoU at 0% Missing",
            "",
            "| Method | girder | pier | deck |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant, label in VARIANT_LABELS.items():
        per_class = variants[variant]["aggregate"]["0.00"]["per_class_iou"]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{float(item['mean']):.4f} +/- {float(item['std']):.4f}"
                for item in per_class
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Backbone Comparison",
            "",
            "| Backbone / method | 0% mIoU | 25% mIoU | 50% mIoU | 75% mIoU |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    comparison_labels = {
        "pointnet_baseline": "PointNet baseline",
        "pointnet_topology": "PointNet topology order",
        "pointnet_topology_mirror_tta": "PointNet topology + mirror TTA",
    }
    for key, label in comparison_labels.items():
        aggregate = comparators[key]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(_format_mean_std(aggregate[rate]) for rate in RATES)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired Seed Differences",
            "",
            "Negative values mean the first named method is worse.",
            "",
            "| Comparison | 0% | 25% | 50% | 75% |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    comparisons = (
        ("topology", "baseline", "topology - baseline"),
        ("topology_mirror_tta", "topology", "mirror TTA - topology"),
        ("topology_mirror_tta", "baseline", "mirror TTA - baseline"),
    )
    for left, right, label in comparisons:
        differences = []
        for rate in RATES:
            left_values = variants[left]["per_seed"]
            right_values = variants[right]["per_seed"]
            differences.append(
                _mean_std(
                    [
                        float(left_values[seed][rate]["mean_iou"])
                        - float(right_values[seed][rate]["mean_iou"])
                        for seed in ("42", "43", "44")
                    ]
                )
            )
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{value:+.4f} +/- {std:.4f}"
                for value, std in differences
            )
            + " |"
        )

    best_pointnetpp_clean = max(
        float(variants[variant]["aggregate"]["0.00"]["mean_iou"])
        for variant in VARIANT_LABELS
    )
    pointnet_clean = float(
        comparators["pointnet_topology_mirror_tta"]["0.00"]["mean_iou"]
    )
    if best_pointnetpp_clean < pointnet_clean - 0.15:
        decision = [
            "REJECT for the mainline.",
            "",
            "The strongest PointNet++ variant remains far below the PointNet "
            "topology + mirror TTA comparator in clean mIoU. The PointNet++ "
            "topology group also has very large seed variance, including a "
            "seed where `pier` is not learned and another where it dominates.",
            "",
            "This is a negative result for the current pure-PyTorch "
            "implementation and training recipe, not a general claim that "
            "PointNet++ is unsuitable for bridge segmentation.",
        ]
    else:
        decision = [
            "RETEST before adoption.",
            "",
            "The PointNet++ pilot does not yet show a decisive and stable "
            "advantage over the PointNet comparator.",
        ]

    lines.extend(
        [
            "",
            "## Decision",
            "",
            *decision,
            "",
            "## Artifact Paths",
            "",
            f"- Combined queue summary: `{PILOT_ROOT / 'pilot_summary.csv'}`",
            f"- Pilot manifest: `{PILOT_ROOT / 'pilot_manifest.json'}`",
            f"- Machine-readable analysis: `{OUTPUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    pilot = _load_pilot()
    comparators = _load_pointnet_comparators()
    payload = {
        "pilot": pilot,
        "pointnet_comparators": comparators,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(
        _build_markdown(pilot, comparators),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output_json": str(OUTPUT_JSON),
            "output_markdown": str(OUTPUT_MD),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
