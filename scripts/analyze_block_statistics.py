"""Paired block-level bootstrap analysis for the main fusion experiments.

Reads ``work/block_statistics_v1.json`` and writes:

* ``outputs/block_paired_statistics_v1.json``
* ``outputs/block_paired_statistics_v1.md``

The statistical unit is an evaluation block.  Seed-level values are first
averaged within each block, then 10,000 bootstrap resamples of the 352 blocks
are used to estimate 95% confidence intervals.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "work" / "block_statistics_v1.json"
OUTPUT_JSON = ROOT / "outputs" / "block_paired_statistics_v1.json"
OUTPUT_MD = ROOT / "outputs" / "block_paired_statistics_v1.md"

BOOTSTRAP = 10_000
RNG_SEED = 20260919
RATES = (0.0, 0.25, 0.5, 0.75)


def _key(row: dict[str, object]) -> tuple[str, int, float, int]:
    return (
        str(row["variant"]),
        int(row["seed"]),
        float(row["rate"]),
        int(row["dataset_index"]),
    )


def _index(
    records: list[dict[str, object]],
) -> dict[tuple[str, int, float, int], dict[str, object]]:
    return {_key(row): row for row in records}


def _block_values(
    left: dict[tuple[str, int, float, int], dict[str, object]],
    right: dict[tuple[str, int, float, int], dict[str, object]],
    variant: str,
    rate: float,
    metric: str,
) -> np.ndarray:
    block_values: dict[int, list[float]] = defaultdict(list)
    for (row_variant, _seed, row_rate, dataset_index), left_row in left.items():
        if row_variant != variant or row_rate != rate:
            continue
        right_key = (row_variant, _seed, row_rate, dataset_index)
        right_row = right[right_key]
        left_metrics = left_row["fusion"]
        right_metrics = right_row["pointnet"]
        left_value = (
            float(left_metrics["mean_iou"])
            if metric == "mean_iou"
            else float(left_metrics["per_class_iou"][2])
        )
        right_value = (
            float(right_metrics["mean_iou"])
            if metric == "mean_iou"
            else float(right_metrics["per_class_iou"][2])
        )
        block_values[dataset_index].append(left_value - right_value)
    return np.asarray(
        [
            float(np.mean(block_values[index]))
            for index in sorted(block_values)
        ],
        dtype=np.float64,
    )


def _variant_delta(
    records: dict[tuple[str, int, float, int], dict[str, object]],
    left_variant: str,
    right_variant: str,
    rate: float,
    metric: str,
) -> np.ndarray:
    block_values: dict[int, list[float]] = defaultdict(list)
    for (variant, seed, row_rate, dataset_index), row in records.items():
        if variant != left_variant or row_rate != rate:
            continue
        partner = records[(right_variant, seed, row_rate, dataset_index)]
        left_metrics = row["fusion"]
        right_metrics = partner["fusion"]
        left_value = (
            float(left_metrics["mean_iou"])
            if metric == "mean_iou"
            else float(left_metrics["per_class_iou"][2])
        )
        right_value = (
            float(right_metrics["mean_iou"])
            if metric == "mean_iou"
            else float(right_metrics["per_class_iou"][2])
        )
        block_values[dataset_index].append(left_value - right_value)
    return np.asarray(
        [
            float(np.mean(block_values[index]))
            for index in sorted(block_values)
        ],
        dtype=np.float64,
    )


def _two_sided_sign_test(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-12]
    count = len(nonzero)
    if count == 0:
        return 1.0
    positive = int(np.count_nonzero(nonzero > 0))
    negative = count - positive
    tail = min(positive, negative)
    probability = sum(
        math.comb(count, index)
        for index in range(tail + 1)
    ) / (2**count)
    return float(min(1.0, 2.0 * probability))


def _summarize(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    samples = rng.choice(
        values,
        size=(BOOTSTRAP, len(values)),
        replace=True,
    ).mean(axis=1)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return {
        "blocks": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "ci95": [float(ci_low), float(ci_high)],
        "positive_blocks": int(np.count_nonzero(values > 0)),
        "negative_blocks": int(np.count_nonzero(values < 0)),
        "zero_blocks": int(np.count_nonzero(values == 0)),
        "sign_test_p": _two_sided_sign_test(values),
        "bootstrap_p": float(
            min(
                1.0,
                2.0
                * min(
                    float(np.mean(samples <= 0.0)),
                    float(np.mean(samples >= 0.0)),
                ),
            )
        ),
    }


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    records = payload["records"]
    indexed = _index(records)
    rng = np.random.default_rng(RNG_SEED)

    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for variant in ("original", "aligned"):
        comparison = {}
        for rate in RATES:
            comparison[f"{rate:.2f}"] = {
                "mIoU": _summarize(
                    _block_values(
                        indexed,
                        indexed,
                        variant,
                        rate,
                        "mean_iou",
                    ),
                    rng,
                ),
                "deck_iou": _summarize(
                    _block_values(
                        indexed,
                        indexed,
                        variant,
                        rate,
                        "deck_iou",
                    ),
                    rng,
                ),
            }
        comparisons[f"{variant}_fusion_minus_pointnet"] = comparison

    aligned_minus_original = {}
    for rate in RATES:
        aligned_minus_original[f"{rate:.2f}"] = {
            "mIoU": _summarize(
                _variant_delta(
                    indexed,
                    "aligned",
                    "original",
                    rate,
                    "mean_iou",
                ),
                rng,
            ),
            "deck_iou": _summarize(
                _variant_delta(
                    indexed,
                    "aligned",
                    "original",
                    rate,
                    "deck_iou",
                ),
                rng,
            ),
        }
    comparisons["aligned_fusion_minus_original_fusion"] = aligned_minus_original

    output = {
        "protocol": payload["protocol"],
        "method": (
            "Paired block-level bootstrap; seed values averaged within block; "
            "10,000 bootstrap resamples over blocks."
        ),
        "comparisons": comparisons,
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Paired block-level statistics for the main fusion experiments",
        "",
        "Date: 2026-09-19",
        "",
        "Statistical unit: evaluation block. Three seed values are averaged "
        "within each block before 10,000 bootstrap resamples.",
        "",
        "| Comparison | Metric | 0% | 25% | 50% | 75% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, comparison in comparisons.items():
        for metric in ("mIoU", "deck_iou"):
            row = f"| {name} | {metric} | "
            row += " | ".join(
                f"{comparison[f'{rate:.2f}'][metric]['mean']:+.4f} "
                f"[{comparison[f'{rate:.2f}'][metric]['ci95'][0]:+.4f}, "
                f"{comparison[f'{rate:.2f}'][metric]['ci95'][1]:+.4f}]"
                for rate in RATES
            )
            row += " |"
            lines.append(row)
    lines.extend(
        [
            "",
            "`ci95` is the percentile bootstrap interval over blocks.",
            "`sign_test_p` and positive/negative block counts are stored in "
            "the JSON artifact.",
            "",
            f"Artifact: `{OUTPUT_JSON}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(OUTPUT_JSON),
                "output_md": str(OUTPUT_MD),
                "records": len(records),
            }
        )
    )


if __name__ == "__main__":
    main()
