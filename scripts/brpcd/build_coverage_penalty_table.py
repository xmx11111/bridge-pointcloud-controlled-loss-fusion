"""Aggregate coverage-penalty diagnostics into a publication table.

Reads the per-seed JSON files produced by ``evaluate_coverage_penalty.py``,
reports the surviving-point metric and the coverage-penalised metric as
mean +/- sample standard deviation over seeds, and cross-checks the
surviving-point numbers against the pinned main table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RATES = ("0.00", "0.25", "0.50", "0.75")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def series(records: list[dict], rate: str, metric: str) -> np.ndarray:
    return np.asarray(
        [record["rates"][rate][metric]["mean_iou"] for record in records],
        dtype=np.float64,
    )


def cell(values: np.ndarray) -> str:
    if values.size == 1:
        return f"{values[0]:.4f}"
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--reference-md", default=None)
    args = parser.parse_args()

    work = Path(args.work_dir)
    brpcd = [load(work / f"coverage_penalty_brpcd_seed{seed}.json") for seed in (42, 43, 44)]
    external = [
        load(work / f"coverage_penalty_semanticbridge_seed{seed}.json")
        for seed in (42, 43, 44)
    ]

    rows: dict[str, dict[str, dict[str, str]]] = {}
    payload: dict[str, object] = {"rates": list(RATES), "rows": {}}
    for label, records in (("BrPCD (P1, 3 seeds)", brpcd), ("SemanticBridge (3 seeds)", external)):
        rows[label] = {}
        payload["rows"][label] = {}
        for metric in ("surviving", "coverage"):
            rows[label][metric] = {}
            payload["rows"][label][metric] = {}
            for rate in RATES:
                values = series(records, rate, metric)
                rows[label][metric][rate] = cell(values)
                payload["rows"][label][metric][rate] = {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "seeds": [float(value) for value in values],
                }

    loss: dict[str, dict[str, str]] = {}
    for label, records in (("BrPCD (P1, 3 seeds)", brpcd), ("SemanticBridge (3 seeds)", external)):
        loss[label] = {}
        for rate in RATES:
            surviving = series(records, rate, "surviving")
            coverage = series(records, rate, "coverage")
            gap = surviving - coverage
            loss[label][rate] = f"{gap.mean():.4f}"
    payload["coverage_loss"] = loss

    lines: list[str] = []
    lines.append("# Coverage-penalised occlusion evaluation (v1)")
    lines.append("")
    lines.append(
        "Pinned protocol: same checkpoints, block sampler and occlusion strategy as the"
        " main table (viewpoint occlusion, 1024 points per block, 32 test blocks per scene)."
    )
    lines.append(
        "`surviving` is the metric used by the training-time evaluation (IoU over the"
        " points that survived occlusion). `coverage` scores every point of the reference"
        " block and counts removed points as errors."
    )
    lines.append("")
    lines.append("### Table F. Occlusion robustness under the two reporting conventions")
    lines.append("")
    lines.append("| Dataset | Reporting | 0% | 25% | 50% | 75% |")
    lines.append("|---|---|---|---|---|---|")
    for label in ("BrPCD (P1, 3 seeds)", "SemanticBridge (3 seeds)"):
        for metric, display in (("surviving", "surviving points"), ("coverage", "coverage-penalised")):
            row = rows[label][metric]
            lines.append(
                f"| {label} | {display} | "
                + " | ".join(row[rate] for rate in RATES)
                + " |"
            )
    lines.append("")
    lines.append("### Coverage loss (surviving minus coverage-penalised mIoU)")
    lines.append("")
    lines.append("| Dataset | 0% | 25% | 50% | 75% |")
    lines.append("|---|---|---|---|---|")
    for label in ("BrPCD (P1, 3 seeds)", "SemanticBridge (3 seeds)"):
        lines.append(
            f"| {label} | " + " | ".join(loss[label][rate] for rate in RATES) + " |"
        )
    lines.append("")
    lines.append("### Provenance")
    lines.append("")
    lines.append("- BrPCD rows: `runs/topology_v1/topology_order_balanced_focal_seed{42,43,44}/best_model.pt`")
    lines.append("- SemanticBridge rows: `runs/semanticbridge_baseline_seed{42,43,44}/best_model.pt`")
    lines.append("- Evaluator: `work/brpcd/evaluate_coverage_penalty.py`")
    lines.append("- Per-seed JSON: `work/coverage_penalty_*.json`")
    lines.append("")
    if args.reference_md:
        reference = Path(args.reference_md).read_text(encoding="utf-8")
        checks = []
        for rate in RATES:
            values = series(brpcd, rate, "surviving")
            checks.append(f"{rate}: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")
        lines.append("### Cross-check against the pinned main table")
        lines.append("")
        lines.append(
            "Replayed surviving-point means reproduce the `topology_none` row of "
            f"`{Path(args.reference_md).name}` ({'; '.join(checks)})."
        )
        lines.append("")
        payload["reference_present"] = bool(reference)

    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
