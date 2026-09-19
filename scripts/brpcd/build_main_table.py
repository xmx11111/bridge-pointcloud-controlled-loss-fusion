"""Rebuild the paper's main tables from the canonical result files.

Every row is traced back to a stored artifact, and the script refuses to mix
recipes: the pinned protocol is topology_weight=1.0, seeds 42/43/44, P1 test
family c-bridge4, and evaluation rates 0/25/50/75 % missing points.

Run from the project root:

    $env:PYTHONPATH = "src"
    D:\\Codex\\envs\\pointcloud\\Scripts\\python.exe work\\brpcd\\build_main_table.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"

SEEDS = (42, 43, 44)
RATES = (0.0, 0.25, 0.5, 0.75)
ELIGIBLE_FOLDS = ("test_c-bridge4", "test_c-bridge5")
INELIGIBLE_FOLDS = (
    "test_c-bridge1",
    "test_c-bridge2",
    "test_c-bridge3",
    "test_s-bridge1",
)

WARNINGS: list[str] = []
NOTES: list[str] = []
PROVENANCE: list[dict[str, Any]] = []


def load_json(path: Path) -> Any:
    if not path.is_file():
        WARNINGS.append(f"missing artifact: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return float("nan"), float("nan"), 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return (
        statistics.fmean(values),
        statistics.stdev(values),
        len(values),
    )


def rate_key(value: float) -> float:
    return round(float(value), 2)


def fmt(mean: float, sd: float) -> str:
    if mean != mean:  # NaN
        return "n/a"
    return f"{mean:.4f} +/- {sd:.4f}"


def collect_p1_no_tta() -> dict[float, dict[str, Any]]:
    """P1 baseline: topology-trained PointNet without TTA, from run metrics."""
    collected: dict[float, dict[str, Any]] = {}
    seeds_seen: set[int] = set()
    topology_weights: set[float] = set()
    epochs_seen: set[int] = set()
    for seed in SEEDS:
        path = (
            RUNS
            / "topology_v1"
            / f"topology_order_balanced_focal_seed{seed}"
            / "metrics.json"
        )
        metrics = load_json(path)
        if metrics is None:
            continue
        config = metrics.get("configuration", {})
        seeds_seen.add(int(metrics.get("seed", -1)))
        topology_weights.add(float(config.get("topology_weight", -1)))
        epochs_seen.add(int(config.get("epochs", -1)))
        for key, record in metrics.get("test_by_occlusion", {}).items():
            collected.setdefault(rate_key(float(key)), []).append(
                {
                    "seed": seed,
                    "mean_iou": float(record["mean_iou"]),
                    "per_class_iou": [float(v) for v in record["per_class_iou"]],
                    "points": int(record["points"]),
                }
            )
    PROVENANCE.append(
        {
            "table": "A",
            "row": "topology_none",
            "source": "runs/topology_v1/topology_order_balanced_focal_seed{42,43,44}/metrics.json",
            "topology_weight": sorted(topology_weights),
            "epochs": sorted(epochs_seen),
            "seeds": sorted(seeds_seen),
        }
    )
    if topology_weights != {1.0}:
        WARNINGS.append(
            f"P1 baseline topology_weight is {sorted(topology_weights)}, expected [1.0]"
        )
    if seeds_seen != set(SEEDS):
        WARNINGS.append(f"P1 baseline seeds are {sorted(seeds_seen)}, expected {list(SEEDS)}")
    return collected


def collect_records(
    records: list[dict[str, Any]],
    selector,
    group_key: str | None = None,
) -> dict[Any, dict[float, list[dict[str, Any]]]]:
    out: dict[Any, dict[float, list[dict[str, Any]]]] = {}
    for record in records:
        if not selector(record):
            continue
        key = group_key(record) if group_key else "all"
        out.setdefault(key, {}).setdefault(rate_key(record["rate"]), []).append(
            {
                "seed": record.get("seed"),
                "fold": record.get("fold"),
                "mean_iou": float(record["mean_iou"]),
                "per_class_iou": [float(v) for v in record["per_class_iou"]],
                "points": int(record.get("points", 0)),
            }
        )
    return out


def table_from_groups(
    groups: dict[Any, dict[float, list[dict[str, Any]]]],
) -> dict[Any, dict[str, Any]]:
    table: dict[Any, dict[str, Any]] = {}
    for group, by_rate in groups.items():
        entry: dict[str, Any] = {}
        for rate, records in by_rate.items():
            entry[f"{rate:.2f}"] = {
                "mean_iou": summarize([r["mean_iou"] for r in records]),
                "per_class_iou": [
                    summarize([r["per_class_iou"][index] for r in records])
                    for index in range(3)
                ],
                "points": sorted({r["points"] for r in records}),
                "n": len(records),
            }
        table[group] = entry
    return table


def keyed_records(records: list[dict[str, Any]], selector) -> dict[tuple, dict[str, Any]]:
    """Index records by (fold, seed, rate) after applying a selector."""
    out: dict[tuple, dict[str, Any]] = {}
    for record in records:
        if not selector(record):
            continue
        key = (
            record.get("fold"),
            int(record.get("seed", -1)),
            rate_key(record["rate"]),
        )
        out[key] = {
            "fold": record.get("fold"),
            "seed": record.get("seed"),
            "mean_iou": float(record["mean_iou"]),
            "per_class_iou": [float(v) for v in record["per_class_iou"]],
            "points": int(record.get("points", 0)),
        }
    return out


def aggregate(keyed: dict[tuple, dict[str, Any]], predicate=None) -> dict[str, Any]:
    """Mean +/- sample sd at each rate, using the project's LOBO convention.

    Seeds are averaged inside each family fold first, and the reported
    standard deviation is then taken over folds. This is what the archived
    scope tables used, e.g. always_none at 0 % missing is
    0.3801 +/- 0.1227 (6 folds) rather than 0.3801 +/- 0.1420 (18 runs).
    """
    by_rate: dict[float, list[dict[str, Any]]] = {}
    for (fold, _seed, rate), record in keyed.items():
        if predicate is not None and not predicate(fold):
            continue
        by_rate.setdefault(rate, []).append(record)
    table: dict[str, Any] = {}
    for rate, records in by_rate.items():
        per_fold: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            per_fold.setdefault(record["fold"], []).append(record)
        units = [
            {
                "mean_iou": statistics.fmean([r["mean_iou"] for r in group]),
                "per_class_iou": [
                    statistics.fmean([r["per_class_iou"][index] for r in group])
                    for index in range(3)
                ],
            }
            for group in per_fold.values()
        ]
        table[f"{rate:.2f}"] = {
            "mean_iou": summarize([u["mean_iou"] for u in units]),
            "per_class_iou": [
                summarize([u["per_class_iou"][index] for u in units])
                for index in range(3)
            ],
            "points": sorted({r["points"] for r in records}),
            "n": len(units),
            "n_runs": len(records),
        }
    return table


def combine(
    base: dict[tuple, dict[str, Any]],
    alternate: dict[tuple, dict[str, Any]],
    use_alternate,
) -> dict[tuple, dict[str, Any]]:
    """Swap in the alternate record for the keys where the predicate holds."""
    out: dict[tuple, dict[str, Any]] = {}
    for key, record in base.items():
        swap = key in alternate and use_alternate(key[0])
        out[key] = alternate[key] if swap else record
    return out


def collect_2d3d() -> dict[str, dict[str, Any]]:
    # The evaluator always names the fused row `fusion_side_alpha_*`; which
    # views it used is recorded in the file's configuration block instead.
    configs = (
        ("side view", "2D-3D_topo1_512_side_seed{seed}_r{rate}.json"),
        ("side + side_left", "2D-3D_topo1_512_side_left_seed{seed}_r{rate}.json"),
    )
    collected: dict[str, dict[float, list[dict[str, Any]]]] = {}
    for view_label, pattern in configs:
        for seed in SEEDS:
            for rate in RATES:
                path = OUTPUTS / pattern.format(
                    seed=seed, rate=f"{rate:.2f}"
                )
                payload = load_json(path)
                if payload is None:
                    continue
                views = payload.get("configuration", {}).get(
                    "side_views", []
                )
                projection_checkpoint = str(
                    payload.get("configuration", {}).get(
                        "projection_checkpoint", ""
                    )
                )
                if f"seed{seed}" not in projection_checkpoint:
                    WARNINGS.append(
                        f"{path.name}: 2D checkpoint '{projection_checkpoint}' "
                        f"does not match the 3D seed {seed}"
                    )
                for row_label, method in (
                    ("topology only (paired)", "pointnet"),
                    ("+ 2D, alpha=1", "fusion_side_alpha_1"),
                    ("+ 2D, alpha=0.25", "fusion_side_alpha_0.25"),
                ):
                    record = next(
                        (
                            item
                            for item in payload.get("overall", [])
                            if item.get("method") == method
                        ),
                        None,
                    )
                    if record is None:
                        WARNINGS.append(f"{path.name}: no '{method}' record")
                        continue
                    key = f"{view_label} ({'+'.join(views)}) | {row_label}"
                    collected.setdefault(key, {}).setdefault(rate, []).append(
                        {
                            "seed": seed,
                            "mean_iou": float(record["mean_iou"]),
                            "per_class_iou": [
                                float(v) for v in record["per_class_iou"]
                            ],
                            "points": int(record.get("points", 0)),
                        }
                    )
    PROVENANCE.append(
        {
            "table": "D",
            "row": "2D-3D fusion",
            "source": "outputs/2D-3D_topo1_512_{side,side_left}_seed{42,43,44}.json",
            "note": "3D backbone is the weight-1.0 topology checkpoint; paired sampling",
        }
    )
    return table_from_groups(collected)


def collect_external_baseline() -> dict[str, dict[str, Any]]:
    """BridgeNetv2 adapted to the pinned P1 split, budget, and metrics."""
    collected: dict[float, list[dict[str, Any]]] = {}
    seeds_seen: set[int] = set()
    for seed in SEEDS:
        path = RUNS / f"bridgenetv2_baseline_v2_seed{seed}" / "metrics.json"
        metrics = load_json(path)
        if metrics is None:
            continue
        config = metrics.get("configuration", {})
        observed_seed = int(metrics.get("seed", -1))
        seeds_seen.add(observed_seed)
        if observed_seed != seed:
            WARNINGS.append(
                f"{path}: seed {observed_seed} does not match path seed {seed}"
            )
        if metrics.get("backbone") != "bridgenetv2":
            WARNINGS.append(f"{path}: unexpected backbone")
        if int(config.get("pad_to_points", -1)) != 1024:
            WARNINGS.append(f"{path}: pad_to_points is not 1024")
        if int(config.get("test_blocks_per_scene", -1)) != 32:
            WARNINGS.append(f"{path}: test_blocks_per_scene is not 32")
        for key, record in metrics.get("test_by_occlusion", {}).items():
            collected.setdefault(rate_key(float(key)), []).append(
                {
                    "seed": seed,
                    "mean_iou": float(record["mean_iou"]),
                    "per_class_iou": [
                        float(value) for value in record["per_class_iou"]
                    ],
                    "points": int(record["points"]),
                }
            )
    PROVENANCE.append(
        {
            "table": "E",
            "row": "BridgeNetv2 external baseline",
            "source": "runs/bridgenetv2_baseline_v2_seed{42,43,44}/metrics.json",
            "note": (
                "same P1 split, block sampler, 20-epoch budget and occlusion "
                "protocol; fixed-cardinality padding for occluded blocks"
            ),
        }
    )
    if seeds_seen != set(SEEDS):
        WARNINGS.append(
            f"BridgeNetv2 seeds are {sorted(seeds_seen)}, "
            f"expected {list(SEEDS)}"
        )
    return table_from_groups({"BridgeNetv2 (matched protocol)": collected})


def render_table(title: str, table: dict[Any, dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", ""]
    header = "| Row | " + " | ".join(f"{rate:.0%}" for rate in RATES) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(RATES) + 1))
    for row, by_rate in table.items():
        cells = []
        for rate in RATES:
            entry = by_rate.get(f"{rate:.2f}")
            cells.append(fmt(*entry["mean_iou"][:2]) if entry else "n/a")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_per_class(
    title: str,
    table: dict[Any, dict[str, Any]],
    rate: float,
) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| Row | girder | pier | deck |")
    lines.append("|---|---|---|---|")
    for row, by_rate in table.items():
        entry = by_rate.get(f"{rate:.2f}")
        if not entry:
            lines.append(f"| {row} | n/a | n/a | n/a |")
            continue
        cells = [fmt(*entry["per_class_iou"][i][:2]) for i in range(3)]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> int:
    p1_none = collect_p1_no_tta()
    p1_none_table = table_from_groups({"topology_none": p1_none})

    main_tta = load_json(WORK / "main_tta_gating_results.json")
    tta_records = (main_tta or {}).get("records", [])
    p1_tta = table_from_groups(
        {
            f"topology_{mode}_tta": collect_records(
                tta_records, lambda r, m=mode: r["mode"] == m
            )["all"]
            for mode in ("fixed", "geometry")
            if any(r["mode"] == mode for r in tta_records)
        }
    )
    PROVENANCE.append(
        {
            "table": "A",
            "row": "topology_{fixed,geometry}_tta",
            "source": "work/main_tta_gating_results.json",
            "note": "inference-only TTA over the same weight-1.0 checkpoints",
        }
    )

    classwise = load_json(WORK / "classwise_fusion_results.json")
    lobo_none_keyed = keyed_records(
        (classwise or {}).get("records", []),
        lambda r: r.get("condition") == "none",
    )

    tta_weighted = load_json(WORK / "tta_weight_mode_results.json")
    tta_weighted_records = (tta_weighted or {}).get("records", [])
    lobo_tta_keyed: dict[str, dict[tuple, dict[str, Any]]] = {}
    for mode in ("fixed", "geometry"):
        lobo_tta_keyed[mode] = keyed_records(
            tta_weighted_records,
            lambda r, m=mode: r.get("method") == "topology_tta" and r["mode"] == m,
        )
    expected_folds = set(ELIGIBLE_FOLDS) | set(INELIGIBLE_FOLDS)
    for label, keyed in (
        ("LOBO none", lobo_none_keyed),
        ("LOBO fixed TTA", lobo_tta_keyed["fixed"]),
        ("LOBO geometry TTA", lobo_tta_keyed["geometry"]),
    ):
        observed = {fold for fold, _seed, _rate in keyed}
        if observed != expected_folds:
            WARNINGS.append(
                f"{label}: folds {sorted(observed)} != expected {sorted(expected_folds)}"
            )
    PROVENANCE.append(
        {
            "table": "B/C",
            "row": "LOBO none / fixed / geometry",
            "source": "work/classwise_fusion_results.json (condition=none), work/tta_weight_mode_results.json (method=topology_tta)",
            "note": "6 family folds x 3 seeds; weight-1.0 LOBO checkpoints",
        }
    )

    p1_table: dict[str, Any] = {}
    p1_table.update(p1_none_table)
    p1_table.update(p1_tta)
    fusion_table = collect_2d3d()
    external_table = collect_external_baseline()

    baseline_p1 = p1_none_table["topology_none"]["0.00"]["mean_iou"][0]
    paired_rows = [
        row for row in fusion_table if row.endswith("topology only (paired)")
    ]
    if paired_rows:
        paired = fusion_table[paired_rows[0]]["0.00"]["mean_iou"][0]
        NOTES.append(
            f"2D-3D evaluator baseline {paired:.4f} vs Table A baseline "
            f"{baseline_p1:.4f} (delta {paired - baseline_p1:+.4f}). The fusion "
            "evaluator draws its own test blocks, so only compare rows within "
            "Table D, and quote fusion gains against the paired row."
        )
    lines: list[str] = []
    lines.append("# Main Table v1 (unified protocol)")
    lines.append("")
    lines.append(
        "Pinned protocol: `topology_weight=1.0`, seeds 42/43/44, P1 test family "
        "`c-bridge4`, evaluation rates 0/25/50/75 % missing points. "
        "Values are mean +/- sample standard deviation. P1 rows (Tables A and D) "
        "report the sample sd over the three seeds; LOBO rows (Tables B and C) "
        "average the seeds inside each family fold first and then report the "
        "sample sd over folds, which is the convention the archived scope "
        "tables used."
    )
    lines.append("")
    lines.extend(render_table("Table A. P1 component segmentation (mIoU)", p1_table))
    lines.extend(
        render_per_class("Table A2. P1 per-class IoU at 0 % missing", p1_table, 0.0)
    )
    lines.extend(
        render_per_class("Table A3. P1 per-class IoU at 75 % missing", p1_table, 0.75)
    )

    scope_table: dict[str, Any] = {}
    for label, fold_predicate in (
        ("mirror_eligible", lambda fold: fold in ELIGIBLE_FOLDS),
        ("mirror_ineligible", lambda fold: fold in INELIGIBLE_FOLDS),
    ):
        scope_table[f"{label} | none"] = aggregate(lobo_none_keyed, fold_predicate)
        for mode in ("fixed", "geometry"):
            scope_table[f"{label} | {mode}"] = aggregate(
                lobo_tta_keyed[mode], fold_predicate
            )
    lines.extend(render_table("Table B. Bridge-type scope (LOBO, mIoU)", scope_table))

    eligible = lambda fold: fold in ELIGIBLE_FOLDS
    policy_table: dict[str, Any] = {
        "always_none": aggregate(lobo_none_keyed),
        "always_fixed": aggregate(lobo_tta_keyed["fixed"]),
        "scope_fixed": aggregate(
            combine(lobo_none_keyed, lobo_tta_keyed["fixed"], eligible)
        ),
        "scope_geometry": aggregate(
            combine(lobo_none_keyed, lobo_tta_keyed["geometry"], eligible)
        ),
    }
    NOTES.append(
        "scope_fixed - always_none at 0 %: "
        f"{policy_table['scope_fixed']['0.00']['mean_iou'][0] - policy_table['always_none']['0.00']['mean_iou'][0]:+.4f}"
    )
    lines.extend(render_table("Table C. Scope-aware inference policy (LOBO, mIoU)", policy_table))
    lines.extend(render_table("Table D. 2D-3D side-view fusion, P1 (mIoU)", fusion_table))
    lines.extend(
        render_per_class(
            "Table D2. 2D-3D fusion per-class IoU at 0 % missing",
            fusion_table,
            0.0,
        )
    )
    lines.extend(
        render_table(
            "Table E. BridgeNetv2 external baseline, P1 (mIoU)",
            external_table,
        )
    )
    lines.extend(
        render_per_class(
            "Table E2. External baseline per-class IoU at 0 % missing",
            external_table,
            0.0,
        )
    )
    lines.extend(
        render_per_class(
            "Table E3. External baseline per-class IoU at 75 % missing",
            external_table,
            0.75,
        )
    )
    NOTES.append(
        "Table E is an external baseline under the project's own data and "
        "training budget, not the published BridgeNetv2 synthetic-data result. "
        "Occluded blocks repeat surviving points to satisfy its fixed kNN "
        "cardinality; predictions are sliced back before metrics."
    )

    lines.append("## Provenance")
    lines.append("")
    lines.append("| Table | Row | Source |")
    lines.append("|---|---|---|")
    for item in PROVENANCE:
        lines.append(f"| {item['table']} | {item['row']} | `{item['source']}` |")
    lines.append("")
    lines.append("## Consistency Checks")
    lines.append("")
    if WARNINGS:
        for warning in WARNINGS:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- All sources present; pinned protocol satisfied.")
    lines.append("")
    if NOTES:
        lines.append("### Cross-checks")
        lines.append("")
        for note in NOTES:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("### Excluded by design")
    lines.append("")
    lines.append(
        "- `runs/seg_tuning_topo05_*` and `runs/seg_tuning_ohem_*`: "
        "`topology_weight=0.5`, kept as ablations only."
    )
    lines.append(
        "- The archived `+0.085` 2D-3D figure: anchored on the weight-0.5 "
        "seed-42 checkpoint; Table D uses the weight-1.0 checkpoints instead."
    )
    lines.append("")

    (WORK / "main_table_v1.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "protocol": {
            "topology_weight": 1.0,
            "seeds": list(SEEDS),
            "test_family": "c-bridge4",
            "rates": list(RATES),
            "std": "sample (ddof=1)",
        },
        "table_a_p1": p1_table,
        "table_b_scope": scope_table,
        "table_c_policy": policy_table,
        "table_d_fusion": fusion_table,
        "table_e_external": external_table,
        "provenance": PROVENANCE,
        "warnings": WARNINGS,
    }
    (WORK / "main_table_v1.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    rows = ["table,row,rate,metric,mean,std,n"]
    for name, table in (
        ("A", p1_table),
        ("B", scope_table),
        ("C", policy_table),
        ("D", fusion_table),
        ("E", external_table),
    ):
        for row, by_rate in table.items():
            for rate, entry in sorted(by_rate.items()):
                mean, sd, n = entry["mean_iou"]
                rows.append(f"{name},{row},{rate},mean_iou,{mean},{sd},{n}")
                for index, cls in enumerate(("girder", "pier", "deck")):
                    cmean, csd, cn = entry["per_class_iou"][index]
                    rows.append(f"{name},{row},{rate},{cls},{cmean},{csd},{cn}")
    (WORK / "main_table_v1.csv").write_text("\n".join(rows), encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
