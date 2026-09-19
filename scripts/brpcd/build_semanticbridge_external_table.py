"""Aggregate the SemanticBridge external-validation fusion sweep.

Reads the JSON files written by ``evaluate_geometry_2d3d_fusion.py`` under
``work/sb_external/fusion`` and reports, per method/view/rate, the
surviving-point and coverage-penalised mean IoU as mean +/- std over seeds.
Pairs every fusion configuration with the 3D baseline of the same seed so the
deltas are within-seed.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

FUSION_DIR = Path("work/sb_external/fusion")
OUT_MD = Path("outputs/semanticbridge_external_table_v1.md")
OUT_JSON = Path("outputs/semanticbridge_external_table_v1.json")

RATES = (0.0, 0.25, 0.5, 0.75)
ALPHA_TAG = "fusion_side_alpha_0.25"
FILENAME = re.compile(
    r"sb_(?P<method>vanilla|aligned)_(?P<view>side|side_left)"
    r"_seed(?P<seed>\d+)_r(?P<rate>[\d.]+)\.json$"
)


def load_runs() -> dict[tuple[str, str, float], dict[int, dict]]:
    runs: dict[tuple[str, str, float], dict[int, dict]] = {}
    for path in sorted(FUSION_DIR.glob("sb_*.json")):
        match = FILENAME.search(path.name)
        if match is None:
            continue
        key = (
            match.group("method"),
            match.group("view"),
            round(float(match.group("rate")), 4),
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(key, {})[int(match.group("seed"))] = payload
    return runs


def pick(payload: dict, section: str, method: str) -> float | None:
    for row in payload.get(section, []):
        if row.get("method") == method:
            return float(row["mean_iou"])
    return None


def stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def cell(mean: float, std: float) -> str:
    if mean != mean:
        return "NA"
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no fusion results under {FUSION_DIR}")

    table: list[dict[str, object]] = []
    lines: list[str] = []
    for method in ("vanilla", "aligned"):
        for view in ("side", "side_left"):
            seeds = sorted(
                set.intersection(
                    *[
                        set(runs.get((method, view, rate), {}))
                        for rate in RATES
                    ]
                )
            )
            if not seeds:
                print(f"skip {method}/{view}: incomplete")
                continue
            label_view = "side" if view == "side" else "side + side_left"
            for metric, section, method_key, name in (
                (
                    "surviving",
                    "overall",
                    "pointnet",
                    "三维基线（存活点口径）",
                ),
                (
                    "coverage",
                    "overall_coverage",
                    "pointnet",
                    "三维基线（覆盖惩罚口径）",
                ),
                (
                    "surviving",
                    "overall",
                    ALPHA_TAG,
                    f"融合（{label_view}，α=0.25，存活点口径）",
                ),
                (
                    "coverage",
                    "overall_coverage",
                    ALPHA_TAG,
                    f"融合（{label_view}，α=0.25，覆盖惩罚口径）",
                ),
            ):
                row: dict[str, object] = {
                    "method": method,
                    "view": view,
                    "metric": metric,
                    "target": method_key,
                    "seeds": seeds,
                }
                cells = []
                for rate in RATES:
                    values = []
                    for seed in seeds:
                        payload = runs[(method, view, rate)][seed]
                        value = pick(payload, section, method_key)
                        if value is not None:
                            values.append(value)
                    mean, std = stats(values)
                    row[f"r{rate:.2f}"] = {"mean": mean, "std": std, "n": len(values)}
                    cells.append(cell(mean, std))
                table.append(row)
                lines.append(
                    f"| {method} | {name} | " + " | ".join(cells) + " |"
                )

            # Within-seed delta of the fused configuration against the same
            # seed's 3D baseline, coverage-penalised.
            for metric, section, name in (
                ("surviving", "overall", "融合增益（存活点口径）"),
                ("coverage", "overall_coverage", "融合增益（覆盖惩罚口径）"),
            ):
                cells = []
                for rate in RATES:
                    deltas = []
                    for seed in seeds:
                        payload = runs[(method, view, rate)][seed]
                        fused = pick(payload, section, ALPHA_TAG)
                        base = pick(payload, section, "pointnet")
                        if fused is not None and base is not None:
                            deltas.append(fused - base)
                    mean, std = stats(deltas)
                    cells.append(cell(mean, std))
                lines.append(
                    f"| {method} | {name} | " + " | ".join(cells) + " |"
                )

    header = "| 配置 | 指标 | 0% | 25% | 50% | 75% |\n|---|---|---|---|---|---|"
    body = "\n".join(lines)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        "# 第二数据集（SemanticBridge）外部验证：融合与力对齐约束\n\n"
        "口径说明：\"存活点\"仅在被保留下来的点上计算 IoU；\"覆盖惩罚\"在参考块的\n"
        "全部原始点上计算，被移除的点一律计为错误。数值为三个随机种子的\n"
        "均值 ± 样本标准差；\"融合增益\"为同一随机种子下融合减去三维基线。\n\n"
        f"{header}\n{body}\n",
        encoding="utf-8",
    )
    OUT_JSON.write_text(
        json.dumps({"rows": table}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{header}\n{body}")
    print(f"\nwrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
