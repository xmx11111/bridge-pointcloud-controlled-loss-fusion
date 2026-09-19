"""Draw Fig. 2 (BrPCD) and Fig. 4 (SemanticBridge external validation).

Every number comes from a recorded experiment file:

* BrPCD 3D baseline        -> outputs/main_table_v1.md
* BrPCD fusion / + force   -> outputs/aligned_fusion_table_v1.md (Table G,
                              ``side_left`` rows; the vanilla column is the
                              archived fusion sweep)
* SemanticBridge           -> outputs/semanticbridge_external_table_v1.json

No value is recomputed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RATES = [0.0, 25.0, 50.0, 75.0]
OUT_DIR = Path("outputs/figures")

# BrPCD, P1 evaluation family, surviving-point criterion (Section 5.2/5.3).
BRPCD = {
    "3D baseline (PointNet)": (
        [0.5096, 0.4235, 0.3484, 0.2617],
        [0.0591, 0.0457, 0.0282, 0.0320],
    ),
    "Fusion, side (alpha=0.25)": (
        [0.5508, 0.4850, 0.4251, 0.3450],
        [0.0649, 0.0667, 0.0667, 0.0649],
    ),
    "Fusion, side + side_left (alpha=0.25)": (
        [0.5488, 0.4988, 0.4544, 0.3945],
        [0.0517, 0.0572, 0.0592, 0.0598],
    ),
    "Fusion + force-alignment, side + side_left": (
        [0.5571, 0.5059, 0.4595, 0.3953],
        [0.0581, 0.0668, 0.0684, 0.0671],
    ),
}

STYLE = {
    "3D baseline (PointNet)": dict(color="#3b3b3b", marker="o", linestyle="-"),
    "Fusion, side (alpha=0.25)": dict(color="#1f77b4", marker="s", linestyle="--"),
    "Fusion, side + side_left (alpha=0.25)": dict(
        color="#2ca02c", marker="^", linestyle="-."
    ),
    "Fusion + force-alignment, side + side_left": dict(
        color="#d62728", marker="D", linestyle="-"
    ),
}


def load_external() -> dict[str, dict[str, list[float]]]:
    payload = json.loads(
        Path("outputs/semanticbridge_external_table_v1.json").read_text(
            encoding="utf-8"
        )
    )
    table: dict[str, dict[str, list[float]]] = {}
    wanted = {
        ("vanilla", "side_left", "surviving", "pointnet"): "3D baseline (PointNet)",
        (
            "vanilla",
            "side_left",
            "surviving",
            "fusion_side_alpha_0.25",
        ): "Fusion, side + side_left (alpha=0.25)",
        (
            "aligned",
            "side_left",
            "surviving",
            "fusion_side_alpha_0.25",
        ): "Fusion + force-alignment, side + side_left",
        ("vanilla", "side_left", "coverage", "pointnet"): "3D baseline (PointNet)",
        (
            "vanilla",
            "side_left",
            "coverage",
            "fusion_side_alpha_0.25",
        ): "Fusion, side + side_left (alpha=0.25)",
        (
            "aligned",
            "side_left",
            "coverage",
            "fusion_side_alpha_0.25",
        ): "Fusion + force-alignment, side + side_left",
    }
    for row in payload["rows"]:
        key = (row["method"], row["view"], row["metric"], row["target"])
        label = wanted.get(key)
        if label is None:
            continue
        bucket = table.setdefault(row["metric"], {})
        bucket[label] = (
            [row[f"r{rate / 100:.2f}"]["mean"] for rate in RATES],
            [row[f"r{rate / 100:.2f}"]["std"] for rate in RATES],
        )
    return table


def draw(ax, series: dict[str, tuple[list[float], list[float]]], title: str) -> None:
    for label, (mean, std) in series.items():
        style = STYLE[label]
        ax.errorbar(
            RATES,
            mean,
            yerr=std,
            capsize=3,
            linewidth=1.6,
            markersize=5,
            label=label,
            **style,
        )
    ax.set_xlabel("Removed points (%)")
    ax.set_ylabel("mIoU")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(RATES)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=7, loc="best", frameon=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    draw(ax, BRPCD, "(a) BrPCD, surviving-point criterion")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"fig2_brpcd_curves{suffix}", dpi=300)
    plt.close(fig)

    external = load_external()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    draw(
        axes[0],
        external["surviving"],
        "(a) SemanticBridge, surviving-point criterion",
    )
    draw(
        axes[1],
        external["coverage"],
        "(b) SemanticBridge, coverage-penalised",
    )
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(
            OUT_DIR / f"fig4_semanticbridge_curves{suffix}",
            dpi=300,
        )
    plt.close(fig)
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
