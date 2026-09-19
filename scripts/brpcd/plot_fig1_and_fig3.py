"""Draw Fig. 1 (method overview) and Fig. 3 (per-class IoU) for the revision.

Fig. 1 contains no mirror / symmetry / TTA module: the discarded branch is
gone. Fig. 3 uses only numbers that exist in
``outputs/aligned_fusion_perclass_v1.md`` (side + side_left, alpha = 0.25).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT_DIR = Path("outputs/figures")
RATES = [0.0, 25.0, 50.0, 75.0]

# outputs/aligned_fusion_perclass_v1.md, side + side_left, alpha = 0.25.
PER_CLASS = {
    "girder": {
        "vanilla": [0.4186, 0.3769, 0.3310, 0.2804],
        "+ force-alignment": [0.4415, 0.3949, 0.3435, 0.2854],
    },
    "pier": {
        "vanilla": [0.8525, 0.7941, 0.7229, 0.6167],
        "+ force-alignment": [0.8639, 0.8071, 0.7365, 0.6295],
    },
    "deck": {
        "vanilla": [0.3753, 0.3254, 0.3094, 0.2863],
        "+ force-alignment": [0.3659, 0.3156, 0.2983, 0.2710],
    },
}

BOX_FILL = {
    "3d": "#e8eef7",
    "2d": "#eef3e8",
    "shared": "#f2f2f2",
    "train": "#fdeceb",
}
EDGE = "#333333"


def box(ax, x, y, w, h, text, fill="shared", fontsize=8.2, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=EDGE,
        facecolor=BOX_FILL[fill],
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.35,
    )
    return patch


def arrow(ax, start, end, dashed=False, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.1,
            color=EDGE,
            linestyle="--" if dashed else "-",
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def draw_fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = 0.155, 0.125
    y3, y2 = 0.735, 0.505

    # --- 3D branch -------------------------------------------------------
    x0 = 0.02
    box(ax, x0, y3, w, h, "Bridge point cloud\n(x, y, z, r, g, b)")
    box(
        ax,
        x0 + 0.183,
        y3,
        w,
        h,
        "Lightweight\nPointNet",
    )
    box(ax, x0 + 0.366, y3, w, h, "$P_{3D}$:\nper-point\nprobabilities")
    arrow(ax, (x0 + w, y3 + h / 2), (x0 + 0.183, y3 + h / 2))
    arrow(ax, (x0 + 0.183 + w, y3 + h / 2), (x0 + 0.366, y3 + h / 2))

    # --- 2D branch -------------------------------------------------------
    box(ax, x0, y2, w, h, "Virtual views\n(geometric\nfeatures only)")
    box(ax, x0 + 0.183, y2, w, h, "U-Net\nsegmentation")
    box(ax, x0 + 0.366, y2, w, h, "$P_{2D}$:\nper-pixel\nprobabilities")
    arrow(ax, (x0 + w, y2 + h / 2), (x0 + 0.183, y2 + h / 2))
    arrow(ax, (x0 + 0.183 + w, y2 + h / 2), (x0 + 0.366, y2 + h / 2))

    # --- Z-buffer --------------------------------------------------------
    box(
        ax,
        x0 + 0.558,
        y2 - 0.03,
        w + 0.045,
        h + 0.06,
        "Z-buffer\nvisibility\ntest",
        "shared",
    )
    arrow(
        ax,
        (x0 + 0.366 + w, y2 + h / 2),
        (x0 + 0.558, y2 + h / 2),
    )
    ax.text(
        x0 + 0.558 + (w + 0.045) / 2,
        y2 - 0.055,
        "2D evidence only on\nvisible points",
        ha="center",
        va="top",
        fontsize=7.2,
        color="#555555",
    )

    # --- fusion ----------------------------------------------------------
    box(
        ax,
        x0 + 0.558,
        y3 - 0.02,
        w + 0.045,
        h + 0.04,
        "Fusion:\n$\\arg\\max_c\\,(P_{3D}+\\alpha P_{2D})$",
        "shared",
        fontsize=8.0,
    )
    box(
        ax,
        x0 + 0.795,
        y3 + 0.09,
        w + 0.02,
        h + 0.02,
        "Component labels\ngirder / pier /\ndeck",
        "shared",
    )
    arrow(
        ax,
        (x0 + 0.366 + w, y3 + h / 2),
        (x0 + 0.558, y3 + h / 2),
    )
    arrow(
        ax,
        (x0 + 0.558 + w + 0.045, y3 + h / 2),
        (x0 + 0.795, y3 + 0.09 + (h + 0.02) / 2),
    )
    arrow(
        ax,
        (x0 + 0.558 + (w + 0.045) / 2, y2 - 0.03),
        (x0 + 0.558 + (w + 0.045) / 2, y3 - 0.02),
    )

    # --- training-time constraint ---------------------------------------
    box(
        ax,
        x0 + 0.183,
        0.285,
        w + 0.24,
        0.115,
        "Force-alignment constraint\n(masked soft-target CE on\nvisible 2D pixels only)",
        "train",
        fontsize=7.6,
        dashed=True,
    )
    arrow(
        ax,
        (x0 + 0.183 + (w + 0.24) / 2, 0.4),
        (x0 + 0.183 + (w + 0.24) / 2, y2 - 0.005),
        dashed=True,
        style="<|-|>",
    )
    ax.text(
        x0 + 0.183 + (w + 0.24) + 0.012,
        0.3425,
        "training only,\nno inference cost",
        ha="left",
        va="center",
        fontsize=7.2,
        color="#555555",
    )

    # --- protocol note ---------------------------------------------------
    ax.text(
        0.02,
        0.215,
        "Evaluation protocol: points are removed BEFORE the forward pass at 0 %, 25 %, 50 % and 75 %\n"
        "(removal follows virtual-view visibility); the model never sees the full geometry.",
        ha="left",
        va="top",
        fontsize=7.6,
        color="#333333",
        linespacing=1.5,
    )
    ax.text(
        0.02,
        0.055,
        "3D branch (light blue)   |   2D branch (light green)   |   shared / fusion (grey)   |   training-only (red dashed)",
        ha="left",
        va="top",
        fontsize=7.0,
        color="#666666",
    )

    fig.tight_layout()
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(OUT_DIR / f"fig1_method_overview{suffix}", dpi=300)
    plt.close(fig)


def draw_fig3() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=True)
    styles = {
        "vanilla": dict(color="#1f77b4", marker="o", linestyle="--"),
        "+ force-alignment": dict(color="#d62728", marker="D", linestyle="-"),
    }
    for ax, (cls, series) in zip(axes, PER_CLASS.items()):
        for label, values in series.items():
            ax.plot(RATES, values, linewidth=1.6, markersize=5, label=label, **styles[label])
        ax.set_title(f"{cls}", fontsize=10)
        ax.set_xticks(RATES)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.set_xlabel("Removed points (%)")
    axes[0].set_ylabel("IoU")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    axes[2].annotate(
        "deck degrades at every rate",
        xy=(75.0, 0.2710),
        xytext=(28.0, 0.232),
        fontsize=7.5,
        color="#8b1a1a",
        arrowprops=dict(arrowstyle="->", color="#8b1a1a", linewidth=0.9),
    )
    fig.suptitle(
        "Per-class IoU, side + side_left fusion (alpha = 0.25), three seeds",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"fig3_per_class_iou{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_fig1()
    draw_fig3()
    print(f"wrote fig1 and fig3 to {OUT_DIR}")


if __name__ == "__main__":
    main()
