"""Draw Fig. 5 (qualitative success and failure cases).

The figure reads the auditable selection and per-point predictions produced by
``export_fig5_cases.py``.  For each selected block it displays:

1. the reference component labels,
2. the 3D PointNet prediction on the surviving points,
3. the side-view 2D evidence (masked where the point is invisible),
4. the probability-sum fusion prediction.

Removed points are kept as small grey dots so the 75% missing rate is visible;
only surviving points receive a colour-coded prediction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (_ROOT / "outputs" / "figures",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from paper_plot_style import configure_style  # noqa: E402


CLASS_NAMES = ("girder", "pier", "deck")
CLASS_COLORS = ("#0072B2", "#E69F00", "#D55E00")
REMOVED_COLOR = "#C9C9C9"
UNOBSERVED_2D_COLOR = "#9A9A9A"
CASE_DATA = _ROOT / "work" / "fig5_case_data_v1.npz"
CASE_METRICS = _ROOT / "outputs" / "fig5_case_metrics_v1.json"
FIGURE_DIR = _ROOT / "outputs" / "figures"


def _class_f1(
    labels: np.ndarray,
    prediction: np.ndarray,
    class_index: int,
) -> float:
    true_positive = int(
        np.count_nonzero((labels == class_index) & (prediction == class_index))
    )
    false_positive = int(
        np.count_nonzero((labels != class_index) & (prediction == class_index))
    )
    false_negative = int(
        np.count_nonzero((labels == class_index) & (prediction != class_index))
    )
    return 2.0 * true_positive / max(
        2 * true_positive + false_positive + false_negative,
        1,
    )


def _mean_iou(labels: np.ndarray, prediction: np.ndarray) -> float:
    ious = []
    for class_index in range(len(CLASS_NAMES)):
        true_positive = int(
            np.count_nonzero(
                (labels == class_index) & (prediction == class_index)
            )
        )
        false_positive = int(
            np.count_nonzero(
                (labels != class_index) & (prediction == class_index)
            )
        )
        false_negative = int(
            np.count_nonzero(
                (labels == class_index) & (prediction != class_index)
            )
        )
        ious.append(
            true_positive
            / max(true_positive + false_positive + false_negative, 1)
        )
    return float(np.mean(ious))


def _case_seed(
    metrics: dict[str, object],
    case_name: str,
) -> int:
    scene = metrics[f"{case_name}_scene"]
    block_index = int(metrics[f"{case_name}_block_index"])
    record = next(
        row
        for row in metrics["blocks"]
        if row["scene"] == scene and row["block_index"] == block_index
    )
    deltas = []
    for seed_record in record["seeds"]:
        prediction = np.load(CASE_DATA, allow_pickle=False)
        labels = prediction[f"{case_name}_seed{seed_record['seed']}_labels"]
        kept = prediction[f"{case_name}_seed{seed_record['seed']}_kept"]
        pointnet = prediction[
            f"{case_name}_seed{seed_record['seed']}_pointnet_prediction"
        ][kept]
        fusion = prediction[
            f"{case_name}_seed{seed_record['seed']}_fusion_prediction"
        ][kept]
        deltas.append(
            (
                _class_f1(labels[kept], fusion, 2)
                - _class_f1(labels[kept], pointnet, 2),
                int(seed_record["seed"]),
            )
        )
    deltas.sort()
    return deltas[len(deltas) // 2][1]


def _scatter_panel(
    axis: plt.Axes,
    xyz: np.ndarray,
    labels: np.ndarray,
    kept: np.ndarray,
    prediction: np.ndarray | None,
    title: str,
    footer: str,
    view: tuple[float, float] = (24.0, -58.0),
) -> None:
    removed = ~kept
    axis.scatter(
        xyz[removed, 0],
        xyz[removed, 1],
        xyz[removed, 2],
        s=1.8,
        c=REMOVED_COLOR,
        marker=".",
        alpha=0.55,
        depthshade=False,
        rasterized=True,
    )
    if prediction is None:
        for class_index, color in enumerate(CLASS_COLORS):
            mask = kept & (labels == class_index)
            axis.scatter(
                xyz[mask, 0],
                xyz[mask, 1],
                xyz[mask, 2],
                s=11.0 if class_index != 2 else 16.0,
                c=color,
                marker="o",
                edgecolors="white",
                linewidths=0.25,
                depthshade=False,
                rasterized=True,
            )
    else:
        unobserved = kept & (prediction < 0)
        axis.scatter(
            xyz[unobserved, 0],
            xyz[unobserved, 1],
            xyz[unobserved, 2],
            s=3.0,
            c=UNOBSERVED_2D_COLOR,
            marker=".",
            alpha=0.8,
            depthshade=False,
            rasterized=True,
        )
        for class_index, color in enumerate(CLASS_COLORS):
            mask = kept & (prediction == class_index)
            axis.scatter(
                xyz[mask, 0],
                xyz[mask, 1],
                xyz[mask, 2],
                s=14.0 if class_index != 2 else 22.0,
                c=color,
                marker="o",
                edgecolors="black" if class_index == 2 else "white",
                linewidths=0.35 if class_index == 2 else 0.2,
                depthshade=False,
                rasterized=True,
            )
        reference_deck = kept & (labels == 2)
        axis.scatter(
            xyz[reference_deck, 0],
            xyz[reference_deck, 1],
            xyz[reference_deck, 2],
            s=26.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.45,
            depthshade=False,
            rasterized=True,
        )

    axis.set_title(title, fontsize=8.6, pad=1.5)
    axis.text2D(
        0.5,
        -0.05,
        footer,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=6.6,
        linespacing=1.25,
    )
    axis.view_init(elev=view[0], azim=view[1])
    axis.set_proj_type("ortho")
    axis.set_box_aspect((1.0, 1.0, 0.58))
    axis.set_axis_off()
    finite = xyz[np.isfinite(xyz).all(axis=1)]
    center = finite.mean(axis=0)
    radius = max(float(np.max(np.ptp(finite, axis=0))) / 2.0, 1e-6)
    for setter, value in zip(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim),
        (
            (center[0] - radius, center[0] + radius),
            (center[1] - radius, center[1] + radius),
            (center[2] - radius * 0.6, center[2] + radius * 0.6),
        ),
    ):
        setter(value)


def _case_footers(
    labels: np.ndarray,
    kept: np.ndarray,
    pointnet: np.ndarray,
    fusion: np.ndarray,
    side_visible: np.ndarray,
) -> dict[str, str]:
    surviving_labels = labels[kept]
    visible_kept = int(np.count_nonzero(kept & side_visible))
    return {
        "reference": (
            f"deck: {int(np.count_nonzero(surviving_labels == 2))} kept / "
            f"{int(np.count_nonzero(labels == 2))} points\n"
            f"75% of points removed (grey)"
        ),
        "pointnet": (
            f"deck F1 {_class_f1(surviving_labels, pointnet, 2):.2f}\n"
            f"mIoU {_mean_iou(surviving_labels, pointnet):.2f}"
        ),
        "side": (
            f"2D evidence: {visible_kept}/{int(kept.sum())} kept points\n"
            "grey = no side-view evidence"
        ),
        "fusion": (
            f"deck F1 {_class_f1(surviving_labels, fusion, 2):.2f} "
            f"({_class_f1(surviving_labels, fusion, 2) - _class_f1(surviving_labels, pointnet, 2):+.2f})\n"
            f"mIoU {_mean_iou(surviving_labels, fusion):.2f}"
        ),
    }


def main() -> None:
    configure_style()
    metrics = json.loads(CASE_METRICS.read_text(encoding="utf-8"))
    data = np.load(CASE_DATA, allow_pickle=False)

    cases = []
    for case_name in ("success", "failure"):
        seed = _case_seed(metrics, case_name)
        scene = str(data[f"{case_name}_scene"][0])
        block = int(data[f"{case_name}_block_index"][0])
        labels = data[f"{case_name}_seed{seed}_labels"]
        kept = data[f"{case_name}_seed{seed}_kept"]
        xyz = data[f"{case_name}_seed{seed}_xyz_local"]
        pointnet_all = data[f"{case_name}_seed{seed}_pointnet_prediction"]
        fusion_all = data[f"{case_name}_seed{seed}_fusion_prediction"]
        pointnet = pointnet_all[kept]
        fusion = fusion_all[kept]
        side = data[f"{case_name}_seed{seed}_side_prediction"]
        side_visible = data[f"{case_name}_seed{seed}_side_visible"]
        side_display = np.full_like(side, -1, dtype=np.int8)
        side_display[kept & side_visible] = side[kept & side_visible]
        cases.append(
            {
                "name": case_name,
                "scene": scene,
                "block": block,
                "seed": seed,
                "labels": labels,
                "kept": kept,
                "xyz": xyz,
                "pointnet": pointnet_all,
                "fusion": fusion_all,
                "side": side_display,
                "footers": _case_footers(
                    labels,
                    kept,
                    pointnet,
                    fusion,
                    side_visible,
                ),
            }
        )

    figure = plt.figure(figsize=(10.2, 5.8))
    grid = figure.add_gridspec(
        2,
        4,
        left=0.025,
        right=0.99,
        top=0.90,
        bottom=0.12,
        wspace=0.02,
        hspace=0.30,
    )
    for row, case in enumerate(cases):
        axes = [
            figure.add_subplot(grid[row, column], projection="3d")
            for column in range(4)
        ]
        titles = (
            "reference labels",
            "3D baseline",
            "2D side evidence",
            "fusion (side + side_left, $\\alpha=0.25$)",
        )
        footers = (
            case["footers"]["reference"],
            case["footers"]["pointnet"],
            case["footers"]["side"],
            case["footers"]["fusion"],
        )
        labels = case["labels"]
        kept = case["kept"]
        xyz = case["xyz"]
        _scatter_panel(
            axes[0],
            xyz,
            labels,
            kept,
            None,
            titles[0],
            footers[0],
        )
        _scatter_panel(
            axes[1],
            xyz,
            labels,
            kept,
            case["pointnet"],
            titles[1],
            footers[1],
        )
        _scatter_panel(
            axes[2],
            xyz,
            labels,
            kept,
            case["side"],
            titles[2],
            footers[2],
        )
        _scatter_panel(
            axes[3],
            xyz,
            labels,
            kept,
            case["fusion"],
            titles[3],
            footers[3],
        )
        figure.text(
            0.012,
            0.885 - row * 0.475,
            ("Success case" if case["name"] == "success" else "Failure case")
            + f": {case['scene']}, block {case['block']}, seed {case['seed']}",
            ha="left",
            va="top",
            fontsize=8.4,
            fontweight="bold",
        )

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=name)
        for name, color in zip(CLASS_NAMES, CLASS_COLORS)
    ] + [
        Patch(facecolor=REMOVED_COLOR, edgecolor="none", label="removed (75%)"),
        Patch(
            facecolor=UNOBSERVED_2D_COLOR,
            edgecolor="none",
            label="no 2D evidence",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=4,
            label="GT deck point",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=7.4,
        handlelength=1.0,
        columnspacing=1.1,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.text(
        0.012,
        0.955,
        "Component predictions on surviving points after 75% viewpoint occlusion",
        ha="left",
        va="top",
        fontsize=9.5,
    )
    figure.text(
        0.012,
        0.925,
        "grey points are removed from the prediction; colour encodes the predicted class",
        ha="left",
        va="top",
        fontsize=7.4,
        color="#555555",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = FIGURE_DIR / f"fig5_cases{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
