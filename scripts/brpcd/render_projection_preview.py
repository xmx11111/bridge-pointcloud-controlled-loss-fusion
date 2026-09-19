from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROJECTION_ROOT = Path(r"D:\Codex\datasets\BrPCD\official\projection_192_v1")
OUTPUT = ROOT / "work" / "brpcd" / "projection_preview.png"
SCENES = (
    ("train", "c-bridge1"),
    ("test", "c-bridge4"),
)


def main() -> None:
    views = ("top", "side", "end")
    figure, axes = plt.subplots(
        len(SCENES) * 2,
        len(views),
        figsize=(12, 8),
        squeeze=False,
    )
    for scene_index, (split, scene_name) in enumerate(SCENES):
        path = PROJECTION_ROOT / split / f"{scene_name}.npz"
        with np.load(path) as data:
            images = data["images"]
            labels = data["labels"]
        row_rgb = scene_index * 2
        row_label = row_rgb + 1
        for view_index, view_name in enumerate(views):
            image = images[view_index, :3].transpose(1, 2, 0)
            axes[row_rgb, view_index].imshow(image)
            axes[row_rgb, view_index].set_title(f"{scene_name} {view_name} RGB")
            axes[row_rgb, view_index].axis("off")

            label_image = labels[view_index].astype(np.float64)
            label_image[label_image == 255] = np.nan
            axes[row_label, view_index].imshow(
                label_image,
                cmap="tab10",
                vmin=-0.5,
                vmax=2.5,
            )
            axes[row_label, view_index].set_title(
                f"{scene_name} {view_name} labels"
            )
            axes[row_label, view_index].axis("off")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
