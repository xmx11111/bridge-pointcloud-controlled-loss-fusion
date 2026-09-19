from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bridge_seg.data import load_scene_npz
from bridge_seg.symmetry import estimate_bridge_frame


# (u axis, v axis, depth axis, keep nearest depth)
VIEWS = {
    "top": (0, 1, 2, "max"),
    "side": (0, 2, 1, "max"),
    "end": (1, 2, 0, "max"),
}


def _normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - low) / (high - low)


def project_scene(
    xyz: np.ndarray,
    labels: np.ndarray,
    resolution: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Render label-free orthographic virtual-camera views.

    BrPCD stores constant red RGB, so an RGB-based copy of Point-YOLO has no
    color signal. These views preserve the no-registration and inverse-mapping
    properties while replacing RGB with four geometry channels:
    depth closeness, normalized longitudinal position, normalized vertical
    position, and visibility.
    """

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if labels.shape != (len(xyz),):
        raise ValueError("labels must have shape [N]")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    frame = estimate_bridge_frame(xyz)
    axes = np.stack(
        [frame.longitudinal, frame.transverse, frame.vertical],
        axis=1,
    )
    coordinates = (xyz - frame.origin) @ axes
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    extent = np.maximum(upper - lower, 1e-12)

    normalized = (coordinates - lower[None, :]) / extent[None, :]
    view_names = list(VIEWS)
    view_count = len(view_names)
    images = np.zeros(
        (view_count, 4, resolution, resolution),
        dtype=np.uint8,
    )
    label_images = np.full(
        (view_count, resolution, resolution),
        255,
        dtype=np.uint8,
    )
    masks = np.zeros(
        (view_count, resolution, resolution),
        dtype=np.uint8,
    )
    view_indices = np.zeros(
        (view_count, len(xyz), 2),
        dtype=np.int16,
    )
    point_depths = np.full(
        (view_count, len(xyz)),
        np.nan,
        dtype=np.float32,
    )
    depth_maps = np.full(
        (view_count, resolution, resolution),
        np.nan,
        dtype=np.float32,
    )

    for view_index, (u_axis, v_axis, depth_axis, keep) in enumerate(
        VIEWS.values()
    ):
        u_index = np.clip(
            np.floor(normalized[:, u_axis] * (resolution - 1)).astype(
                np.int64
            ),
            0,
            resolution - 1,
        )
        v_index = np.clip(
            np.floor(normalized[:, v_axis] * (resolution - 1)).astype(
                np.int64
            ),
            0,
            resolution - 1,
        )
        depth = coordinates[:, depth_axis]
        flat_index = v_index * resolution + u_index
        depth_buffer = np.full(
            resolution * resolution,
            -np.inf if keep == "max" else np.inf,
            dtype=np.float64,
        )
        if keep == "max":
            np.maximum.at(depth_buffer, flat_index, depth)
        else:
            np.minimum.at(depth_buffer, flat_index, depth)
        tolerance = max(float(np.ptp(depth)) * 1e-6, 1e-9)
        visible = np.isclose(
            depth,
            depth_buffer[flat_index],
            rtol=0.0,
            atol=tolerance,
        )
        visible_point = np.flatnonzero(visible)
        visible_index = flat_index[visible]
        visible_depth = depth[visible]

        label_image = label_images[view_index].reshape(-1)
        mask_image = masks[view_index].reshape(-1)
        image_depth = images[view_index, 0].reshape(-1)
        image_longitudinal = images[view_index, 1].reshape(-1)
        image_vertical = images[view_index, 2].reshape(-1)
        image_visibility = images[view_index, 3].reshape(-1)
        raw_depth_map = depth_maps[view_index].reshape(-1)

        label_image[visible_index] = labels[visible_point].astype(np.uint8)
        mask_image[visible_index] = 255
        image_visibility[visible_index] = 255
        raw_depth_map[visible_index] = visible_depth.astype(np.float32)

        occupied_depth = depth_buffer[np.isfinite(depth_buffer)]
        if len(occupied_depth) > 0:
            depth_min = float(occupied_depth.min())
            depth_max = float(occupied_depth.max())
            if keep == "max":
                closeness = (
                    visible_depth - depth_min
                ) / max(depth_max - depth_min, 1e-12)
            else:
                closeness = 1.0 - (
                    visible_depth - depth_min
                ) / max(depth_max - depth_min, 1e-12)
            image_depth[visible_index] = np.clip(
                np.rint(closeness * 255.0),
                0,
                255,
            ).astype(np.uint8)

        image_longitudinal[visible_index] = np.clip(
            np.rint(normalized[visible_point, 0] * 255.0),
            0,
            255,
        ).astype(np.uint8)
        image_vertical[visible_index] = np.clip(
            np.rint(normalized[visible_point, 2] * 255.0),
            0,
            255,
        ).astype(np.uint8)

        view_indices[view_index, visible_point, 0] = v_index[visible_point]
        view_indices[view_index, visible_point, 1] = u_index[visible_point]
        point_depths[view_index, visible_point] = visible_depth.astype(
            np.float32
        )

    return (
        images,
        label_images,
        masks,
        view_indices,
        point_depths,
        depth_maps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(r"D:\Codex\datasets\BrPCD\official\prepared_20k_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"D:\Codex\datasets\BrPCD\official\projection_geom_192_v2"
        ),
    )
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ("train", "val", "test"):
        split_dir = args.prepared_root / split
        output_split = args.output_root / split
        output_split.mkdir(parents=True, exist_ok=True)
        for path in sorted(split_dir.glob("*.npz")):
            output_path = output_split / path.name
            if output_path.exists() and not args.overwrite:
                records.append(
                    {
                        "split": split,
                        "scene": path.stem,
                        "path": str(output_path),
                        "skipped": True,
                    }
                )
                print(f"skip {split}/{path.stem}", flush=True)
                continue

            scene = load_scene_npz(path)
            (
                images,
                labels,
                masks,
                view_indices,
                point_depths,
                depth_maps,
            ) = project_scene(
                scene.xyz,
                scene.labels,
                resolution=args.resolution,
            )
            np.savez_compressed(
                output_path,
                images=images,
                labels=labels,
                masks=masks,
                view_indices=view_indices,
                point_depths=point_depths,
                depth_maps=depth_maps,
                view_names=np.asarray(list(VIEWS)),
                resolution=np.asarray([args.resolution], dtype=np.int32),
            )
            records.append(
                {
                    "split": split,
                    "scene": path.stem,
                    "path": str(output_path),
                    "points": int(len(scene.xyz)),
                    "views": len(VIEWS),
                    "resolution": args.resolution,
                }
            )
            print(f"{split}/{path.stem}", flush=True)

    summary = {
        "prepared_root": str(args.prepared_root),
        "output_root": str(args.output_root),
        "resolution": args.resolution,
        "views": list(VIEWS),
        "scene_count": len(records),
        "records": records,
    }
    (args.output_root / "projection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"scene_count": len(records)}))


if __name__ == "__main__":
    main()
