from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bridge_seg.data import load_scene_npz
from bridge_seg.symmetry import estimate_bridge_frame


VIEW_DIRECTIONS = {
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "side_pos": (0.0, 1.0, 0.0),
    "side_neg": (0.0, -1.0, 0.0),
    "end_pos": (1.0, 0.0, 0.0),
    "end_neg": (-1.0, 0.0, 0.0),
    "oblique_pos": (0.7, 0.7, 0.55),
    "oblique_neg": (-0.7, -0.7, 0.55),
}


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _camera_basis(
    direction: np.ndarray,
    vertical: np.ndarray,
    longitudinal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, and forward axes in bridge coordinates."""

    forward = -_unit(direction)
    up_hint = vertical
    if abs(float(np.dot(forward, up_hint))) > 0.9:
        up_hint = longitudinal
    right = np.cross(forward, up_hint)
    if float(np.linalg.norm(right)) <= 1e-8:
        fallback = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(forward, fallback))) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, fallback)
    right = _unit(right)
    up = _unit(np.cross(right, forward))
    return right, up, forward


def project_scene(
    xyz: np.ndarray,
    labels: np.ndarray,
    resolution: int,
    point_radius: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Render label-free geometry views with a Z-buffer.

    BrPCD stores RGB as ``255 0 0`` for every point. Feeding those constant
    colors to a 2D network only duplicates the occupancy mask. This renderer
    keeps the no-registration property of virtual cameras while replacing RGB
    with four geometry channels: depth closeness, normalized longitudinal
    position, normalized vertical position, and visibility.
    """

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if labels.shape != (len(xyz),):
        raise ValueError("labels must have shape [N]")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if point_radius < 0:
        raise ValueError("point_radius must be non-negative")

    frame = estimate_bridge_frame(xyz)
    axes = np.stack(
        [frame.longitudinal, frame.transverse, frame.vertical],
        axis=1,
    )
    coordinates = (xyz - frame.origin) @ axes
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    extent = np.maximum(upper - lower, 1e-12)
    scene_scale = float(np.max(extent))

    normalized_longitudinal = (
        coordinates[:, 0] - lower[0]
    ) / extent[0]
    normalized_vertical = (
        coordinates[:, 2] - lower[2]
    ) / extent[2]
    normalized_transverse = (
        coordinates[:, 1] - lower[1]
    ) / extent[1]
    longitudinal_axis = axes[:, 0]
    vertical_axis = axes[:, 2]

    view_names = list(VIEW_DIRECTIONS)
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
    camera_matrices = np.zeros((view_count, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((view_count, 3, 3), dtype=np.float64)

    distance = scene_scale * 1.8
    half_fov = np.deg2rad(27.5)
    focal = 0.5 * resolution / np.tan(half_fov)
    near = scene_scale * 0.05

    for view_index, (name, direction_values) in enumerate(
        VIEW_DIRECTIONS.items()
    ):
        direction = _unit(np.asarray(direction_values, dtype=np.float64))
        position = direction * distance
        right, up, forward = _camera_basis(
            direction,
            vertical_axis,
            longitudinal_axis,
        )
        camera_axes = np.stack([right, up, forward], axis=1)
        camera_points = (coordinates - position[None, :]) @ camera_axes
        depth = camera_points[:, 2]
        valid = depth > near

        horizontal = np.full(len(xyz), -1, dtype=np.int64)
        vertical = np.full(len(xyz), -1, dtype=np.int64)
        horizontal[valid] = np.rint(
            resolution * 0.5
            + focal * camera_points[valid, 0] / depth[valid]
        ).astype(np.int64)
        vertical[valid] = np.rint(
            resolution * 0.5
            - focal * camera_points[valid, 1] / depth[valid]
        ).astype(np.int64)
        in_frame = (
            valid
            & (horizontal >= 0)
            & (horizontal < resolution)
            & (vertical >= 0)
            & (vertical < resolution)
        )
        valid_indices = np.flatnonzero(in_frame)
        if len(valid_indices) == 0:
            continue

        horizontal = horizontal[in_frame]
        vertical = vertical[in_frame]
        point_depth = depth[in_frame]
        flat_index = vertical * resolution + horizontal
        if point_radius > 0:
            neighbors = []
            for delta_vertical in range(
                -point_radius,
                point_radius + 1,
            ):
                for delta_horizontal in range(
                    -point_radius,
                    point_radius + 1,
                ):
                    candidate_horizontal = horizontal + delta_horizontal
                    candidate_vertical = vertical + delta_vertical
                    keep = (
                        (candidate_horizontal >= 0)
                        & (candidate_horizontal < resolution)
                        & (candidate_vertical >= 0)
                        & (candidate_vertical < resolution)
                    )
                    neighbors.append(
                        (
                            candidate_vertical[keep] * resolution
                            + candidate_horizontal[keep],
                            valid_indices[keep],
                            point_depth[keep],
                        )
                    )
            splat_index = np.concatenate([item[0] for item in neighbors])
            splat_point = np.concatenate([item[1] for item in neighbors])
            splat_depth = np.concatenate([item[2] for item in neighbors])
        else:
            splat_index = flat_index
            splat_point = valid_indices
            splat_depth = point_depth

        depth_buffer = np.full(
            resolution * resolution,
            np.inf,
            dtype=np.float64,
        )
        np.minimum.at(depth_buffer, splat_index, splat_depth)
        visible = np.isclose(
            splat_depth,
            depth_buffer[splat_index],
            rtol=0.0,
            atol=1e-6 * max(scene_scale, 1.0),
        )
        visible_index = splat_index[visible]
        visible_point = splat_point[visible]
        visible_depth = splat_depth[visible]

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

        occupied_depth = depth_buffer[depth_buffer < np.inf]
        if len(occupied_depth) > 0:
            nearest = float(occupied_depth.min())
            farthest = float(occupied_depth.max())
            closeness = 1.0 - (
                (visible_depth - nearest)
                / max(farthest - nearest, 1e-12)
            )
            image_depth[visible_index] = np.clip(
                np.rint(closeness * 255.0),
                0,
                255,
            ).astype(np.uint8)
        image_longitudinal[visible_index] = np.clip(
            np.rint(
                normalized_longitudinal[visible_point] * 255.0
            ),
            0,
            255,
        ).astype(np.uint8)
        image_vertical[visible_index] = np.clip(
            np.rint(normalized_vertical[visible_point] * 255.0),
            0,
            255,
        ).astype(np.uint8)

        view_indices[view_index, visible_point, 0] = vertical[
            np.searchsorted(valid_indices, visible_point)
        ]
        view_indices[view_index, visible_point, 1] = horizontal[
            np.searchsorted(valid_indices, visible_point)
        ]
        point_depths[view_index, visible_point] = visible_depth.astype(
            np.float32
        )

        camera_matrices[view_index, :, :3] = camera_axes.T
        camera_matrices[view_index, :, 3] = (
            -camera_axes.T @ position
        )
        intrinsics[view_index] = np.asarray(
            [
                [focal, 0.0, resolution * 0.5],
                [0.0, focal, resolution * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    return (
        images,
        label_images,
        masks,
        view_indices,
        point_depths,
        depth_maps,
        camera_matrices,
        intrinsics,
        normalized_transverse,
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
    parser.add_argument("--point-radius", type=int, default=1)
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
                camera_matrices,
                intrinsics,
                _,
            ) = project_scene(
                scene.xyz,
                scene.labels,
                resolution=args.resolution,
                point_radius=args.point_radius,
            )
            np.savez_compressed(
                output_path,
                images=images,
                labels=labels,
                masks=masks,
                view_indices=view_indices,
                point_depths=point_depths,
                depth_maps=depth_maps,
                camera_matrices=camera_matrices,
                intrinsics=intrinsics,
                view_names=np.asarray(list(VIEW_DIRECTIONS)),
                resolution=np.asarray([args.resolution], dtype=np.int32),
            )
            records.append(
                {
                    "split": split,
                    "scene": path.stem,
                    "path": str(output_path),
                    "points": int(len(scene.xyz)),
                    "views": len(VIEW_DIRECTIONS),
                    "resolution": args.resolution,
                }
            )
            print(f"{split}/{path.stem}", flush=True)

    summary = {
        "prepared_root": str(args.prepared_root),
        "output_root": str(args.output_root),
        "resolution": args.resolution,
        "point_radius": args.point_radius,
        "views": list(VIEW_DIRECTIONS),
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
