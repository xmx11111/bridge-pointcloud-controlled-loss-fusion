from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bridge_seg.data import load_scene_npz
from bridge_seg.symmetry import estimate_bridge_frame


VIEW_DIRECTIONS = {
    "top": (0.0, 0.0, 1.0),
    "side": (0.0, 1.0, 0.0),
    "end": (1.0, 0.0, 0.0),
    "side_left": (-0.12, 0.99, 0.0),
    "side_right": (0.12, 0.99, 0.0),
}


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def _basis(
    direction: np.ndarray,
    vertical_axis: np.ndarray,
    longitudinal_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = _unit(direction)
    forward = -direction
    up_hint = vertical_axis
    if abs(float(np.dot(forward, up_hint))) > 0.9:
        up_hint = longitudinal_axis
    right = np.cross(forward, up_hint)
    if float(np.linalg.norm(right)) <= 1e-8:
        fallback = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(forward, fallback))) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, fallback)
    right = _unit(right)
    up = _unit(np.cross(right, forward))
    return right, up, direction


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
    """Render five orthographic virtual-camera views with a Z-buffer."""

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
    normalized = (coordinates - lower[None, :]) / extent[None, :]
    scene_scale = float(np.max(extent))

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
    view_directions = np.zeros((view_count, 3), dtype=np.float64)

    for view_index, (name, direction_values) in enumerate(
        VIEW_DIRECTIONS.items()
    ):
        direction = _unit(direction_values)
        right, up, camera_direction = _basis(
            direction,
            axes[:, 2],
            axes[:, 0],
        )
        u_values = coordinates @ right
        v_values = coordinates @ up
        depth = coordinates @ camera_direction
        u_index = np.clip(
            np.floor(_normalize(u_values) * (resolution - 1)).astype(
                np.int64
            ),
            0,
            resolution - 1,
        )
        v_index = np.clip(
            np.floor(_normalize(v_values) * (resolution - 1)).astype(
                np.int64
            ),
            0,
            resolution - 1,
        )
        base_flat = v_index * resolution + u_index

        splat_index_chunks: list[np.ndarray] = []
        splat_point_chunks: list[np.ndarray] = []
        splat_depth_chunks: list[np.ndarray] = []
        for delta_v in range(-point_radius, point_radius + 1):
            for delta_u in range(-point_radius, point_radius + 1):
                candidate_u = u_index + delta_u
                candidate_v = v_index + delta_v
                keep = (
                    (candidate_u >= 0)
                    & (candidate_u < resolution)
                    & (candidate_v >= 0)
                    & (candidate_v < resolution)
                )
                splat_index_chunks.append(
                    candidate_v[keep] * resolution + candidate_u[keep]
                )
                splat_point_chunks.append(np.flatnonzero(keep))
                splat_depth_chunks.append(depth[keep])

        splat_index = np.concatenate(splat_index_chunks)
        splat_point = np.concatenate(splat_point_chunks)
        splat_depth = np.concatenate(splat_depth_chunks)
        depth_buffer = np.full(
            resolution * resolution,
            -np.inf,
            dtype=np.float64,
        )
        np.maximum.at(depth_buffer, splat_index, splat_depth)
        tolerance = max(scene_scale * 1e-6, 1e-9)
        visible = np.isclose(
            splat_depth,
            depth_buffer[splat_index],
            rtol=0.0,
            atol=tolerance,
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

        occupied_depth = depth_buffer[np.isfinite(depth_buffer)]
        if len(occupied_depth) > 0:
            nearest = float(np.max(occupied_depth))
            farthest = float(np.min(occupied_depth))
            closeness = 1.0 - (
                nearest - visible_depth
            ) / max(nearest - farthest, 1e-12)
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
        view_directions[view_index] = camera_direction

    return (
        images,
        label_images,
        masks,
        view_indices,
        point_depths,
        depth_maps,
        view_directions,
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
            r"D:\Codex\datasets\BrPCD\official\projection_geom_384_v3"
        ),
    )
    parser.add_argument("--resolution", type=int, default=384)
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
                view_directions,
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
                view_directions=view_directions,
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
        "view_directions": {
            name: list(direction)
            for name, direction in VIEW_DIRECTIONS.items()
        },
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
