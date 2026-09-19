from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .classes import CLASS_NAMES, bridge_family_from_scene, source_type_from_scene
from .occlusion import occlude_points
from .symmetry import BridgeFrame, estimate_bridge_frame, mirror_points

SAMPLING_MODES = ("uniform", "class_balanced")
SYMMETRY_MODES = (
    "none",
    "augmentation",
    "consistency",
    "wrong_axis",
    "cross_side",
)


@dataclass(frozen=True)
class SceneData:
    name: str
    path: Path
    xyz: np.ndarray
    rgb: np.ndarray
    labels: np.ndarray
    center: np.ndarray
    extent: np.ndarray
    scale: float
    source_type: str
    bridge_family: str


def _required_array(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"Prepared scene is missing '{key}'")
    return np.asarray(data[key])


def load_scene_npz(path: Path) -> SceneData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        xyz = _required_array(data, "xyz").astype(np.float32, copy=False)
        rgb = _required_array(data, "rgb").astype(np.float32, copy=False)
        labels = _required_array(data, "labels").astype(np.int64, copy=False)
        center = _required_array(data, "center").astype(np.float32, copy=False)
        extent = _required_array(data, "extent").astype(np.float32, copy=False)
        scale_array = _required_array(data, "scale").astype(np.float32, copy=False)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{path}: xyz must have shape [N, 3]")
    if rgb.shape != xyz.shape:
        raise ValueError(f"{path}: rgb must have the same shape as xyz")
    if labels.shape != (len(xyz),):
        raise ValueError(f"{path}: labels must have shape [N]")
    if center.shape != (3,) or extent.shape != (3,) or scale_array.size != 1:
        raise ValueError(f"{path}: invalid center, extent, or scale metadata")
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError(f"{path}: labels contain values outside the class map")

    scene_name = path.stem
    return SceneData(
        name=scene_name,
        path=path,
        xyz=xyz,
        rgb=rgb,
        labels=labels,
        center=center,
        extent=extent,
        scale=float(scale_array[0]),
        source_type=source_type_from_scene(scene_name),
        bridge_family=bridge_family_from_scene(scene_name),
    )


def discover_prepared_splits(prepared_root: Path) -> dict[str, list[Path]]:
    prepared_root = Path(prepared_root)
    splits: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        paths = sorted((prepared_root / split).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No prepared NPZ files found for split '{split}'")
        splits[split] = paths
    return splits


def collect_split_statistics(paths: list[Path]) -> dict[str, object]:
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    source_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    point_count = 0

    for path in paths:
        scene = load_scene_npz(path)
        point_count += len(scene.labels)
        class_counts += np.bincount(scene.labels, minlength=len(CLASS_NAMES))
        source_counts[scene.source_type] = source_counts.get(scene.source_type, 0) + 1
        family_counts[scene.bridge_family] = family_counts.get(scene.bridge_family, 0) + 1

    return {
        "scene_count": len(paths),
        "point_count": point_count,
        "class_point_counts": {
            name: int(class_counts[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "source_scene_counts": dict(sorted(source_counts.items())),
        "bridge_family_scene_counts": dict(sorted(family_counts.items())),
        "scenes": sorted(path.stem for path in paths),
    }


def audit_prepared_dataset(prepared_root: Path) -> dict[str, object]:
    splits = discover_prepared_splits(prepared_root)
    return {
        "root": str(Path(prepared_root)),
        "class_names": list(CLASS_NAMES),
        "splits": {
            split: collect_split_statistics(paths)
            for split, paths in splits.items()
        },
    }


def class_balanced_probabilities(
    labels: np.ndarray,
    num_classes: int = len(CLASS_NAMES),
) -> np.ndarray:
    """Return inverse-square-root class probabilities for the observed classes."""

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    if not np.any(present):
        raise ValueError("Cannot balance an empty label array")
    probabilities = np.zeros(num_classes, dtype=np.float64)
    inverse = 1.0 / np.sqrt(counts[present])
    probabilities[present] = inverse / inverse.sum()
    return probabilities


def sample_block(
    scene: SceneData,
    points_per_block: int,
    rng: np.random.Generator,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb, labels = sample_raw_block(
        scene,
        points_per_block=points_per_block,
        rng=rng,
        sampling_mode=sampling_mode,
    )
    features = build_block_features(xyz, rgb, scene)
    return features, labels.astype(np.int64)


def sample_raw_block(
    scene: SceneData,
    points_per_block: int,
    rng: np.random.Generator,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_per_block <= 0:
        raise ValueError("points_per_block must be positive")
    if len(scene.xyz) == 0:
        raise ValueError("Cannot sample from an empty scene")
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    if sampling_mode == "class_balanced":
        probabilities = class_balanced_probabilities(scene.labels)
        seeded_class = int(rng.choice(len(CLASS_NAMES), p=probabilities))
        seed_index = int(rng.choice(np.flatnonzero(scene.labels == seeded_class)))
    else:
        seed_index = int(rng.integers(0, len(scene.xyz)))

    distances = np.sum((scene.xyz - scene.xyz[seed_index]) ** 2, axis=1)
    count = min(points_per_block, len(scene.xyz))
    indices = np.argpartition(distances, count - 1)[:count]
    rng.shuffle(indices)

    xyz = scene.xyz[indices]
    rgb = scene.rgb[indices]
    labels = scene.labels[indices]
    return xyz, rgb, labels


def build_block_features(
    xyz: np.ndarray,
    rgb: np.ndarray,
    scene: SceneData,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if rgb.shape != xyz.shape:
        raise ValueError("rgb must have the same shape as xyz")

    local_center = xyz.mean(axis=0, keepdims=True)
    local_extent = np.ptp(xyz, axis=0)
    local_scale = float(np.max(local_extent))
    if local_scale <= 0:
        local_scale = 1.0

    local_xyz = (xyz - local_center) / local_scale
    global_xyz = (xyz - scene.center) / max(scene.scale, 1e-12)
    features = np.concatenate([local_xyz, global_xyz, rgb], axis=1).astype(np.float32)
    return features


def sample_symmetry_block(
    scene: SceneData,
    points_per_block: int,
    rng: np.random.Generator,
    sampling_mode: str,
    symmetry_mode: str,
    frame: BridgeFrame | None = None,
    occlusion_rate: float = 0.0,
    occlusion_strategy: str = "random",
    occlusion_seed: int = 0,
    symmetry_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an original block and its mirrored counterpart in point order."""

    if symmetry_mode not in SYMMETRY_MODES or symmetry_mode == "none":
        raise ValueError("symmetry_mode must enable a mirrored block")
    xyz, rgb, labels = sample_raw_block(
        scene,
        points_per_block=points_per_block,
        rng=rng,
        sampling_mode=sampling_mode,
    )
    frame = frame or estimate_bridge_frame(scene.xyz, scene.labels)
    if symmetry_mode == "cross_side":
        mirrored_xyz_clean = mirror_points(xyz, frame, axis="transverse")
        if occlusion_rate > 0:
            result = occlude_points(
                xyz,
                rgb,
                labels,
                missing_rate=occlusion_rate,
                strategy=occlusion_strategy,
                seed=occlusion_seed,
            )
            xyz, rgb, labels = result.xyz, result.rgb, result.labels
            mirrored_xyz = mirrored_xyz_clean[result.kept_indices]
        else:
            mirrored_xyz = mirrored_xyz_clean
        features = build_block_features(xyz, rgb, scene)
        mirrored_features = build_block_features(mirrored_xyz, rgb, scene)
        return features, labels.astype(np.int64), mirrored_features

    if occlusion_rate > 0:
        result = occlude_points(
            xyz,
            rgb,
            labels,
            missing_rate=occlusion_rate,
            strategy=occlusion_strategy,
            seed=occlusion_seed,
        )
        xyz, rgb, labels = result.xyz, result.rgb, result.labels

    features = build_block_features(xyz, rgb, scene)
    if not symmetry_enabled:
        return features, labels.astype(np.int64), features

    axis = "longitudinal" if symmetry_mode == "wrong_axis" else "transverse"
    mirrored_xyz = mirror_points(xyz, frame, axis=axis)
    mirrored_features = build_block_features(mirrored_xyz, rgb, scene)
    return features, labels.astype(np.int64), mirrored_features
