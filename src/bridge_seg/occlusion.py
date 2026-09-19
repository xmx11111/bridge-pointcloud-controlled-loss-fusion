from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OCCLUSION_STRATEGIES = ("random", "viewpoint")


@dataclass(frozen=True)
class OcclusionResult:
    xyz: np.ndarray
    rgb: np.ndarray
    labels: np.ndarray
    kept_indices: np.ndarray
    metadata: dict[str, object]


def _validate_inputs(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    missing_rate: float,
    strategy: str,
) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if rgb.shape != xyz.shape:
        raise ValueError("rgb must have the same shape as xyz")
    if labels.shape != (len(xyz),):
        raise ValueError("labels must have shape [N]")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    if strategy not in OCCLUSION_STRATEGIES:
        raise ValueError(f"Unsupported occlusion strategy: {strategy}")


def _viewpoint_risk(xyz: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    norm = np.linalg.norm(direction)
    if norm <= 0:
        direction = np.asarray([1.0, 0.0, 0.0])
    else:
        direction = direction / norm

    reference = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference, direction))) > 0.9:
        reference = np.asarray([1.0, 0.0, 0.0])
    axis_u = np.cross(direction, reference)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(direction, axis_u)

    depth = xyz @ direction
    transverse = np.sqrt((xyz @ axis_u) ** 2 + (xyz @ axis_v) ** 2)
    depth_rank = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-12)
    transverse_rank = transverse / max(float(transverse.max()), 1e-12)
    return depth_rank + 0.25 * transverse_rank


def occlude_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    missing_rate: float,
    strategy: str,
    seed: int,
) -> OcclusionResult:
    """Remove an exact number of points using random or depth-proxy occlusion."""

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    _validate_inputs(xyz, rgb, labels, missing_rate, strategy)

    point_count = len(xyz)
    remove_count = int(round(point_count * missing_rate))
    remove_count = min(remove_count, max(point_count - 1, 0))
    keep_count = point_count - remove_count
    if remove_count == 0:
        kept_indices = np.arange(point_count, dtype=np.int64)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        removed = rng.choice(point_count, remove_count, replace=False)
        kept_indices = np.setdiff1d(
            np.arange(point_count, dtype=np.int64),
            removed,
            assume_unique=False,
        )
    else:
        risk = _viewpoint_risk(xyz, seed)
        removed = np.argpartition(risk, -remove_count)[-remove_count:]
        kept_indices = np.setdiff1d(
            np.arange(point_count, dtype=np.int64),
            removed,
            assume_unique=False,
        )

    return OcclusionResult(
        xyz=xyz[kept_indices],
        rgb=rgb[kept_indices],
        labels=labels[kept_indices],
        kept_indices=kept_indices,
        metadata={
            "strategy": strategy,
            "requested_missing_rate": float(missing_rate),
            "point_count": int(point_count),
            "kept_count": int(keep_count),
            "removed_count": int(remove_count),
            "actual_missing_rate": float(remove_count / max(point_count, 1)),
        },
    )


def occlude_feature_block(
    features: np.ndarray,
    labels: np.ndarray,
    missing_rate: float,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply occlusion to a feature block whose first three columns are XYZ."""

    xyz = features[:, :3]
    result = occlude_points(
        xyz,
        xyz,
        labels,
        missing_rate=missing_rate,
        strategy=strategy,
        seed=seed,
    )
    return features[result.kept_indices], result.labels, result.metadata
