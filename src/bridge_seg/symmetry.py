from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

MirrorAxis = Literal["transverse", "longitudinal", "vertical"]


@dataclass(frozen=True)
class BridgeFrame:
    """Right-handed bridge frame estimated in the scene coordinate system."""

    origin: np.ndarray
    longitudinal: np.ndarray
    transverse: np.ndarray
    vertical: np.ndarray


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length bridge axis")
    return vector / norm


def _stabilize_sign(vector: np.ndarray, centered: np.ndarray) -> np.ndarray:
    projection = centered @ vector
    skew = float(np.mean(projection**3))
    if abs(skew) <= 1e-12:
        index = int(np.argmax(np.abs(vector)))
        sign = 1.0 if vector[index] >= 0 else -1.0
    else:
        sign = 1.0 if skew >= 0 else -1.0
    return vector * sign


def estimate_bridge_frame(
    xyz: np.ndarray,
    labels: np.ndarray | None = None,
    longitudinal_class: int = 0,
) -> BridgeFrame:
    """Estimate longitudinal, transverse, and vertical axes from scene geometry.

    Girder points, when available, drive the axis estimate because concentrated
    pier points can otherwise dominate all-point PCA. The longest-variance
    girder direction is treated as longitudinal, and the smallest-variance
    direction as vertical for the shallow bridge scenes used here. The origin
    is the full-scene bounding-box center.
    """

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if len(xyz) < 3:
        raise ValueError("At least three points are required to estimate a frame")

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(xyz),):
            raise ValueError("labels must have shape [N]")
        selected = xyz[labels == longitudinal_class]
        fit_points = selected if len(selected) >= 3 else xyz
    else:
        fit_points = xyz

    origin = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    centered = fit_points - origin
    covariance = centered.T @ centered / max(len(fit_points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    longitudinal = eigenvectors[:, order[0]]
    vertical = eigenvectors[:, order[-1]]
    longitudinal = _stabilize_sign(_unit_vector(longitudinal), centered)

    transverse = np.cross(longitudinal, vertical)
    if float(np.linalg.norm(transverse)) <= 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, longitudinal))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        transverse = np.cross(longitudinal, fallback)
    transverse = _unit_vector(transverse)
    vertical = _unit_vector(np.cross(transverse, longitudinal))

    return BridgeFrame(
        origin=origin.astype(np.float64),
        longitudinal=longitudinal.astype(np.float64),
        transverse=transverse.astype(np.float64),
        vertical=vertical.astype(np.float64),
    )


def mirror_points(
    xyz: np.ndarray,
    frame: BridgeFrame,
    axis: MirrorAxis = "transverse",
) -> np.ndarray:
    """Reflect points across a frame plane through the bridge-frame origin."""

    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if axis not in ("transverse", "longitudinal", "vertical"):
        raise ValueError(f"Unsupported mirror axis: {axis}")

    normal = getattr(frame, axis)
    centered = xyz - frame.origin
    signed_distance = centered @ normal
    mirrored = centered - 2.0 * signed_distance[:, None] * normal[None, :]
    return (mirrored + frame.origin).astype(xyz.dtype, copy=False)


def _chamfer_symmetry_score(
    xyz: np.ndarray,
    frame: BridgeFrame,
    axis: MirrorAxis,
    sample_size: int,
    seed: int,
) -> float:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) > sample_size:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(xyz), sample_size, replace=False)
        xyz = xyz[indices]
    mirrored = mirror_points(xyz, frame, axis=axis)
    squared = ((xyz[:, None, :] - mirrored[None, :, :]) ** 2).sum(axis=-1)
    return float(
        0.5 * (np.sqrt(squared.min(axis=1)).mean() + np.sqrt(squared.min(axis=0)).mean())
    )


def geometry_symmetry_quality(
    xyz: np.ndarray,
    sample_size: int = 2048,
    seed: int = 0,
) -> dict[str, float | bool]:
    """Estimate transverse mirror reliability without using semantic labels."""

    xyz = np.asarray(xyz, dtype=np.float64)
    frame = estimate_bridge_frame(xyz)
    correct = _chamfer_symmetry_score(
        xyz,
        frame,
        axis="transverse",
        sample_size=sample_size,
        seed=seed,
    )
    wrong = _chamfer_symmetry_score(
        xyz,
        frame,
        axis="longitudinal",
        sample_size=sample_size,
        seed=seed,
    )
    evidence = max(0.0, wrong - correct) / max(wrong + correct, 1e-12)
    return {
        "correct_score": correct,
        "wrong_score": wrong,
        "wrong_to_correct_ratio": float(wrong / max(correct, 1e-12)),
        "evidence": float(evidence),
        "eligible": bool(wrong > correct),
    }


def symmetry_axis_quality(
    xyz: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 512,
    seed: int = 0,
) -> dict[str, float | bool]:
    frame = estimate_bridge_frame(xyz, labels)
    correct = _chamfer_symmetry_score(
        xyz,
        frame,
        axis="transverse",
        sample_size=sample_size,
        seed=seed,
    )
    wrong = _chamfer_symmetry_score(
        xyz,
        frame,
        axis="longitudinal",
        sample_size=sample_size,
        seed=seed,
    )
    ratio = wrong / max(correct, 1e-12)
    return {
        "correct_score": correct,
        "wrong_score": wrong,
        "wrong_to_correct_ratio": float(ratio),
        "eligible": bool(wrong > correct),
    }
