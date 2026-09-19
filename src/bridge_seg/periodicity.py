from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .symmetry import BridgeFrame


@dataclass(frozen=True)
class PeriodicityResult:
    period_length: float
    confidence: float
    best_lag_bins: int
    repeats: float
    best_similarity: float
    baseline_similarity: float
    axis_length: float


def longitudinal_cross_section_descriptors(
    xyz: np.ndarray,
    frame: BridgeFrame,
    longitudinal_bins: int = 96,
    cross_section_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build geometry-only cross-section descriptors along the bridge axis."""

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if longitudinal_bins < 8 or cross_section_bins < 2:
        raise ValueError("descriptor grid is too small")

    basis = np.stack(
        [frame.longitudinal, frame.transverse, frame.vertical],
        axis=1,
    )
    canonical = (xyz - frame.origin) @ basis
    longitudinal = canonical[:, 0]
    lower, upper = np.percentile(longitudinal, [1.0, 99.0])
    axis_length = float(upper - lower)
    if axis_length <= 1e-8:
        raise ValueError("bridge longitudinal extent is too small")

    transverse_edges = np.linspace(
        np.percentile(canonical[:, 1], 1.0),
        np.percentile(canonical[:, 1], 99.0),
        cross_section_bins + 1,
    )
    vertical_edges = np.linspace(
        np.percentile(canonical[:, 2], 1.0),
        np.percentile(canonical[:, 2], 99.0),
        cross_section_bins + 1,
    )
    if np.any(np.diff(transverse_edges) <= 0) or np.any(
        np.diff(vertical_edges) <= 0
    ):
        raise ValueError("cross-section extent is degenerate")

    longitudinal_indices = np.clip(
        np.floor(
            (longitudinal - lower)
            / axis_length
            * longitudinal_bins
        ).astype(np.int64),
        0,
        longitudinal_bins - 1,
    )
    descriptors = np.zeros(
        (longitudinal_bins, cross_section_bins * cross_section_bins),
        dtype=np.float64,
    )
    for index in range(longitudinal_bins):
        selected = canonical[longitudinal_indices == index]
        if not len(selected):
            continue
        histogram, _, _ = np.histogram2d(
            selected[:, 1],
            selected[:, 2],
            bins=(transverse_edges, vertical_edges),
        )
        descriptor = histogram.reshape(-1)
        norm = float(np.linalg.norm(descriptor))
        if norm > 0:
            descriptors[index] = descriptor / norm
    return descriptors, np.asarray([lower, upper], dtype=np.float64)


def _lag_similarities(
    descriptors: np.ndarray,
    maximum_lag: int,
) -> np.ndarray:
    similarities = np.full(maximum_lag + 1, np.nan, dtype=np.float64)
    norms = np.linalg.norm(descriptors, axis=1)
    for lag in range(1, maximum_lag + 1):
        left = descriptors[:-lag]
        right = descriptors[lag:]
        valid = (norms[:-lag] > 0) & (norms[lag:] > 0)
        if valid.sum() < 4:
            continue
        numerator = np.einsum("ij,ij->i", left[valid], right[valid])
        denominator = norms[:-lag][valid] * norms[lag:][valid]
        similarities[lag] = float(np.mean(numerator / denominator))
    return similarities


def estimate_periodicity(
    xyz: np.ndarray,
    frame: BridgeFrame,
    longitudinal_bins: int = 96,
    cross_section_bins: int = 8,
    min_period_fraction: float = 0.10,
    max_period_fraction: float = 0.50,
) -> PeriodicityResult:
    """Estimate a repeated longitudinal unit without using semantic labels."""

    if not 0.0 < min_period_fraction < max_period_fraction < 1.0:
        raise ValueError("period fractions must satisfy 0 < min < max < 1")
    descriptors, extent = longitudinal_cross_section_descriptors(
        xyz,
        frame,
        longitudinal_bins=longitudinal_bins,
        cross_section_bins=cross_section_bins,
    )
    axis_length = float(extent[1] - extent[0])
    minimum_lag = max(1, int(round(longitudinal_bins * min_period_fraction)))
    maximum_lag = max(
        minimum_lag + 1,
        int(round(longitudinal_bins * max_period_fraction)),
    )
    similarities = _lag_similarities(descriptors, maximum_lag)
    candidates = similarities[minimum_lag : maximum_lag + 1]
    finite = candidates[np.isfinite(candidates)]
    if len(finite) < 3:
        return PeriodicityResult(
            period_length=0.0,
            confidence=0.0,
            best_lag_bins=0,
            repeats=0.0,
            best_similarity=0.0,
            baseline_similarity=0.0,
            axis_length=axis_length,
        )

    local_best = int(np.nanargmax(candidates)) + minimum_lag
    best_similarity = float(similarities[local_best])
    baseline = float(np.nanmedian(candidates))
    confidence = float(
        np.clip(
            (best_similarity - baseline) / max(1.0 - baseline, 1e-8),
            0.0,
            1.0,
        )
    )
    period_length = axis_length * local_best / longitudinal_bins
    repeats = axis_length / max(period_length, 1e-12)
    return PeriodicityResult(
        period_length=float(period_length),
        confidence=confidence,
        best_lag_bins=local_best,
        repeats=float(repeats),
        best_similarity=best_similarity,
        baseline_similarity=baseline,
        axis_length=axis_length,
    )
