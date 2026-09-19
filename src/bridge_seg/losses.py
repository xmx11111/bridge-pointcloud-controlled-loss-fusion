from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .classes import CLASS_NAMES

LOSS_NAMES = (
    "ce_unweighted",
    "ce_weighted",
    "focal_weighted",
    "ohem_weighted",
)


def inverse_frequency_weights(
    labels: np.ndarray,
    num_classes: int = len(CLASS_NAMES),
    minimum: float = 0.5,
    maximum: float = 5.0,
) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Cannot compute class weights; missing classes: {missing}")
    weights = np.sqrt(counts.sum() / (counts * num_classes))
    return torch.tensor(np.clip(weights, minimum, maximum), dtype=torch.float32)


class WeightedFocalLoss(nn.Module):
    """Class-weighted focal cross entropy for point-level segmentation."""

    def __init__(
        self,
        weights: torch.Tensor,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.register_buffer("weights", weights.detach().clone())
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_point = F.cross_entropy(
            logits.reshape(logits.shape[0], logits.shape[1], -1)
            .transpose(1, 2)
            .reshape(-1, logits.shape[1]),
            targets.reshape(-1),
            weight=self.weights.to(logits.device),
            reduction="none",
        )
        loss = (1.0 - torch.exp(-per_point)).pow(self.gamma) * per_point
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class StrongSupervisionLoss(nn.Module):
    """Batch-frequency weighted cross entropy with OHEM filtering.

    Two components of the strong-supervision recipe:

    1. class weights are recomputed from the valid points of the current
       batch as inverse frequencies, so neither the number of classes nor
       a weight array has to be hard-coded;
    2. only the hardest ``ohem_ratio`` fraction of valid points contributes
       to the gradient, with an automatic fallback to all valid points when
       the batch is too small to filter safely.

    The geometric weak-supervision term of the arch-bridge variant is not
    part of this module: it is defined on an arch-rib class that does not
    exist in the girder/pier/deck label set. The structural prior that is
    defined here lives in :func:`topology_order_loss`.
    """

    def __init__(
        self,
        ohem_ratio: float = 0.2,
        min_ohem_points: int = 100,
        ignore_index: int = -100,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 < ohem_ratio <= 1.0:
            raise ValueError("ohem_ratio must be in (0, 1]")
        if min_ohem_points < 0:
            raise ValueError("min_ohem_points must be non-negative")
        self.ohem_ratio = float(ohem_ratio)
        self.min_ohem_points = int(min_ohem_points)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def _batch_class_weights(
        self,
        targets: torch.Tensor,
        num_classes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        counts = torch.bincount(targets, minlength=num_classes).to(dtype=dtype)
        weights = 1.0 / (counts + self.eps)
        present = counts > 0
        if bool(present.any()):
            weights = weights / (weights[present].mean() + self.eps)
        else:
            weights = torch.zeros_like(weights)
        return torch.where(present, weights, torch.zeros_like(weights))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [N, num_classes]")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [N]")
        if logits.shape[0] == 0:
            return logits.sum() * 0.0

        targets = targets.long()
        valid = targets != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0
        valid_logits = logits[valid]
        valid_targets = targets[valid]
        num_classes = int(logits.shape[1])
        if (
            int(valid_targets.min()) < 0
            or int(valid_targets.max()) >= num_classes
        ):
            raise ValueError(
                "targets contain class indices outside [0, num_classes)"
            )

        weights = self._batch_class_weights(
            valid_targets,
            num_classes,
            logits.dtype,
        )
        per_point = F.cross_entropy(
            valid_logits,
            valid_targets,
            weight=weights,
            reduction="none",
        )
        point_count = int(per_point.numel())
        if point_count <= self.min_ohem_points or self.ohem_ratio >= 1.0:
            return per_point.mean()
        keep = max(1, int(point_count * self.ohem_ratio))
        return torch.topk(
            per_point,
            k=min(keep, point_count),
            largest=True,
            sorted=False,
        ).values.mean()


def build_segmentation_loss(
    name: str,
    class_weights: torch.Tensor,
    focal_gamma: float = 2.0,
    ohem_ratio: float = 0.2,
    ohem_min_points: int = 100,
) -> nn.Module:
    if name == "ce_unweighted":
        return nn.CrossEntropyLoss()
    if name == "ce_weighted":
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal_weighted":
        return WeightedFocalLoss(class_weights, gamma=focal_gamma)
    if name == "ohem_weighted":
        return StrongSupervisionLoss(
            ohem_ratio=ohem_ratio,
            min_ohem_points=ohem_min_points,
        )
    raise ValueError(f"Unsupported loss: {name}")


def symmetry_consistency_loss(
    logits: torch.Tensor,
    mirrored_logits: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared difference between original and mirrored class probabilities."""

    if logits.shape != mirrored_logits.shape:
        raise ValueError("logits and mirrored_logits must have the same shape")
    if logits.ndim < 2:
        raise ValueError("logits must include a class dimension")
    probabilities = torch.softmax(logits, dim=-1)
    mirrored_probabilities = torch.softmax(mirrored_logits, dim=-1)
    per_point = (probabilities - mirrored_probabilities).pow(2)
    if sample_weights is None:
        return per_point.mean()
    if per_point.ndim < 2:
        raise ValueError("weighted symmetry loss requires a batch dimension")
    per_sample = per_point.flatten(start_dim=1).mean(dim=1)
    sample_weights = sample_weights.reshape(-1).to(per_sample.device)
    if sample_weights.shape != per_sample.shape:
        raise ValueError("sample_weights must match the batch dimension")
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def symmetry_weight_for_epoch(
    base_weight: float,
    epoch: int,
    warmup_epochs: int,
) -> float:
    if base_weight < 0:
        raise ValueError("base_weight must be non-negative")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if warmup_epochs <= 0:
        return float(base_weight)
    return float(base_weight) * min(1.0, epoch / float(warmup_epochs))


def topology_order_loss(
    logits: torch.Tensor,
    vertical: torch.Tensor,
    margin: float = 0.01,
    mass_weighting: bool = False,
    min_class_mass: float = 0.02,
) -> torch.Tensor:
    """Encourage deck above girder above pier using soft class centroids."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, C, N]")
    if vertical.ndim != 2 or vertical.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("vertical must have shape [B, N]")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if not 0.0 <= min_class_mass < 1.0:
        raise ValueError("min_class_mass must be in [0, 1)")
    probabilities = torch.softmax(logits, dim=1)
    mass = probabilities.sum(dim=2, keepdim=True).clamp_min(1e-6)
    centroids = (
        (probabilities * vertical[:, None, :]).sum(dim=2, keepdim=True) / mass
    ).squeeze(2)
    pier_gap = centroids[:, 1] - centroids[:, 0]
    deck_gap = centroids[:, 2] - centroids[:, 1]
    pier_loss = torch.relu(margin - pier_gap)
    deck_loss = torch.relu(margin - deck_gap)
    if not mass_weighting:
        return pier_loss.mean() + deck_loss.mean()

    mass_fraction = (mass.squeeze(2) / logits.shape[2]).clamp(0.0, 1.0)
    valid = mass_fraction >= min_class_mass
    pair_pier = valid[:, 0] & valid[:, 1]
    pair_deck = valid[:, 1] & valid[:, 2]
    pier_weight = mass_fraction[:, 0] * mass_fraction[:, 1] * pair_pier
    deck_weight = mass_fraction[:, 1] * mass_fraction[:, 2] * pair_deck
    denominator = (pier_weight + deck_weight).sum().clamp_min(1.0)
    return (pier_loss * pier_weight + deck_loss * deck_weight).sum() / denominator
