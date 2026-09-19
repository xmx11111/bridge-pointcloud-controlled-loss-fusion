import numpy as np
import torch

from bridge_seg.losses import (
    StrongSupervisionLoss,
    WeightedFocalLoss,
    build_segmentation_loss,
    inverse_frequency_weights,
    symmetry_consistency_loss,
    symmetry_weight_for_epoch,
    topology_order_loss,
)


def test_inverse_frequency_weights_favor_rare_classes() -> None:
    weights = inverse_frequency_weights(
        np.asarray([0] * 100 + [1] * 10 + [2])
    )

    assert weights[2] > weights[1] > weights[0]


def test_weighted_focal_loss_is_finite() -> None:
    logits = torch.randn(2, 3, 8)
    targets = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]] * 2)
    loss = WeightedFocalLoss(torch.tensor([1.0, 2.0, 3.0]), gamma=2.0)

    value = loss(logits, targets)

    assert torch.isfinite(value)
    assert value.item() > 0


def test_symmetry_consistency_loss_is_zero_for_equal_logits() -> None:
    logits = torch.randn(2, 8, 3)

    assert symmetry_consistency_loss(logits, logits).item() == 0.0
    assert symmetry_consistency_loss(logits, logits + 0.5).item() > 0.0

    weights = torch.tensor([0.0, 1.0])
    assert symmetry_consistency_loss(logits, logits + 0.5, weights).item() >= 0.0


def test_symmetry_weight_warms_up_linearly() -> None:
    assert symmetry_weight_for_epoch(1.0, epoch=1, warmup_epochs=4) == 0.25
    assert symmetry_weight_for_epoch(1.0, epoch=2, warmup_epochs=4) == 0.5
    assert symmetry_weight_for_epoch(1.0, epoch=5, warmup_epochs=4) == 1.0
    assert symmetry_weight_for_epoch(0.75, epoch=1, warmup_epochs=0) == 0.75


def test_topology_order_loss_penalizes_wrong_vertical_order() -> None:
    vertical = torch.tensor([[0.0, 0.5, 1.0]])
    correct_logits = torch.tensor(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
        dtype=torch.float32,
    )
    wrong_logits = torch.tensor(
        [[[0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    assert topology_order_loss(correct_logits, vertical).item() == 0.0
    assert topology_order_loss(wrong_logits, vertical).item() > 0.0

    correct_mass_weighted = topology_order_loss(
        correct_logits,
        vertical,
        mass_weighting=True,
    )
    wrong_mass_weighted = topology_order_loss(
        wrong_logits,
        vertical,
        mass_weighting=True,
    )
    assert correct_mass_weighted.item() < wrong_mass_weighted.item()


def test_strong_supervision_loss_is_differentiable() -> None:
    logits = torch.randn(64, 3, requires_grad=True)
    targets = torch.randint(0, 3, (64,))
    criterion = StrongSupervisionLoss(ohem_ratio=0.2, min_ohem_points=8)

    loss = criterion(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_strong_supervision_loss_skips_ohem_for_small_batches() -> None:
    logits = torch.randn(16, 3)
    targets = torch.randint(0, 3, (16,))
    unfiltered = StrongSupervisionLoss(ohem_ratio=0.2, min_ohem_points=1000)
    filtered = StrongSupervisionLoss(ohem_ratio=0.2, min_ohem_points=1)

    plain = unfiltered(logits, targets)
    hard = filtered(logits, targets)

    # The small-batch fallback must reproduce the unfiltered mean exactly.
    full_mean = StrongSupervisionLoss(ohem_ratio=1.0, min_ohem_points=1000)
    assert torch.allclose(plain, full_mean(logits, targets))
    assert hard.item() >= plain.item() - 1e-6


def test_strong_supervision_loss_handles_ignore_index_and_empty_batches() -> None:
    criterion = StrongSupervisionLoss(ohem_ratio=0.2, min_ohem_points=1)
    logits = torch.randn(8, 3, requires_grad=True)
    all_ignored = torch.full((8,), -100)

    loss = criterion(logits, all_ignored)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None

    empty_logits = torch.randn(0, 3, requires_grad=True)
    empty = criterion(empty_logits, torch.zeros((0,), dtype=torch.long))
    assert empty.item() == 0.0


def test_strong_supervision_loss_weights_rare_classes_without_hardcoding() -> None:
    # A rare class that is misclassified must dominate the loss.
    logits = torch.full((100, 4), 0.0)
    targets = torch.zeros((100,), dtype=torch.long)
    targets[0] = 3
    logits[0, 3] = -10.0
    logits[0, 0] = 10.0
    criterion = StrongSupervisionLoss(ohem_ratio=0.05, min_ohem_points=1)

    hard_loss = criterion(logits, targets)

    fixed_logits = logits.clone()
    fixed_logits[0, 3] = 10.0
    fixed_logits[0, 0] = -10.0
    easy_loss = criterion(fixed_logits, targets)

    assert hard_loss.item() > easy_loss.item()


def test_build_segmentation_loss_accepts_ohem_options() -> None:
    criterion = build_segmentation_loss(
        "ohem_weighted",
        class_weights=torch.ones(3),
        ohem_ratio=0.5,
        ohem_min_points=7,
    )

    assert isinstance(criterion, StrongSupervisionLoss)
    assert criterion.ohem_ratio == 0.5
    assert criterion.min_ohem_points == 7
