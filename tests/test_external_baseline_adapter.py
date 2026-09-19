from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "work" / "brpcd" / "train_bridgenetv2_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_bridgenetv2_baseline",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FixedCardinalityModel(nn.Module):
    def __init__(self, expected_points: int) -> None:
        super().__init__()
        self.expected_points = expected_points

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        assert xyz.shape[1] == self.expected_points
        logits = xyz.mean(dim=-1, keepdim=True)
        return logits.expand(-1, -1, 2)


def test_adapter_pads_and_restores_prediction_count() -> None:
    module = _load_module()
    adapter = module.BridgeNetV2Adapter(
        _FixedCardinalityModel(expected_points=6),
        pad_to_points=6,
    )
    features = torch.randn(2, 9, 4)
    logits = adapter(features)
    assert logits.shape == (2, 2, 4)


def test_log_probability_focal_loss_is_differentiable() -> None:
    module = _load_module()
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    log_probabilities = torch.log_softmax(logits, dim=1)
    targets = torch.tensor([[0, 1, 2, 1, 0], [2, 2, 1, 0, 1]])
    criterion = module.LogProbabilityFocalLoss(
        weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        gamma=2.0,
    )

    loss = criterion(log_probabilities, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
