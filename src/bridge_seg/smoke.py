from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .classes import CLASS_NAMES
from .models import PointNetSegmentation


def run_smoke(output_path: Path, seed: int = 42) -> dict[str, object]:
    """Run a labelled synthetic forward/backward path and write a smoke artifact."""

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    batch_size = 2
    points_per_scene = 256
    input_channels = 6
    features = torch.randn(
        batch_size,
        input_channels,
        points_per_scene,
        generator=generator,
    )
    labels = torch.arange(
        points_per_scene,
        dtype=torch.long,
        device=features.device,
    ) % len(CLASS_NAMES)
    labels = labels.unsqueeze(0).expand(batch_size, -1)

    model = PointNetSegmentation(
        input_channels=input_channels,
        num_classes=len(CLASS_NAMES),
    )
    criterion = nn.CrossEntropyLoss()
    logits = model(features)
    loss = criterion(logits.reshape(-1, len(CLASS_NAMES)), labels.reshape(-1))
    loss.backward()

    finite = bool(torch.isfinite(loss) and torch.isfinite(logits).all())
    payload: dict[str, object] = {
        "status": "ok" if finite else "failed",
        "artifact_kind": "synthetic_smoke",
        "placeholder_labels": True,
        "seed": seed,
        "batch_size": batch_size,
        "points_per_scene": points_per_scene,
        "input_channels": input_channels,
        "classes": list(CLASS_NAMES),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "finite": finite,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
