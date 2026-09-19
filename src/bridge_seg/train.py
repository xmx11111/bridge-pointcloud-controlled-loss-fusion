from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .classes import CLASS_NAMES
from .data import discover_prepared_splits, load_scene_npz
from .dataset import PreparedSceneBlockDataset
from .losses import (
    LOSS_NAMES,
    build_segmentation_loss,
    inverse_frequency_weights,
    symmetry_consistency_loss,
    symmetry_weight_for_epoch,
    topology_order_loss,
)
from .metrics import segmentation_metrics
from .models import PointNetPlusPlusSegmentation, PointNetSegmentation


def _unpack_batch(
    batch: tuple[torch.Tensor, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if len(batch) == 2:
        features, labels = batch
        return features, labels, None, None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2], None
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    raise ValueError("Batch must contain features, labels, and optional mirrors")


def _training_labels(paths: list[Path]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for path in paths:
        labels = load_scene_npz(path).labels
        chunks.append(labels)
    return np.concatenate(chunks)


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    cross_side: bool = False,
    mirror_tta: bool = False,
    mirror_tta_weight_mode: str = "fixed",
    mirror_tta_confidence_threshold: float = 0.7,
) -> dict[str, object]:
    if cross_side and mirror_tta:
        raise ValueError("cross_side and mirror_tta are mutually exclusive")
    if mirror_tta_weight_mode not in (
        "fixed",
        "none",
        "geometry",
        "random",
        "confidence",
    ):
        raise ValueError(
            f"Unsupported mirror TTA weight mode: {mirror_tta_weight_mode}"
        )
    if not 0.0 <= mirror_tta_confidence_threshold <= 1.0:
        raise ValueError(
            "mirror_tta_confidence_threshold must be in [0, 1]"
        )
    model.eval()
    losses: list[float] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            features, labels, mirrored_features, eligibility = _unpack_batch(batch)
            features = features.to(device)
            labels = labels.to(device)
            if cross_side:
                if mirrored_features is None:
                    raise RuntimeError(
                        "Cross-side evaluation requires mirrored blocks"
                    )
                fused_logits, original_logits = model.forward_cross_side(
                    features,
                    mirrored_features.to(device),
                    return_original=True,
                )
                if eligibility is None:
                    logits = fused_logits
                else:
                    use_cross_side = (
                        eligibility.reshape(-1, 1, 1).to(device) > 0.5
                    )
                    logits = torch.where(
                        use_cross_side,
                        fused_logits,
                        original_logits,
                    )
            elif mirror_tta:
                if mirrored_features is None:
                    raise RuntimeError(
                        "Mirror TTA requires geometry-only mirrored blocks"
                    )
                original_logits = model(features)
                mirrored_logits = model(mirrored_features.to(device))
                original_probabilities = torch.softmax(original_logits, dim=1)
                if mirror_tta_weight_mode == "none":
                    mirror_weight = torch.zeros(
                        (features.shape[0], 1, 1),
                        device=device,
                    )
                elif mirror_tta_weight_mode == "confidence":
                    sample_confidence = original_probabilities.max(
                        dim=1
                    ).values.mean(dim=1)
                    mirror_weight = (
                        (sample_confidence >= mirror_tta_confidence_threshold)
                        .to(original_probabilities.dtype)
                        .reshape(-1, 1, 1)
                        * 0.5
                    )
                elif eligibility is None:
                    mirror_weight = torch.full(
                        (features.shape[0], 1, 1),
                        0.5,
                        device=device,
                    )
                else:
                    mirror_weight = eligibility.reshape(-1, 1, 1).to(device)
                probabilities = (
                    (1.0 - mirror_weight)
                    * original_probabilities
                    + mirror_weight * torch.softmax(mirrored_logits, dim=1)
                )
                logits = torch.log(probabilities.clamp_min(1e-12))
            else:
                logits = model(features)
            loss = criterion(logits.reshape(-1, len(CLASS_NAMES)), labels.reshape(-1))
            losses.append(float(loss.detach().cpu()))
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
            truths.append(labels.detach().cpu().numpy())

    y_true = np.concatenate([item.reshape(-1) for item in truths])
    y_pred = np.concatenate([item.reshape(-1) for item in predictions])
    metrics = segmentation_metrics(y_true, y_pred, len(CLASS_NAMES))
    metrics["loss"] = float(np.mean(losses))
    metrics["points"] = int(len(y_true))
    return metrics


def train_model(
    prepared_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    points_per_block: int,
    train_blocks_per_scene: int,
    eval_blocks_per_scene: int,
    test_blocks_per_scene: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    sampling_mode: str,
    loss_name: str,
    focal_gamma: float,
    occlusion_rate: float,
    occlusion_strategy: str,
    eval_occlusion_rates: tuple[float, ...] = (0.0,),
    eval_occlusion_strategy: str = "viewpoint",
    symmetry_mode: str = "none",
    symmetry_weight: float = 0.0,
    symmetry_warmup_epochs: int = 0,
    topology_weight: float = 0.0,
    topology_mass_weighting: bool = False,
    backbone: str = "pointnet",
    mirror_tta: bool = False,
    mirror_tta_weight_mode: str = "fixed",
    mirror_tta_geometry_evidence_threshold: float = 0.0,
    mirror_tta_confidence_threshold: float = 0.7,
    ohem_ratio: float = 0.2,
    ohem_min_points: int = 100,
) -> dict[str, object]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if symmetry_mode not in (
        "none",
        "augmentation",
        "consistency",
        "wrong_axis",
        "cross_side",
    ):
        raise ValueError(f"Unsupported symmetry mode: {symmetry_mode}")
    if symmetry_weight < 0:
        raise ValueError("symmetry_weight must be non-negative")
    if symmetry_warmup_epochs < 0:
        raise ValueError("symmetry_warmup_epochs must be non-negative")
    if topology_weight < 0:
        raise ValueError("topology_weight must be non-negative")
    if symmetry_mode == "cross_side" and mirror_tta:
        raise ValueError("cross_side and mirror_tta are mutually exclusive")
    if mirror_tta_weight_mode not in (
        "fixed",
        "none",
        "geometry",
        "random",
        "confidence",
    ):
        raise ValueError(
            f"Unsupported mirror TTA weight mode: {mirror_tta_weight_mode}"
        )
    if not 0.0 <= mirror_tta_geometry_evidence_threshold < 1.0:
        raise ValueError(
            "mirror_tta_geometry_evidence_threshold must be in [0, 1)"
        )
    if not 0.0 <= mirror_tta_confidence_threshold <= 1.0:
        raise ValueError(
            "mirror_tta_confidence_threshold must be in [0, 1]"
        )

    splits = discover_prepared_splits(prepared_root)
    train_dataset = PreparedSceneBlockDataset(
        splits["train"],
        points_per_block=points_per_block,
        blocks_per_scene=train_blocks_per_scene,
        seed=seed,
        sampling_mode=sampling_mode,
        occlusion_rate=occlusion_rate,
        occlusion_strategy=occlusion_strategy,
        symmetry_mode=symmetry_mode,
    )
    val_dataset = PreparedSceneBlockDataset(
        splits["val"],
        points_per_block=points_per_block,
        blocks_per_scene=eval_blocks_per_scene,
        seed=seed + 1000,
        sampling_mode="uniform",
        symmetry_mode="cross_side" if symmetry_mode == "cross_side" else "none",
    )
    test_datasets = {
        rate: PreparedSceneBlockDataset(
            splits["test"],
            points_per_block=points_per_block,
            blocks_per_scene=test_blocks_per_scene,
            seed=seed + 2000,
            sampling_mode="uniform",
            occlusion_rate=rate,
            occlusion_strategy=eval_occlusion_strategy,
            symmetry_mode=(
                "cross_side" if symmetry_mode == "cross_side" else "none"
            ),
            mirror_tta=mirror_tta,
            mirror_tta_weight_mode=(
                "fixed"
                if mirror_tta_weight_mode in ("none", "confidence")
                else mirror_tta_weight_mode
            ),
            mirror_tta_geometry_evidence_threshold=(
                mirror_tta_geometry_evidence_threshold
            ),
        )
        for rate in eval_occlusion_rates
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loaders = {
        rate: DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for rate, dataset in test_datasets.items()
    }

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if backbone == "pointnet":
        model = PointNetSegmentation(
            input_channels=9,
            num_classes=len(CLASS_NAMES),
            use_cross_side=symmetry_mode == "cross_side",
        ).to(device)
    elif backbone == "pointnet++":
        if symmetry_mode == "cross_side":
            raise ValueError("cross_side currently requires the pointnet backbone")
        model = PointNetPlusPlusSegmentation(
            input_channels=9,
            num_classes=len(CLASS_NAMES),
        ).to(device)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if loss_name not in LOSS_NAMES:
        raise ValueError(f"Unsupported loss: {loss_name}")
    weights = inverse_frequency_weights(_training_labels(splits["train"])).to(device)
    criterion = build_segmentation_loss(
        name=loss_name,
        class_weights=weights,
        focal_gamma=focal_gamma,
        ohem_ratio=ohem_ratio,
        ohem_min_points=ohem_min_points,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_val_iou = -1.0
    best_path = output_dir / "best_model.pt"
    effective_symmetry_weight = (
        symmetry_weight
        if symmetry_mode in ("consistency", "wrong_axis", "cross_side")
        else 0.0
    )

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_symmetry_loss = 0.0
        total_topology_loss = 0.0
        total_symmetry_points = 0
        total_points = 0
        epoch_symmetry_weight = symmetry_weight_for_epoch(
            effective_symmetry_weight,
            epoch,
            symmetry_warmup_epochs,
        )
        for batch in train_loader:
            features, labels, mirrored_features, eligible = _unpack_batch(batch)
            features = features.to(device)
            labels = labels.to(device)
            symmetry_loss = torch.zeros((), device=device)
            if symmetry_mode == "augmentation":
                if mirrored_features is None:
                    raise RuntimeError("Mirror augmentation requires mirrored blocks")
                mirrored_features = mirrored_features.to(device)
                if eligible is None:
                    eligible = torch.ones(features.shape[0], device=device)
                else:
                    eligible = eligible.reshape(-1).to(device)
                replace = (
                    (torch.rand(features.shape[0], device=device) < 0.5)
                    & eligible.bool()
                )
                features = torch.where(
                    replace[:, None, None],
                    mirrored_features,
                    features,
                )
            optimizer.zero_grad(set_to_none=True)
            if symmetry_mode == "cross_side":
                if mirrored_features is None:
                    raise RuntimeError(
                        "Cross-side propagation requires mirrored blocks"
                    )
                if eligible is None:
                    eligible = torch.ones(features.shape[0], device=device)
                else:
                    eligible = eligible.reshape(-1).to(device)
                fused_logits, original_logits = model.forward_cross_side(
                    features,
                    mirrored_features.to(device),
                    return_original=True,
                )
                flat_logits = fused_logits.reshape(-1, len(CLASS_NAMES))
                flat_original = original_logits.reshape(-1, len(CLASS_NAMES))
                base_loss = criterion(
                    flat_original,
                    labels.reshape(-1),
                )
                sample_loss = criterion(
                    flat_logits,
                    labels.reshape(-1),
                )
                symmetry_loss = sample_loss
                loss = base_loss + epoch_symmetry_weight * symmetry_loss
                active_samples = int(eligible.sum().item())
                total_symmetry_loss += (
                    float(symmetry_loss.detach().cpu())
                    * active_samples
                    * labels.shape[1]
                )
                total_symmetry_points += active_samples * labels.shape[1]
                logits = fused_logits
            else:
                logits = model(features)
                flat_logits = logits.reshape(-1, len(CLASS_NAMES))
                base_loss = criterion(flat_logits, labels.reshape(-1))
                symmetry_loss = torch.zeros((), device=device)
                loss = base_loss
                active_samples = 0
            if symmetry_mode in ("consistency", "wrong_axis"):
                if mirrored_features is None:
                    raise RuntimeError("Symmetry consistency requires mirrored blocks")
                mirrored_logits = model(mirrored_features.to(device))
                if eligible is None:
                    eligible = torch.ones(logits.shape[0], device=device)
                else:
                    eligible = eligible.reshape(-1).to(device)
                symmetry_loss = symmetry_consistency_loss(
                    logits,
                    mirrored_logits,
                    sample_weights=eligible,
                )
                loss = base_loss + epoch_symmetry_weight * symmetry_loss
            if topology_weight > 0:
                vertical = features[:, 5, :]
                topology_loss = topology_order_loss(
                    logits,
                    vertical,
                    mass_weighting=topology_mass_weighting,
                )
                loss = loss + topology_weight * topology_loss
                total_topology_loss += (
                    float(topology_loss.detach().cpu()) * labels.shape[0]
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * labels.numel()
            if symmetry_mode in ("consistency", "wrong_axis"):
                active_samples = int(eligible.sum().item())
                total_symmetry_loss += (
                    float(symmetry_loss.detach().cpu())
                    * active_samples
                    * labels.shape[1]
                )
                total_symmetry_points += active_samples * labels.shape[1]
            total_points += labels.numel()

        validation = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cross_side=symmetry_mode == "cross_side",
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_points, 1),
            "train_symmetry_loss": total_symmetry_loss
            / max(total_symmetry_points, 1),
            "train_topology_loss": total_topology_loss
            / max(len(train_loader), 1),
            "symmetry_weight": epoch_symmetry_weight,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "val_mean_iou": validation["mean_iou"],
            "val_per_class_iou": validation["per_class_iou"],
            "val_points": validation["points"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if float(validation["mean_iou"]) > best_val_iou:
            best_val_iou = float(validation["mean_iou"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "input_channels": 9,
                    "epoch": epoch,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_by_occlusion = {
        f"{rate:.2f}": evaluate(
            model,
            test_loader,
            criterion,
            device,
            cross_side=symmetry_mode == "cross_side",
            mirror_tta=mirror_tta,
            mirror_tta_weight_mode=mirror_tta_weight_mode,
            mirror_tta_confidence_threshold=mirror_tta_confidence_threshold,
        )
        for rate, test_loader in test_loaders.items()
    }
    test_metrics = test_by_occlusion[
        f"{min(eval_occlusion_rates):.2f}"
        if 0.0 not in eval_occlusion_rates
        else "0.00"
    ]

    metrics: dict[str, object] = {
        "classes": list(CLASS_NAMES),
        "seed": seed,
        "device": str(device),
        "configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "points_per_block": points_per_block,
            "train_blocks_per_scene": train_blocks_per_scene,
            "eval_blocks_per_scene": eval_blocks_per_scene,
            "test_blocks_per_scene": test_blocks_per_scene,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "sampling_mode": sampling_mode,
            "loss_name": loss_name,
            "focal_gamma": focal_gamma,
            "ohem_ratio": ohem_ratio,
            "ohem_min_points": ohem_min_points,
            "occlusion_rate": occlusion_rate,
            "occlusion_strategy": occlusion_strategy,
            "eval_occlusion_rates": list(eval_occlusion_rates),
            "eval_occlusion_strategy": eval_occlusion_strategy,
            "train_sampling": sampling_mode,
            "eval_sampling": "nearest-neighbour natural block",
            "symmetry_mode": symmetry_mode,
            "symmetry_weight": effective_symmetry_weight,
            "symmetry_warmup_epochs": symmetry_warmup_epochs,
            "topology_weight": topology_weight,
            "topology_mass_weighting": topology_mass_weighting,
            "backbone": backbone,
            "mirror_tta": mirror_tta,
            "mirror_tta_weight_mode": mirror_tta_weight_mode,
            "mirror_tta_geometry_evidence_threshold": (
                mirror_tta_geometry_evidence_threshold
            ),
            "mirror_tta_confidence_threshold": (
                mirror_tta_confidence_threshold
            ),
        },
        "best_val_mean_iou": best_val_iou,
        "test": test_metrics,
        "test_by_occlusion": test_by_occlusion,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return metrics
