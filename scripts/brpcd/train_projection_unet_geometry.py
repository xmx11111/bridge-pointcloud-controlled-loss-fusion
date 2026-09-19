from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bridge_seg.classes import CLASS_NAMES
from bridge_seg.metrics import confusion_matrix
from bridge_seg.projection import ProjectionSceneDataset, ProjectionUNet


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        occupied = labels != 255
        if not bool(occupied.any()):
            return logits.sum() * 0.0

        ce = nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            ignore_index=255,
        )
        probabilities = torch.softmax(logits, dim=1)
        target = nn.functional.one_hot(
            labels.clamp(min=0, max=len(CLASS_NAMES) - 1),
            num_classes=len(CLASS_NAMES),
        ).permute(0, 3, 1, 2).to(dtype=probabilities.dtype)
        occupied = occupied.unsqueeze(1)
        target = target * occupied
        probabilities = probabilities * occupied
        dimensions = (0, 2, 3)
        intersection = (probabilities * target).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + target.sum(
            dim=dimensions
        )
        dice_per_class = (2.0 * intersection + 1.0) / (denominator + 1.0)
        dice = (dice_per_class * self.class_weights).sum() / self.class_weights.sum()
        return self.ce_weight * ce + self.dice_weight * (1.0 - dice)


def _metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    ious = true_positive / np.maximum(
        true_positive + false_positive + false_negative,
        1.0,
    )
    total = int(matrix.sum())
    return {
        "pixel_count": total,
        "accuracy": float(true_positive.sum() / max(total, 1)),
        "mean_iou": float(ious.mean()),
        "per_class_iou": {
            name: float(ious[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
    }


def _class_weights(dataset: ProjectionSceneDataset) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for label in dataset.labels:
        occupied = label != 255
        counts += np.bincount(
            label[occupied],
            minlength=len(CLASS_NAMES),
        )
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1).astype(np.float64))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    matrix = np.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=np.int64,
    )
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            prediction = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            occupied = labels_np != 255
            matrix += confusion_matrix(
                labels_np[occupied],
                prediction[occupied],
                num_classes=len(CLASS_NAMES),
            )
    return _metrics_from_confusion(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projection-root",
        type=Path,
        default=Path(
            r"D:\Codex\datasets\BrPCD\official\projection_geom_192_v2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\肖\Documents\Codex\2026-09-16\ni"
            r"\runs\projection_geom_unet_seed42"
        ),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    paths = {
        split: sorted((args.projection_root / split).glob("*.npz"))
        for split in ("train", "val", "test")
    }
    datasets = {
        split: ProjectionSceneDataset(
            split_paths,
            augmentation=split == "train",
            seed=args.seed + split_index * 1000,
        )
        for split_index, (split, split_paths) in enumerate(paths.items())
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    model = ProjectionUNet().to(device)
    weights = _class_weights(datasets["train"]).to(device)
    criterion = CrossEntropyDiceLoss(
        weights,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.05,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_iou = -1.0
    best_path = args.output_dir / "best_model.pt"
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_pixels = 0
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            occupied = labels != 255
            total_loss += float(loss.detach().cpu()) * int(occupied.sum())
            total_pixels += int(occupied.sum())
        scheduler.step()

        validation = _evaluate(model, loaders["val"], device)
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / max(total_pixels, 1),
            "val_accuracy": validation["accuracy"],
            "val_mean_iou": validation["mean_iou"],
            "val_per_class_iou": validation["per_class_iou"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if float(validation["mean_iou"]) > best_val_iou:
            best_val_iou = float(validation["mean_iou"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "config": {
                        "base_channels": 32,
                        "input_channels": 4,
                    },
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = _evaluate(model, loaders["test"], device)
    metrics = {
        "classes": list(CLASS_NAMES),
        "seed": args.seed,
        "device": str(device),
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ce_weight": args.ce_weight,
            "dice_weight": args.dice_weight,
            "projection_root": str(args.projection_root),
        },
        "class_weights": weights.cpu().tolist(),
        "best_val_mean_iou": best_val_iou,
        "test": test_metrics,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(metrics["test"], indent=2))
    print(best_path)


if __name__ == "__main__":
    main()
