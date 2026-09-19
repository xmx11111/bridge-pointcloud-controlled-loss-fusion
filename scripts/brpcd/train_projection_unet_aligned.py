"""Train the 2D U-Net with the plug-in force-alignment constraint.

The constraint only acts on pixels that have a visible point correspondence
(the same Z-buffer rule the fusion evaluator uses) and pulls the 2D branch's
softmax towards the frozen 3D branch's class distribution on those points.
It is active during training only; inference is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge_seg.classes import CLASS_NAMES  # noqa: E402
from bridge_seg.metrics import confusion_matrix  # noqa: E402
from bridge_seg.projection import ProjectionUNet  # noqa: E402
from train_projection_unet_geometry import (  # noqa: E402
    CrossEntropyDiceLoss,
    _class_weights,
    _metrics_from_confusion,
)

VISIBILITY_ATOL = 1e-3


class AlignedProjectionSceneDataset(Dataset):
    """Projections bundled with the 3D branch's per-pixel alignment target."""

    def __init__(
        self,
        paths: list[Path],
        prior_dir: Path,
        augmentation: bool = False,
        seed: int = 42,
    ) -> None:
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0
        self.images: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.priors: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.view_names: list[str] = []
        self.scenes: list[str] = []

        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                images = np.asarray(data["images"], dtype=np.uint8)
                labels = np.asarray(data["labels"], dtype=np.uint8)
                view_indices = np.asarray(data["view_indices"], dtype=np.int32)
                point_depths = np.asarray(data["point_depths"], dtype=np.float32)
                depth_maps = np.asarray(data["depth_maps"], dtype=np.float32)
                view_names = [str(name) for name in data["view_names"].tolist()]
            prior_path = prior_dir / f"{path.stem}.npz"
            if not prior_path.exists():
                raise FileNotFoundError(f"missing 3D prior: {prior_path}")
            with np.load(prior_path, allow_pickle=False) as prior_data:
                point_probabilities = np.asarray(
                    prior_data["probabilities"],
                    dtype=np.float32,
                )
                prior_counts = np.asarray(prior_data["counts"], dtype=np.int64)

            height, width = images.shape[2], images.shape[3]
            for view_index, view_name in enumerate(view_names):
                vertical = np.clip(
                    view_indices[view_index, :, 0],
                    0,
                    height - 1,
                )
                horizontal = np.clip(
                    view_indices[view_index, :, 1],
                    0,
                    width - 1,
                )
                depth_at_pixel = depth_maps[view_index, vertical, horizontal]
                visible = (
                    np.isfinite(depth_at_pixel)
                    & np.isfinite(point_depths[view_index])
                    & (prior_counts > 0)
                    & np.isclose(
                        point_depths[view_index],
                        depth_at_pixel,
                        rtol=0.0,
                        atol=VISIBILITY_ATOL,
                    )
                )
                prior_image = np.zeros(
                    (len(CLASS_NAMES), height, width),
                    dtype=np.float16,
                )
                mask = np.zeros((height, width), dtype=np.uint8)
                if bool(visible.any()):
                    rows = vertical[visible]
                    columns = horizontal[visible]
                    prior_image[:, rows, columns] = point_probabilities[
                        visible
                    ].T.astype(np.float16)
                    mask[rows, columns] = 1
                self.images.append(images[view_index])
                self.labels.append(labels[view_index])
                self.priors.append(prior_image)
                self.masks.append(mask)
                self.view_names.append(view_name)
                self.scenes.append(path.stem)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]
        prior = self.priors[index]
        mask = self.masks[index]
        if self.augmentation:
            rng = np.random.default_rng(self.seed + self.epoch * 100003 + index)
            if rng.random() < 0.5:
                image = image[:, :, ::-1]
                label = label[:, ::-1]
                prior = prior[:, :, ::-1]
                mask = mask[:, ::-1]
            if rng.random() < 0.5:
                image = image[:, ::-1, :]
                label = label[::-1, :]
                prior = prior[:, ::-1, :]
                mask = mask[::-1, :]
            image = np.ascontiguousarray(image)
            label = np.ascontiguousarray(label)
            prior = np.ascontiguousarray(prior)
            mask = np.ascontiguousarray(mask)
        return (
            torch.from_numpy(image.astype(np.float32) / 255.0),
            torch.from_numpy(label.astype(np.int64)),
            torch.from_numpy(prior.astype(np.float32)),
            torch.from_numpy(mask.astype(bool)),
        )


def alignment_loss(
    logits: torch.Tensor,
    prior: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy between the 2D softmax and the 3D distribution."""

    if not bool(mask.any()):
        return logits.sum() * 0.0
    log_probabilities = torch.log_softmax(logits, dim=1)
    per_pixel = -(prior * log_probabilities).sum(dim=1)
    weights = mask.to(per_pixel.dtype)
    return (per_pixel * weights).sum() / weights.sum().clamp(min=1.0)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    """Pixel metrics over the validation/test loaders of this script."""

    model.eval()
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-root",
        type=Path,
        default=Path(
            r"D:\Codex\datasets\BrPCD\official\projection_geom_512_v6_radius0"
        ),
    )
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--align-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

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
        split: AlignedProjectionSceneDataset(
            split_paths,
            prior_dir=args.prior_dir,
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
    best_path = args.output_dir / "best_model.pt"
    best_val_iou = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_align = 0.0
        total_pixels = 0
        for images, labels, priors, masks in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            priors = priors.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            segmentation = criterion(logits, labels)
            alignment = alignment_loss(logits, priors, masks)
            loss = segmentation + args.align_weight * alignment
            loss.backward()
            optimizer.step()
            occupied = labels != 255
            total_loss += float(loss.detach().cpu()) * int(occupied.sum())
            total_align += float(alignment.detach().cpu()) * int(occupied.sum())
            total_pixels += int(occupied.sum())
        scheduler.step()

        validation = evaluate_model(model, loaders["val"], device)
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / max(total_pixels, 1),
            "train_alignment": total_align / max(total_pixels, 1),
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
                        "align_weight": args.align_weight,
                    },
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_model(model, loaders["test"], device)
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
            "align_weight": args.align_weight,
            "projection_root": str(args.projection_root),
            "prior_dir": str(args.prior_dir),
            "constraint": "masked soft-target cross-entropy on visible pixels",
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
