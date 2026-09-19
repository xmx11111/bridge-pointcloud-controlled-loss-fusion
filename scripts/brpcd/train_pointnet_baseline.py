from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["girder", "pier", "deck"]


def class_iou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp / np.maximum(tp + fp + fn, 1.0)


@dataclass
class SceneData:
    name: str
    xyz: np.ndarray
    rgb: np.ndarray
    labels: np.ndarray
    center: np.ndarray
    scale: float
    class_indices: dict[int, np.ndarray]
    tree: cKDTree


class SceneBlockDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        points_per_block: int,
        blocks_per_scene: int,
        seed: int,
        stratified: bool,
        local_patch: bool = True,
    ) -> None:
        self.scenes: list[SceneData] = []
        self.points_per_block = points_per_block
        self.blocks_per_scene = blocks_per_scene
        self.seed = seed
        self.stratified = stratified
        self.local_patch = local_patch
        self.epoch = 0

        for path in sorted(paths):
            data = np.load(path)
            labels = data["labels"].astype(np.int64)
            class_indices = {
                class_id: np.flatnonzero(labels == class_id)
                for class_id in range(len(CLASS_NAMES))
            }
            self.scenes.append(
                SceneData(
                    name=path.stem,
                    xyz=data["xyz"].astype(np.float32),
                    rgb=data["rgb"].astype(np.float32),
                    labels=labels,
                    center=data["center"].astype(np.float32),
                    scale=float(data["scale"][0]),
                    class_indices=class_indices,
                    tree=cKDTree(data["xyz"].astype(np.float32)),
                )
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.scenes) * self.blocks_per_scene

    def _sample_indices(
        self, scene: SceneData, rng: np.random.Generator
    ) -> np.ndarray:
        if not self.stratified:
            if self.local_patch:
                seed_index = int(rng.integers(0, len(scene.xyz)))
                _, indices = scene.tree.query(
                    scene.xyz[seed_index], k=self.points_per_block
                )
                return np.asarray(indices, dtype=np.int64)
            replace = len(scene.xyz) < self.points_per_block
            return rng.choice(len(scene.xyz), self.points_per_block, replace=replace)

        present = [
            class_id
            for class_id, indices in scene.class_indices.items()
            if len(indices) > 0
        ]
        preferred = np.asarray(
            [0.60 if class_id == 0 else 0.25 if class_id == 1 else 0.15 for class_id in present],
            dtype=np.float64,
        )
        preferred /= preferred.sum()
        seed_class = int(rng.choice(present, p=preferred))
        seed_index = int(rng.choice(scene.class_indices[seed_class]))
        nearest = self.points_per_block if self.local_patch else self.points_per_block
        if self.local_patch:
            _, indices = scene.tree.query(scene.xyz[seed_index], k=nearest)
            indices = np.asarray(indices, dtype=np.int64)
        else:
            counts = rng.multinomial(self.points_per_block, preferred)
            chunks = [
                rng.choice(
                    scene.class_indices[class_id],
                    count,
                    replace=count > len(scene.class_indices[class_id]),
                )
                for class_id, count in zip(present, counts)
                if count > 0
            ]
            indices = np.concatenate(chunks)
        rng.shuffle(indices)
        return indices

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene = self.scenes[index % len(self.scenes)]
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + index * 97
        )
        indices = self._sample_indices(scene, rng)
        xyz = scene.xyz[indices]
        rgb = scene.rgb[indices]
        labels = scene.labels[indices]

        global_xyz = (xyz - scene.center) / scene.scale
        local_center = xyz.mean(axis=0, keepdims=True)
        local_scale = float(np.max(np.ptp(xyz, axis=0)))
        if local_scale <= 0:
            local_scale = 1.0
        local_xyz = (xyz - local_center) / local_scale
        features = np.concatenate(
            [local_xyz, global_xyz, rgb], axis=1
        ).astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(labels)


class PointNetSegmentation(nn.Module):
    def __init__(
        self, input_channels: int, num_classes: int, use_global: bool = True
    ) -> None:
        super().__init__()
        self.use_global = use_global

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.local = nn.Sequential(
            block(input_channels, 64),
            block(64, 128),
            block(128, 256),
        )
        self.global_encoder = nn.Sequential(
            block(256, 512),
            block(512, 1024),
        )
        self.segmentation = nn.Sequential(
            block(256 + (1024 if use_global else 0), 512),
            block(512, 256),
            nn.Dropout(0.25),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        local = self.local(x)
        if not self.use_global:
            return self.segmentation(local)
        global_feature = torch.max(self.global_encoder(local), dim=2, keepdim=True).values
        global_feature = global_feature.expand(-1, -1, local.shape[2])
        return self.segmentation(torch.cat([local, global_feature], dim=1))


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    point_count = 0
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits.reshape(-1, len(CLASS_NAMES)), labels.reshape(-1))
            loss_sum += float(loss.item()) * len(features)
            point_count += labels.numel()
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            truths.append(labels.cpu().numpy())

    y_pred = np.concatenate([item.ravel() for item in predictions])
    y_true = np.concatenate([item.ravel() for item in truths])
    ious = class_iou(y_true, y_pred, len(CLASS_NAMES))
    return {
        "loss": loss_sum / max(1, len(loader)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_iou": float(ious.mean()),
        "per_class_iou": ious.tolist(),
        "points": int(point_count),
    }


def training_class_weights(paths: list[Path]) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for path in paths:
        labels = np.load(path)["labels"]
        counts += np.bincount(labels, minlength=len(CLASS_NAMES))
    weights = np.sqrt(counts.sum() / np.maximum(counts * len(CLASS_NAMES), 1.0))
    return np.clip(weights, 0.5, 5.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--points-per-block", type=int, default=1024)
    parser.add_argument("--train-blocks-per-scene", type=int, default=8)
    parser.add_argument("--eval-blocks-per-scene", type=int, default=16)
    parser.add_argument("--test-blocks-per-scene", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-global", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_paths = sorted((args.prepared_root / "train").glob("*.npz"))
    val_paths = sorted((args.prepared_root / "val").glob("*.npz"))
    test_paths = sorted((args.prepared_root / "test").glob("*.npz"))
    if not train_paths or not val_paths or not test_paths:
        raise SystemExit("Missing prepared train/val/test NPZ files.")

    train_dataset = SceneBlockDataset(
        train_paths,
        args.points_per_block,
        args.train_blocks_per_scene,
        args.seed,
        stratified=True,
    )
    val_dataset = SceneBlockDataset(
        val_paths,
        args.points_per_block,
        args.eval_blocks_per_scene,
        args.seed + 1000,
        stratified=False,
    )
    test_dataset = SceneBlockDataset(
        test_paths,
        args.points_per_block,
        args.test_blocks_per_scene,
        args.seed + 2000,
        stratified=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(training_class_weights(train_paths), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    model = PointNetSegmentation(
        9, len(CLASS_NAMES), use_global=args.use_global
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_state = copy.deepcopy(model.state_dict())
    best_val_iou = -1.0
    history: list[dict[str, object]] = []

    print(
        f"device={device} train_blocks={len(train_dataset)} "
        f"val_blocks={len(val_dataset)} test_blocks={len(test_dataset)} "
        f"class_weights={weights.tolist()}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen_blocks = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits.reshape(-1, len(CLASS_NAMES)), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(features)
            seen_blocks += len(features)
        scheduler.step()

        train_loss = running_loss / max(1, seen_blocks)
        val_result = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_result.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if float(val_result["mean_iou"]) > best_val_iou:
            best_val_iou = float(val_result["mean_iou"])
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, args.output_dir / "best_model.pt")

    model.load_state_dict(best_state)
    test_result = evaluate(model, test_loader, criterion, device)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    metrics = {
        "classes": CLASS_NAMES,
        "seed": args.seed,
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "points_per_block": args.points_per_block,
            "train_blocks_per_scene": args.train_blocks_per_scene,
            "eval_blocks_per_scene": args.eval_blocks_per_scene,
            "test_blocks_per_scene": args.test_blocks_per_scene,
            "architecture": "PointNet-style baseline pilot",
            "train_sampling": "stratified girder/pier/deck 0.60/0.25/0.15",
            "eval_sampling": "natural scene distribution",
        },
        "best_val_mean_iou": best_val_iou,
        "test": test_result,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("TEST " + json.dumps(test_result), flush=True)


if __name__ == "__main__":
    main()
