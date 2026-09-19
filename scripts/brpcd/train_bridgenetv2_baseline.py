"""BridgeNetv2 external baseline under the paper's own training protocol.

The published BridgeNetv2 checkpoint is trained on the authors' synthetic
masonry-bridge data, so re-using their weights would confound backbone quality
with a domain gap. This adapter therefore keeps everything except the backbone
identical to our main runs: the same prepared split, the same class-balanced
block sampling, the same loss and the same 0/25/50/75 % occlusion evaluation.

Only the first three (per-block normalised) coordinates are handed to the
model, because BridgeNetv2 consumes xyz and our RGB channel is degenerate.

Run from the project root:

    $env:PYTHONPATH = "src"
    D:\\Codex\\envs\\pointcloud\\Scripts\\python.exe work\\brpcd\\train_bridgenetv2_baseline.py `
        --output-dir runs\bridgenetv2_baseline_seed42 --seed 42 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bridge_seg.classes import CLASS_NAMES
from bridge_seg.data import discover_prepared_splits, load_scene_npz
from bridge_seg.dataset import PreparedSceneBlockDataset
from bridge_seg.losses import build_segmentation_loss, inverse_frequency_weights
from bridge_seg.metrics import segmentation_metrics

DEFAULT_REPO = Path(__file__).resolve().parents[1] / "repro" / "BridgeNetv2-master"


class BridgeNetV2Adapter(nn.Module):
    """Expose BridgeNetv2 through the framework's [B, C, N] interface.

    BridgeNetv2's hierarchical kNN blocks require a fixed input cardinality.
    When occlusion removes points, ``pad_to_points`` repeats surviving points
    so the backbone receives a valid block. Predictions are sliced back to the
    original count, so repeated points do not enter the metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        pad_to_points: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.pad_to_points = pad_to_points

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, 9, N] -> xyz only, as [B, N, 3]
        xyz = features[:, :3, :].transpose(1, 2).contiguous()
        point_count = int(xyz.shape[1])
        if self.pad_to_points is not None:
            if point_count > self.pad_to_points:
                raise ValueError(
                    "point count exceeds pad_to_points: "
                    f"{point_count} > {self.pad_to_points}"
                )
            if point_count < self.pad_to_points:
                indices = torch.arange(
                    self.pad_to_points,
                    device=xyz.device,
                ) % point_count
                xyz = xyz[:, indices, :]
        logits = self.model(xyz)
        logits = logits[:, :point_count, :]
        return logits.transpose(1, 2).contiguous()


class LogProbabilityFocalLoss(nn.Module):
    """Weighted focal loss for a module that already returns log-probabilities."""

    def __init__(self, weights: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.register_buffer("weights", weights.detach().clone())
        self.gamma = float(gamma)

    def forward(
        self,
        log_probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if log_probabilities.ndim != 3:
            raise ValueError(
                "log_probabilities must have shape [B, num_classes, N]"
            )
        expected_shape = (
            log_probabilities.shape[0],
            log_probabilities.shape[2],
        )
        if tuple(targets.shape) != expected_shape:
            raise ValueError(f"targets must have shape {expected_shape}")

        class_ids = targets.reshape(-1).long()
        flat = log_probabilities.transpose(1, 2).reshape(
            -1,
            log_probabilities.shape[1],
        )
        target_log_probability = flat.gather(
            1,
            class_ids.unsqueeze(1),
        ).squeeze(1)
        focal_factor = (
            1.0 - target_log_probability.exp()
        ).pow(self.gamma)
        point_weights = self.weights.to(log_probabilities.device)[class_ids]
        return (
            point_weights * focal_factor * (-target_log_probability)
        ).mean()


def _training_labels(paths: list[Path]) -> np.ndarray:
    chunks = [load_scene_npz(path).labels for path in paths]
    return np.concatenate(chunks)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
            truths.append(labels.detach().cpu().numpy())
    y_true = np.concatenate([item.reshape(-1) for item in truths])
    y_pred = np.concatenate([item.reshape(-1) for item in predictions])
    metrics = segmentation_metrics(y_true, y_pred, len(CLASS_NAMES))
    metrics["loss"] = float(np.mean(losses))
    metrics["points"] = int(len(y_true))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(r"D:\Codex\datasets\BrPCD\official\prepared_20k_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--points-per-block", type=int, default=1024)
    parser.add_argument("--train-blocks-per-scene", type=int, default=8)
    parser.add_argument("--eval-blocks-per-scene", type=int, default=8)
    parser.add_argument("--test-blocks-per-scene", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-mode", default="class_balanced")
    parser.add_argument("--loss", default="focal_weighted")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--eval-occlusion-rates", default="0,0.25,0.5,0.75")
    parser.add_argument("--eval-occlusion-strategy", default="viewpoint")
    parser.add_argument(
        "--pad-to-points",
        type=int,
        default=None,
        help=(
            "Repeat surviving points to this fixed count before BridgeNetv2. "
            "Use the training block size for occluded evaluation."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Load output-dir/best_model.pt and only run the test evaluation. "
            "BridgeNetv2 needs at least 544 input points (N/32 >= 17 for the "
            "deepest kNN), so rates below that are not evaluable at 1024-point "
            "blocks."
        ),
    )
    parser.add_argument(
        "--history-log",
        type=Path,
        default=None,
        help="Optional JSON-lines log used to restore history in eval-only mode.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / "conf").is_dir():
        raise SystemExit(f"BridgeNetv2 repo not found at {repo}")
    sys.path.insert(0, str(repo))

    rates = tuple(
        float(part)
        for part in str(args.eval_occlusion_rates).split(",")
        if part.strip()
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    with initialize_config_dir(version_base=None, config_dir=str(repo / "conf")):
        config = compose(
            config_name="bridgenet_params",
            overrides=[
                "general.backbone=bridgenetv2",
                f"general.n_points={args.points_per_block}",
            ],
        )
    model_config = config.models.base[config.general.backbone]
    OmegaConf.set_struct(model_config, False)
    model_config.n_cls = len(CLASS_NAMES)
    backbone = hydra.utils.instantiate(model_config)
    model = BridgeNetV2Adapter(
        backbone,
        pad_to_points=args.pad_to_points,
    ).to(device)

    splits = discover_prepared_splits(args.prepared_root)
    train_dataset = PreparedSceneBlockDataset(
        splits["train"],
        points_per_block=args.points_per_block,
        blocks_per_scene=args.train_blocks_per_scene,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
    )
    val_dataset = PreparedSceneBlockDataset(
        splits["val"],
        points_per_block=args.points_per_block,
        blocks_per_scene=args.eval_blocks_per_scene,
        seed=args.seed + 1000,
        sampling_mode="uniform",
    )
    test_loaders = {
        rate: DataLoader(
            PreparedSceneBlockDataset(
                splits["test"],
                points_per_block=args.points_per_block,
                blocks_per_scene=args.test_blocks_per_scene,
                seed=args.seed + 2000,
                sampling_mode="uniform",
                occlusion_rate=rate,
                occlusion_strategy=args.eval_occlusion_strategy,
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        for rate in rates
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    weights = inverse_frequency_weights(_training_labels(splits["train"])).to(
        device
    )
    if args.loss != "focal_weighted":
        raise ValueError(
            "BridgeNetv2 returns log-probabilities; this adapter expects "
            "--loss focal_weighted"
        )
    criterion = LogProbabilityFocalLoss(
        weights=weights,
        gamma=args.focal_gamma,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pt"
    history: list[dict[str, object]] = []
    best_val_iou = -1.0

    if args.eval_only and args.history_log is not None:
        history = [
            json.loads(line)
            for line in args.history_log.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip().startswith("{")
        ]
        if history:
            best_val_iou = max(
                float(record["val_mean_iou"]) for record in history
            )

    minimum_points = int(args.points_per_block * (1.0 - min(rates)))
    print(
        f"minimum points across rates: {minimum_points}",
        flush=True,
    )
    if minimum_points < 544 and args.pad_to_points is None:
        raise SystemExit(
            "BridgeNetv2 needs at least 544 points at the deepest kNN; "
            "pass --pad-to-points to repeat surviving points."
        )

    for epoch in range(1, (1 if args.eval_only else args.epochs + 1)):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_points = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_points += labels.numel()

        validation = _evaluate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_points, 1),
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
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_by_occlusion = {
        f"{rate:.2f}": _evaluate(model, loader, criterion, device)
        for rate, loader in test_loaders.items()
    }
    metrics = {
        "classes": list(CLASS_NAMES),
        "seed": args.seed,
        "device": str(device),
        "backbone": "bridgenetv2",
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "points_per_block": args.points_per_block,
            "train_blocks_per_scene": args.train_blocks_per_scene,
            "eval_blocks_per_scene": args.eval_blocks_per_scene,
            "test_blocks_per_scene": args.test_blocks_per_scene,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "sampling_mode": args.sampling_mode,
            "loss_name": f"{args.loss}_log_probability",
            "focal_gamma": args.focal_gamma,
            "eval_occlusion_rates": list(rates),
            "eval_occlusion_strategy": args.eval_occlusion_strategy,
            "pad_to_points": args.pad_to_points,
            "symmetry_mode": "none",
            "topology_weight": 0.0,
            "backbone": "bridgenetv2",
            "mirror_tta": False,
            "input_channels": 3,
            "protocol": (
                "BridgeNetv2 backbone under the paper's own data, sampling, "
                "loss and evaluation protocol; no published weights reused. "
                "The native log-probability output is paired with a focal NLL "
                "loss; no second softmax is applied. "
                "Occluded blocks repeat surviving points to keep the fixed "
                "cardinality required by BridgeNetv2; repeated predictions "
                "are removed before metrics."
            ),
        },
        "best_val_mean_iou": best_val_iou,
        "test": test_by_occlusion[f"{min(rates):.2f}"],
        "test_by_occlusion": test_by_occlusion,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    if history:
        with (args.output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    print(json.dumps(test_by_occlusion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
