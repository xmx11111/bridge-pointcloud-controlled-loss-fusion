from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .classes import CLASS_NAMES


class ProjectionSceneDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load each train/val/test scene's top, side, and end projections."""

    def __init__(
        self,
        paths: list[Path],
        augmentation: bool = False,
        seed: int = 42,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0
        self.images: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.view_names: list[str] = []
        self.scenes: list[str] = []

        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                images = np.asarray(data["images"], dtype=np.uint8)
                labels = np.asarray(data["labels"], dtype=np.uint8)
                view_names = [str(name) for name in data["view_names"].tolist()]
            if images.ndim != 4 or images.shape[1] != 4:
                raise ValueError(f"{path}: images must have shape [V, 4, H, W]")
            if labels.shape != (images.shape[0], images.shape[2], images.shape[3]):
                raise ValueError(f"{path}: invalid projection label shape")
            for view_index, view_name in enumerate(view_names):
                self.images.append(images[view_index])
                self.labels.append(labels[view_index])
                self.view_names.append(view_name)
                self.scenes.append(path.stem)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]
        if self.augmentation:
            rng = np.random.default_rng(self.seed + self.epoch * 100003 + index)
            if rng.random() < 0.5:
                image = image[:, :, ::-1]
                label = label[:, ::-1]
            if rng.random() < 0.5:
                image = image[:, ::-1, :]
                label = label[::-1, :]
            image = np.ascontiguousarray(image)
            label = np.ascontiguousarray(label)
        return (
            torch.from_numpy(image.astype(np.float32) / 255.0),
            torch.from_numpy(label.astype(np.int64)),
        )


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ProjectionUNet(nn.Module):
    """Small U-Net for 192x192 virtual bridge projections."""

    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = len(CLASS_NAMES),
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.enc1 = _conv_block(input_channels, base_channels)
        self.enc2 = _conv_block(base_channels, base_channels * 2)
        self.enc3 = _conv_block(base_channels * 2, base_channels * 4)
        self.bottleneck = _conv_block(
            base_channels * 4,
            base_channels * 8,
        )
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            2,
            stride=2,
        )
        self.dec3 = _conv_block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            2,
            stride=2,
        )
        self.dec2 = _conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            2,
            stride=2,
        )
        self.dec1 = _conv_block(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 4:
            raise ValueError("images must have shape [B, 4, H, W]")
        enc1 = self.enc1(images)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.head(dec1)
