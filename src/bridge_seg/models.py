from __future__ import annotations

import torch
from torch import nn


class PointNetSegmentation(nn.Module):
    """Lightweight PointNet-style baseline used by the first research rungs."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        use_global: bool = True,
        use_cross_side: bool = False,
    ) -> None:
        super().__init__()
        self.use_global = use_global
        self.use_cross_side = use_cross_side

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
        if use_cross_side:
            self.cross_side_gate = nn.Sequential(
                nn.Conv1d(256 * 3, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, 1),
                nn.Sigmoid(),
            )
            gate_conv = self.cross_side_gate[2]
            nn.init.zeros_(gate_conv.weight)
            nn.init.constant_(gate_conv.bias, -4.0)

    def _encode_local(self, features: torch.Tensor) -> torch.Tensor:
        return self.local(features)

    def _segment_local(self, local: torch.Tensor) -> torch.Tensor:
        if not self.use_global:
            return self.segmentation(local)
        global_feature = torch.max(
            self.global_encoder(local), dim=2, keepdim=True
        ).values
        global_feature = global_feature.expand(-1, -1, local.shape[2])
        return self.segmentation(torch.cat([local, global_feature], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, C, N]")
        return self._segment_local(self._encode_local(features))

    def forward_cross_side(
        self,
        features: torch.Tensor,
        mirrored_features: torch.Tensor,
        return_original: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self.use_cross_side:
            raise RuntimeError("Model was not configured for cross-side propagation")
        if features.shape != mirrored_features.shape:
            raise ValueError("features and mirrored_features must have the same shape")
        local = self._encode_local(features)
        mirrored_local = self._encode_local(mirrored_features)
        gate_input = torch.cat(
            [local, mirrored_local, torch.abs(local - mirrored_local)],
            dim=1,
        )
        gate = self.cross_side_gate(gate_input)
        fused = local + gate * mirrored_local
        fused_logits = self._segment_local(fused)
        if not return_original:
            return fused_logits
        return fused_logits, self._segment_local(local)


def _sample_count(point_count: int, ratio: float) -> int:
    return max(1, min(point_count, int(round(point_count * ratio))))


class SetAbstraction(nn.Module):
    """PointNet++-style kNN set abstraction without external CUDA operators."""

    def __init__(
        self,
        ratio: float,
        nsample: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.nsample = nsample
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + 3, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, point_count = features.shape
        npoint = _sample_count(point_count, self.ratio)
        indices = torch.linspace(
            0,
            point_count - 1,
            steps=npoint,
            device=features.device,
        ).long()
        indices = torch.unique(indices, sorted=True)
        npoint = len(indices)
        center_positions = positions[:, indices, :]
        neighbor_count = min(self.nsample, point_count)

        distances = torch.cdist(center_positions, positions)
        neighbor_indices = torch.topk(
            distances,
            k=neighbor_count,
            dim=2,
            largest=False,
            sorted=False,
        ).indices

        batch_offsets = (
            torch.arange(batch_size, device=features.device)[:, None, None]
            * point_count
        )
        flat_features = features.transpose(1, 2).reshape(
            batch_size * point_count,
            features.shape[1],
        )
        flat_positions = positions.reshape(batch_size * point_count, 3)
        neighbor_flat = (neighbor_indices + batch_offsets).reshape(
            batch_size * npoint * neighbor_count
        )
        center_flat = (
            indices[None, :].expand(batch_size, -1) + batch_offsets[:, :, 0]
        ).reshape(batch_size * npoint)

        neighbor_features = flat_features[neighbor_flat].reshape(
            batch_size * npoint,
            neighbor_count,
            features.shape[1],
        )
        center_features = flat_features[center_flat].reshape(
            batch_size * npoint,
            1,
            features.shape[1],
        )
        neighbor_positions = flat_positions[neighbor_flat].reshape(
            batch_size * npoint,
            neighbor_count,
            3,
        )
        center_positions_flat = flat_positions[center_flat].reshape(
            batch_size * npoint,
            1,
            3,
        )
        relative = neighbor_positions - center_positions_flat
        grouped = torch.cat(
            [relative, neighbor_features - center_features],
            dim=2,
        )
        grouped = grouped.transpose(1, 2)
        aggregated = self.mlp(grouped).max(dim=2).values
        aggregated = aggregated.reshape(
            batch_size,
            npoint,
            aggregated.shape[1],
        )
        return center_positions, aggregated


class FeaturePropagation(nn.Module):
    """Interpolate coarse features to fine positions with kNN weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nsample: int = 3,
    ) -> None:
        super().__init__()
        self.nsample = nsample
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        coarse_positions: torch.Tensor,
        coarse_features: torch.Tensor,
        fine_positions: torch.Tensor,
        fine_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, fine_count = fine_features.shape
        coarse_count = coarse_positions.shape[1]
        nsample = min(self.nsample, coarse_count)
        distances = torch.cdist(fine_positions, coarse_positions)
        distances, indices = torch.topk(
            distances,
            k=nsample,
            dim=2,
            largest=False,
            sorted=False,
        )
        weights = 1.0 / distances.clamp_min(1e-8)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        gathered = coarse_features[
            torch.arange(batch_size, device=coarse_features.device)[:, None, None],
            indices,
            :,
        ]
        interpolated = (weights[..., None] * gathered).sum(dim=2)
        combined = torch.cat(
            [fine_features, interpolated.transpose(1, 2)],
            dim=1,
        )
        return self.mlp(combined).transpose(1, 2).contiguous()


class PointNetPlusPlusSegmentation(nn.Module):
    """Pure-PyTorch PointNet++-style segmentation backbone."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.sa1 = SetAbstraction(0.25, 16, input_channels, 64)
        self.sa2 = SetAbstraction(0.25, 16, 64, 128)
        self.sa3 = SetAbstraction(0.25, 16, 128, 256)
        self.fp3 = FeaturePropagation(256 + 128, 128)
        self.fp2 = FeaturePropagation(128 + 64, 64)
        self.fp1 = FeaturePropagation(64 + input_channels, 64)
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(64, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"features must have shape [B, {self.input_channels}, N]"
            )
        positions = features[:, :3, :].transpose(1, 2).contiguous()
        pos1, feat1 = self.sa1(features, positions)
        pos2, feat2 = self.sa2(feat1.transpose(1, 2).contiguous(), pos1)
        pos3, feat3 = self.sa3(feat2.transpose(1, 2).contiguous(), pos2)
        up2 = self.fp3(
            pos3,
            feat3,
            pos2,
            feat2.transpose(1, 2).contiguous(),
        )
        up1 = self.fp2(
            pos2,
            up2,
            pos1,
            feat1.transpose(1, 2).contiguous(),
        )
        out = self.fp1(
            pos1,
            up1,
            positions,
            features,
        )
        return self.head(out.transpose(1, 2).contiguous())
