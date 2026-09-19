from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import (
    SYMMETRY_MODES,
    SceneData,
    load_scene_npz,
    sample_block,
    sample_symmetry_block,
)
from .occlusion import occlude_feature_block
from .symmetry import (
    BridgeFrame,
    estimate_bridge_frame,
    geometry_symmetry_quality,
    symmetry_axis_quality,
)


class PreparedSceneBlockDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Deterministic scene-block sampler over prepared BrPCD NPZ files."""

    def __init__(
        self,
        paths: list[Path],
        points_per_block: int,
        blocks_per_scene: int,
        seed: int,
        sampling_mode: str,
        occlusion_rate: float = 0.0,
        occlusion_strategy: str = "random",
        symmetry_mode: str = "none",
        mirror_tta: bool = False,
        mirror_tta_weight_mode: str = "fixed",
        mirror_tta_geometry_evidence_threshold: float = 0.0,
    ) -> None:
        if not paths:
            raise ValueError("At least one prepared scene is required")
        if blocks_per_scene <= 0:
            raise ValueError("blocks_per_scene must be positive")
        if symmetry_mode not in SYMMETRY_MODES:
            raise ValueError(f"Unsupported symmetry mode: {symmetry_mode}")
        if mirror_tta_weight_mode not in ("fixed", "geometry", "random"):
            raise ValueError(
                f"Unsupported mirror TTA weight mode: {mirror_tta_weight_mode}"
            )
        if not 0.0 <= mirror_tta_geometry_evidence_threshold < 1.0:
            raise ValueError(
                "mirror_tta_geometry_evidence_threshold must be in [0, 1)"
            )
        self.scenes: list[SceneData] = [load_scene_npz(path) for path in paths]
        self.frames: list[BridgeFrame | None] = [
            (
                estimate_bridge_frame(scene.xyz)
                if mirror_tta
                else (
                    None
                    if symmetry_mode == "none"
                    else estimate_bridge_frame(scene.xyz, scene.labels)
                )
            )
            for scene in self.scenes
        ]
        self.symmetry_quality: list[dict[str, float | bool]] = [
            {
                "correct_score": 0.0,
                "wrong_score": 0.0,
                "wrong_to_correct_ratio": 0.0,
                "eligible": False,
            }
            if mirror_tta or symmetry_mode == "none"
            else symmetry_axis_quality(scene.xyz, scene.labels)
            for scene in self.scenes
        ]
        self.points_per_block = points_per_block
        self.blocks_per_scene = blocks_per_scene
        self.seed = seed
        self.sampling_mode = sampling_mode
        self.occlusion_rate = occlusion_rate
        self.occlusion_strategy = occlusion_strategy
        self.symmetry_mode = symmetry_mode
        self.mirror_tta = mirror_tta
        self.mirror_tta_weight_mode = mirror_tta_weight_mode
        self.mirror_tta_geometry_evidence_threshold = (
            mirror_tta_geometry_evidence_threshold
        )
        if not mirror_tta:
            self.mirror_tta_weights = [0.0] * len(self.scenes)
        elif mirror_tta_weight_mode == "fixed":
            self.mirror_tta_weights = [0.5] * len(self.scenes)
        else:
            evidence = [
                float(geometry_symmetry_quality(scene.xyz)["evidence"])
                for scene in self.scenes
            ]
            geometry_weights = [
                0.5 if value > mirror_tta_geometry_evidence_threshold else 0.0
                for value in evidence
            ]
            if mirror_tta_weight_mode == "geometry":
                self.mirror_tta_weights = geometry_weights
            else:
                activation_probability = min(
                    1.0,
                    2.0 * float(np.mean(geometry_weights)),
                )
                rng = np.random.default_rng(seed + 7919)
                self.mirror_tta_weights = [
                    0.5 if draw else 0.0
                    for draw in rng.random(len(self.scenes))
                    < activation_probability
                ]
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.scenes) * self.blocks_per_scene

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene = self.scenes[index % len(self.scenes)]
        scene_index = index % len(self.scenes)
        frame = self.frames[scene_index]
        quality = self.symmetry_quality[scene_index]
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + index * 97
        )
        if self.mirror_tta or self.symmetry_mode != "none":
            symmetry_mode = "cross_side" if self.mirror_tta else self.symmetry_mode
            features, labels, mirrored_features = sample_symmetry_block(
                scene,
                points_per_block=self.points_per_block,
                rng=rng,
                sampling_mode=self.sampling_mode,
                symmetry_mode=symmetry_mode,
                frame=frame,
                occlusion_rate=self.occlusion_rate,
                occlusion_strategy=self.occlusion_strategy,
                occlusion_seed=self.seed + self.epoch * 97 + index,
                symmetry_enabled=(
                    True if self.mirror_tta else bool(quality["eligible"])
                ),
            )
            return (
                torch.from_numpy(features).transpose(0, 1).contiguous(),
                torch.from_numpy(labels),
                torch.from_numpy(mirrored_features)
                .transpose(0, 1)
                .contiguous(),
                torch.tensor(
                    [
                        self.mirror_tta_weights[scene_index]
                        if self.mirror_tta
                        else (1.0 if quality["eligible"] else 0.0)
                    ],
                    dtype=torch.float32,
                ),
            )

        features, labels = sample_block(
            scene,
            points_per_block=self.points_per_block,
            rng=rng,
            sampling_mode=self.sampling_mode,
        )
        if self.occlusion_rate > 0:
            features, labels, _ = occlude_feature_block(
                features,
                labels,
                missing_rate=self.occlusion_rate,
                strategy=self.occlusion_strategy,
                seed=self.seed + self.epoch * 97 + index,
            )
        return (
            torch.from_numpy(features).transpose(0, 1).contiguous(),
            torch.from_numpy(labels),
        )
