from pathlib import Path

import numpy as np
import torch

from bridge_seg.dataset import PreparedSceneBlockDataset

from .test_data import write_scene


def test_prepared_scene_dataset_returns_channel_first_tensors(tmp_path: Path) -> None:
    path = tmp_path / "c-bridge1.npz"
    write_scene(path, np.asarray([0, 1, 2, 0]))
    dataset = PreparedSceneBlockDataset(
        [path],
        points_per_block=4,
        blocks_per_scene=2,
        seed=42,
        sampling_mode="class_balanced",
        occlusion_rate=0.25,
        occlusion_strategy="random",
    )

    features, labels = dataset[0]

    assert features.shape == (9, 3)
    assert labels.shape == (3,)
    assert len(dataset) == 2


def test_prepared_scene_dataset_returns_mirrored_pair(tmp_path: Path) -> None:
    path = tmp_path / "c-bridge1.npz"
    write_scene(path, np.asarray([0, 1, 2, 0, 1, 2]))
    dataset = PreparedSceneBlockDataset(
        [path],
        points_per_block=4,
        blocks_per_scene=1,
        seed=42,
        sampling_mode="class_balanced",
        symmetry_mode="consistency",
    )

    features, labels, mirrored_features, eligible = dataset[0]

    assert features.shape == (9, 4)
    assert mirrored_features.shape == (9, 4)
    assert labels.shape == (4,)
    assert eligible.shape == (1,)
    if eligible.item() == 1.0:
        assert not torch.equal(features, mirrored_features)
