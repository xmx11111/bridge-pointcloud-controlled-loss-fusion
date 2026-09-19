from pathlib import Path

import numpy as np

from bridge_seg.data import (
    audit_prepared_dataset,
    class_balanced_probabilities,
    load_scene_npz,
    sample_block,
)


def write_scene(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.stack(
        [
            np.arange(len(labels), dtype=np.float32),
            np.zeros(len(labels), dtype=np.float32),
            labels.astype(np.float32),
        ],
        axis=1,
    )
    np.savez(
        path,
        xyz=xyz,
        rgb=np.full_like(xyz, 0.5),
        labels=labels.astype(np.int64),
        center=xyz.mean(axis=0),
        extent=np.ptp(xyz, axis=0),
        scale=np.asarray([max(np.ptp(xyz, axis=0))], dtype=np.float32),
    )


def test_load_scene_and_sample_block(tmp_path: Path) -> None:
    scene_path = tmp_path / "c-f-bridge1_s1.npz"
    write_scene(scene_path, np.asarray([0, 1, 2, 0, 1, 2]))

    scene = load_scene_npz(scene_path)
    features, labels = sample_block(
        scene,
        points_per_block=4,
        rng=np.random.default_rng(42),
        sampling_mode="class_balanced",
    )

    assert scene.source_type == "virtual"
    assert scene.bridge_family == "c-bridge1"
    assert features.shape == (4, 9)
    assert labels.shape == (4,)


def test_class_balanced_probabilities_favor_rare_classes() -> None:
    labels = np.asarray([0] * 100 + [1] * 10 + [2])

    probabilities = class_balanced_probabilities(labels)

    assert probabilities.shape == (3,)
    assert probabilities[2] > probabilities[1] > probabilities[0]
    assert np.isclose(probabilities.sum(), 1.0)


def test_audit_prepared_dataset(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        write_scene(
            tmp_path / split / f"c-bridge1_{split}.npz",
            np.asarray([0, 1, 2, 2]),
        )

    audit = audit_prepared_dataset(tmp_path)

    assert audit["splits"]["train"]["scene_count"] == 1
    assert audit["splits"]["test"]["class_point_counts"] == {
        "girder": 1,
        "pier": 1,
        "deck": 2,
    }
