import json
from pathlib import Path

import numpy as np

from bridge_seg.train import train_model

from .test_data import write_scene


def test_training_smoke_writes_artifacts(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0])
    for split in ("train", "val", "test"):
        write_scene(prepared_root / split / "c-bridge1.npz", labels)

    output_dir = tmp_path / "run"
    metrics = train_model(
        prepared_root=prepared_root,
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        points_per_block=8,
        train_blocks_per_scene=1,
        eval_blocks_per_scene=1,
        test_blocks_per_scene=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=42,
        device_name="cpu",
        sampling_mode="class_balanced",
        loss_name="focal_weighted",
        focal_gamma=2.0,
        occlusion_rate=0.25,
        occlusion_strategy="random",
        eval_occlusion_rates=(0.0, 0.25),
        eval_occlusion_strategy="viewpoint",
    )

    assert (output_dir / "best_model.pt").is_file()
    assert json.loads((output_dir / "metrics.json").read_text())["test"]["points"] > 0
    assert metrics["test_by_occlusion"]["0.25"]["points"] > 0
    assert metrics["history"]


def test_training_smoke_with_symmetry_consistency(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0])
    for split in ("train", "val", "test"):
        write_scene(prepared_root / split / "c-bridge1.npz", labels)

    output_dir = tmp_path / "run"
    metrics = train_model(
        prepared_root=prepared_root,
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        points_per_block=8,
        train_blocks_per_scene=1,
        eval_blocks_per_scene=1,
        test_blocks_per_scene=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=42,
        device_name="cpu",
        sampling_mode="class_balanced",
        loss_name="focal_weighted",
        focal_gamma=2.0,
        occlusion_rate=0.25,
        occlusion_strategy="random",
        eval_occlusion_rates=(0.0, 0.25),
        eval_occlusion_strategy="viewpoint",
        symmetry_mode="consistency",
        symmetry_weight=1.0,
        symmetry_warmup_epochs=1,
    )

    assert metrics["configuration"]["symmetry_mode"] == "consistency"
    assert metrics["history"][0]["symmetry_weight"] == 1.0
    assert "train_symmetry_loss" in metrics["history"][0]


def test_training_smoke_with_cross_side_propagation(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0])
    for split in ("train", "val", "test"):
        write_scene(prepared_root / split / "c-bridge1.npz", labels)

    metrics = train_model(
        prepared_root=prepared_root,
        output_dir=tmp_path / "run",
        epochs=1,
        batch_size=2,
        points_per_block=8,
        train_blocks_per_scene=1,
        eval_blocks_per_scene=1,
        test_blocks_per_scene=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=42,
        device_name="cpu",
        sampling_mode="class_balanced",
        loss_name="focal_weighted",
        focal_gamma=2.0,
        occlusion_rate=0.25,
        occlusion_strategy="random",
        eval_occlusion_rates=(0.0, 0.25),
        eval_occlusion_strategy="viewpoint",
        symmetry_mode="cross_side",
        symmetry_weight=1.0,
        symmetry_warmup_epochs=1,
    )

    assert metrics["configuration"]["symmetry_mode"] == "cross_side"
    assert metrics["history"][0]["symmetry_weight"] == 1.0
    assert "train_symmetry_loss" in metrics["history"][0]


def test_training_smoke_with_geometry_only_mirror_tta(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0])
    for split in ("train", "val", "test"):
        write_scene(prepared_root / split / "c-bridge1.npz", labels)

    metrics = train_model(
        prepared_root=prepared_root,
        output_dir=tmp_path / "run",
        epochs=1,
        batch_size=2,
        points_per_block=8,
        train_blocks_per_scene=1,
        eval_blocks_per_scene=1,
        test_blocks_per_scene=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=42,
        device_name="cpu",
        sampling_mode="class_balanced",
        loss_name="focal_weighted",
        focal_gamma=2.0,
        occlusion_rate=0.0,
        occlusion_strategy="random",
        eval_occlusion_rates=(0.0, 0.25),
        eval_occlusion_strategy="viewpoint",
        mirror_tta=True,
        mirror_tta_weight_mode="geometry",
    )

    assert metrics["configuration"]["mirror_tta"] is True
    assert metrics["configuration"]["mirror_tta_weight_mode"] == "geometry"
    assert metrics["test"]["points"] > 0
    assert metrics["test_by_occlusion"]["0.25"]["points"] > 0


def test_training_smoke_with_confidence_gated_mirror_tta(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0])
    for split in ("train", "val", "test"):
        write_scene(prepared_root / split / "c-bridge1.npz", labels)

    metrics = train_model(
        prepared_root=prepared_root,
        output_dir=tmp_path / "run",
        epochs=1,
        batch_size=2,
        points_per_block=8,
        train_blocks_per_scene=1,
        eval_blocks_per_scene=1,
        test_blocks_per_scene=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=42,
        device_name="cpu",
        sampling_mode="class_balanced",
        loss_name="focal_weighted",
        focal_gamma=2.0,
        occlusion_rate=0.0,
        occlusion_strategy="random",
        eval_occlusion_rates=(0.0, 0.25),
        eval_occlusion_strategy="viewpoint",
        mirror_tta=True,
        mirror_tta_weight_mode="confidence",
        mirror_tta_confidence_threshold=0.7,
    )

    assert metrics["configuration"]["mirror_tta_weight_mode"] == "confidence"
    assert metrics["configuration"]["mirror_tta_confidence_threshold"] == 0.7
