from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABEL_MAP = {
    4: 0,  # superstructure -> girder
    8: 1,  # pillar -> pier
    3: 1,  # abutment -> support / pier
    5: 2,  # top surface -> deck
}


def read_ply_layout(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as handle:
        lines: list[str] = []
        while True:
            line = handle.readline().decode("ascii", errors="replace").strip()
            lines.append(line)
            if line == "end_header":
                break
    count = int(
        next(line.split()[2] for line in lines if line.startswith("element points "))
    )
    color_count = int(
        next(line.split()[2] for line in lines if line.startswith("element color "))
    )
    label_count = int(
        next(line.split()[2] for line in lines if line.startswith("element label "))
    )
    if color_count != count or label_count != count:
        raise ValueError(f"Unexpected PLY element sizes in {path}")
    point_offset = path.stat().st_size - count * (12 + 3 + 1)
    return count, point_offset, point_offset + count * 12, point_offset + count * 15


def load_scene(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, point_offset, color_offset, label_offset = read_ply_layout(path)
    xyz = np.fromfile(path, dtype=np.float32, count=count * 3, offset=point_offset).reshape(
        count, 3
    )
    rgb = np.fromfile(path, dtype=np.uint8, count=count * 3, offset=color_offset).reshape(
        count, 3
    )
    labels = np.fromfile(path, dtype=np.uint8, count=count, offset=label_offset)
    return xyz, rgb, labels


def sample_balanced(labels: np.ndarray, points_per_scene: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    per_class = max(points_per_scene // 3, 1)
    for class_id in range(3):
        indices = np.flatnonzero(labels == class_id)
        if len(indices):
            selected.append(rng.choice(indices, min(per_class, len(indices)), replace=False))
    chosen = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if len(chosen) < points_per_scene:
        remaining = np.setdiff1d(np.arange(len(labels)), chosen)
        extra = min(points_per_scene - len(chosen), len(remaining))
        if extra:
            chosen = np.concatenate([chosen, rng.choice(remaining, extra, replace=False)])
    rng.shuffle(chosen)
    return chosen[:points_per_scene]


def prepare_split(
    files: list[Path],
    output_dir: Path,
    points_per_scene: int,
    seed: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, path in enumerate(sorted(files)):
        xyz, rgb, raw_labels = load_scene(path)
        labels = np.full(len(raw_labels), -1, dtype=np.int64)
        for source_id, target_id in LABEL_MAP.items():
            labels[raw_labels == source_id] = target_id
        valid = labels >= 0
        xyz, rgb, labels = xyz[valid], rgb[valid], labels[valid]
        if len(xyz) == 0:
            continue
        chosen = sample_balanced(labels, points_per_scene, seed + index * 97)
        xyz, rgb, labels = xyz[chosen], rgb[chosen], labels[chosen]
        rgb = rgb.astype(np.float32) / 255.0
        center = xyz.mean(axis=0, dtype=np.float64).astype(np.float32)
        extent = np.ptp(xyz, axis=0).astype(np.float32)
        scale = np.asarray([max(float(extent.max()), 1e-6)], dtype=np.float32)
        output_path = output_dir / f"{path.stem}.npz"
        np.savez_compressed(
            output_path,
            xyz=xyz.astype(np.float32),
            rgb=rgb,
            labels=labels,
            center=center,
            extent=extent,
            scale=scale,
        )
        records.append(
            {
                "scene": path.stem,
                "source": str(path),
                "output": str(output_path),
                "sampled_points": int(len(labels)),
                "class_counts": np.bincount(labels, minlength=3).tolist(),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--points-per-scene", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_files = sorted((args.source_root / "train").glob("*.ply"))
    test_files = sorted(
        path
        for path in (args.source_root / "val").glob("*.ply")
        if path.stem.endswith(("_rtc", "_faro"))
    )
    validation_names = {"bridge_14_fr_rtc", "bridge_20_fr_rtc"}
    validation_files = [path for path in train_files if path.stem in validation_names]
    train_files = [path for path in train_files if path.stem not in validation_names]
    if len(train_files) != 13 or len(validation_files) != 2 or len(test_files) != 5:
        raise ValueError(
            f"Expected 13/2/5 split, found {len(train_files)}/"
            f"{len(validation_files)}/{len(test_files)}"
        )

    summary = {
        "label_map": LABEL_MAP,
        "splits": {
            "train": prepare_split(
                train_files,
                args.output_root / "train",
                args.points_per_scene,
                args.seed,
            ),
            "val": prepare_split(
                validation_files,
                args.output_root / "val",
                args.points_per_scene,
                args.seed + 1000,
            ),
            "test": prepare_split(
                test_files,
                args.output_root / "test",
                args.points_per_scene,
                args.seed + 2000,
            ),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "prepared_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
