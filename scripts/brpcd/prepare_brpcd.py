from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") ^ base


def sample_file(
    path: Path, point_count: int, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(point_count, sample_count, replace=False))
    selected_set = set(int(value) for value in selected)

    xyz_rows: list[np.ndarray] = []
    rgb_rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle):
            if line_index not in selected_set:
                continue
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size != 6:
                raise ValueError(f"Expected 6 fields in {path}:{line_index + 1}")
            xyz_rows.append(values[:3])
            rgb_rows.append(values[3:6])

    if len(xyz_rows) != sample_count:
        raise RuntimeError(
            f"Expected {sample_count} rows from {path}, found {len(xyz_rows)}"
        )
    return np.stack(xyz_rows), np.stack(rgb_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points-per-file", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.split_manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["scene"])].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for (split, scene), scene_rows in sorted(grouped.items()):
        all_xyz: list[np.ndarray] = []
        all_rgb: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for row in sorted(scene_rows, key=lambda item: item["file"]):
            point_count = int(row["points"])
            sample_count = min(point_count, args.max_points_per_file)
            seed = stable_seed(
                args.seed, split, scene, row["component"], row["file"]
            )
            xyz, rgb = sample_file(
                args.dataset_root / row["file"], point_count, sample_count, seed
            )
            labels = np.full(
                sample_count, int(row["component_id"]), dtype=np.int64
            )
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_labels.append(labels)
            records.append(
                {
                    "split": split,
                    "scene": scene,
                    "component": row["component"],
                    "source_file": row["file"],
                    "source_points": point_count,
                    "sampled_points": sample_count,
                }
            )
            print(
                f"{split}/{scene}/{row['component']}: "
                f"{sample_count:,}/{point_count:,}",
                flush=True,
            )

        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)

        center = xyz.mean(axis=0, dtype=np.float64).astype(np.float32)
        extent = np.ptp(xyz, axis=0).astype(np.float32)
        scale = float(np.max(extent))
        if scale <= 0:
            scale = 1.0

        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / f"{scene}.npz"
        np.savez(
            output_path,
            xyz=xyz.astype(np.float32),
            rgb=rgb.astype(np.float32),
            labels=labels.astype(np.int64),
            center=center,
            extent=extent,
            scale=np.asarray([scale], dtype=np.float32),
        )

    record_path = args.output_dir / "prepared_files.csv"
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "seed": args.seed,
        "max_points_per_file": args.max_points_per_file,
        "split_manifest": str(args.split_manifest),
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "scene_files": int(len(grouped)),
        "sampled_points": int(sum(int(row["sampled_points"]) for row in records)),
        "sampled_points_by_split": {
            split: sum(
                int(row["sampled_points"])
                for row in records
                if row["split"] == split
            )
            for split in sorted({str(row["split"]) for row in records})
        },
        "sampled_points_by_component": {
            component: sum(
                int(row["sampled_points"])
                for row in records
                if row["component"] == component
            )
            for component in sorted({str(row["component"]) for row in records})
        },
    }
    (args.output_dir / "prepared_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
