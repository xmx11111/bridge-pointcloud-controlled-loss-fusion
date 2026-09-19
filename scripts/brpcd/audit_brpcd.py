from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def count_lines(path: Path, chunk_size: int = 8 * 1024 * 1024) -> tuple[int, bool, int]:
    count = 0
    last_byte = b""
    first_line = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            if not first_line:
                newline = chunk.find(b"\n")
                first_line = chunk if newline < 0 else chunk[:newline]
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    newline_terminated = last_byte == b"\n"
    if last_byte and not newline_terminated:
        count += 1
    fields = len(first_line.split()) if first_line else 0
    return count, newline_terminated, fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scene_dir = args.root / "girder bridge"
    if not scene_dir.is_dir():
        raise SystemExit(f"Missing scene directory: {scene_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scene in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
        annotation_dir = scene / "Annotations"
        if not annotation_dir.is_dir():
            raise SystemExit(f"Missing annotations: {annotation_dir}")
        for file_path in sorted(annotation_dir.glob("*.txt")):
            component = file_path.stem.split("_", 1)[0]
            point_count, newline_terminated, field_count = count_lines(file_path)
            rows.append(
                {
                    "scene": scene.name,
                    "component": component,
                    "file": str(file_path.relative_to(args.root)),
                    "bytes": file_path.stat().st_size,
                    "points": point_count,
                    "fields_in_first_line": field_count,
                    "newline_terminated": newline_terminated,
                }
            )
            print(f"{scene.name}/{file_path.name}: {point_count:,}", flush=True)

    manifest_path = args.output_dir / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_component = Counter()
    by_scene = Counter()
    bytes_by_component = Counter()
    for row in rows:
        points = int(row["points"])
        size = int(row["bytes"])
        component = str(row["component"])
        scene = str(row["scene"])
        by_component[component] += points
        by_scene[scene] += points
        bytes_by_component[component] += size

    summary = {
        "root": str(args.root),
        "scene_count": len(by_scene),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "total_points": sum(int(row["points"]) for row in rows),
        "points_by_component": dict(sorted(by_component.items())),
        "bytes_by_component": dict(sorted(bytes_by_component.items())),
        "points_by_scene": dict(sorted(by_scene.items())),
    }
    summary_path = args.output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
