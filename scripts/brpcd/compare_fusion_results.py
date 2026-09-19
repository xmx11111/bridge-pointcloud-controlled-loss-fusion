from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _scene_metrics(path: Path, method: str) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, np.ndarray] = {}
    for record in data["overall"]:
        scene = record.get("scene")
        if scene is None:
            continue
        for candidate in record["methods"]:
            if candidate["method"] == method:
                rows[scene] = np.asarray(
                    [
                        float(candidate["mean_iou"]),
                        *map(float, candidate["per_class_iou"]),
                    ],
                    dtype=np.float64,
                )
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--method", default="fusion_side_alpha_1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    left = _scene_metrics(args.left, args.method)
    right = _scene_metrics(args.right, args.method)
    scenes = sorted(set(left) & set(right))
    if not scenes:
        raise ValueError("No overlapping scenes")
    deltas = np.stack([right[scene] - left[scene] for scene in scenes])
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(deltas), size=(args.samples, len(deltas)))
    bootstrap = deltas[indices].mean(axis=1)
    result = {
        "left": str(args.left),
        "right": str(args.right),
        "method": args.method,
        "scene_count": len(scenes),
        "mean_delta": deltas.mean(axis=0).tolist(),
        "bootstrap_95_ci": [
            np.quantile(bootstrap[:, index], [0.025, 0.975]).tolist()
            for index in range(deltas.shape[1])
        ],
        "improved_scenes": int(np.sum(deltas[:, 0] > 0)),
        "scenes": [
            {
                "scene": scene,
                "delta_mIoU": float(right[scene][0] - left[scene][0]),
                "delta_girder": float(right[scene][1] - left[scene][1]),
                "delta_pier": float(right[scene][2] - left[scene][2]),
                "delta_deck": float(right[scene][3] - left[scene][3]),
            }
            for scene in scenes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "delta_mIoU=",
        round(result["mean_delta"][0], 4),
        "CI=",
        [round(value, 4) for value in result["bootstrap_95_ci"][0]],
        "improved=",
        result["improved_scenes"],
        "/",
        result["scene_count"],
    )
    print(args.output)


if __name__ == "__main__":
    main()
