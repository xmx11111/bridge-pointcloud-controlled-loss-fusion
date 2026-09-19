from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import audit_prepared_dataset
from .data import SAMPLING_MODES, SYMMETRY_MODES
from .losses import LOSS_NAMES
from .occlusion import OCCLUSION_STRATEGIES
from .smoke import run_smoke
from .train import train_model


def parse_occlusion_rates(value: str) -> tuple[float, ...]:
    rates: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            rate = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid occlusion rate: {part!r}"
            ) from exc
        if not 0.0 <= rate < 1.0:
            raise argparse.ArgumentTypeError(
                "Occlusion rates must be in [0, 1)"
            )
        if rate not in rates:
            rates.append(rate)
    if not rates:
        raise argparse.ArgumentTypeError("At least one occlusion rate is required")
    return tuple(sorted(rates))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bridge-seg",
        description="Bridge point-cloud segmentation framework commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run a synthetic model smoke test.",
    )
    smoke.add_argument(
        "--output",
        type=Path,
        default=Path("runs/framework_smoke.json"),
    )
    smoke.add_argument("--seed", type=int, default=42)

    audit = subparsers.add_parser(
        "audit",
        help="Audit a prepared train/val/test BrPCD dataset.",
    )
    audit.add_argument("--prepared-root", type=Path, required=True)
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("runs/brpcd_audit.json"),
    )

    train = subparsers.add_parser(
        "train",
        help="Train the lightweight PointNet-style baseline.",
    )
    train.add_argument("--prepared-root", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--points-per-block", type=int, default=1024)
    train.add_argument("--train-blocks-per-scene", type=int, default=8)
    train.add_argument("--eval-blocks-per-scene", type=int, default=16)
    train.add_argument("--test-blocks-per-scene", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default="class_balanced",
    )
    train.add_argument(
        "--loss",
        choices=LOSS_NAMES,
        default="ce_weighted",
    )
    train.add_argument("--focal-gamma", type=float, default=2.0)
    train.add_argument(
        "--ohem-ratio",
        type=float,
        default=0.2,
        help="Fraction of hardest points kept by the ohem_weighted loss.",
    )
    train.add_argument(
        "--ohem-min-points",
        type=int,
        default=100,
        help="Skip OHEM filtering when fewer valid points are in the batch.",
    )
    train.add_argument("--occlusion-rate", type=float, default=0.0)
    train.add_argument(
        "--occlusion-strategy",
        choices=OCCLUSION_STRATEGIES,
        default="random",
    )
    train.add_argument(
        "--eval-occlusion-rates",
        type=parse_occlusion_rates,
        default=(0.0,),
        help="Comma-separated test-time missing rates, e.g. 0,0.25,0.5,0.75.",
    )
    train.add_argument(
        "--eval-occlusion-strategy",
        choices=OCCLUSION_STRATEGIES,
        default="viewpoint",
    )
    train.add_argument(
        "--symmetry-mode",
        choices=SYMMETRY_MODES,
        default="none",
    )
    train.add_argument("--symmetry-weight", type=float, default=1.0)
    train.add_argument("--symmetry-warmup-epochs", type=int, default=5)
    train.add_argument("--topology-weight", type=float, default=0.0)
    train.add_argument(
        "--topology-mass-weighting",
        action="store_true",
    )
    train.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    train.add_argument(
        "--backbone",
        choices=("pointnet", "pointnet++"),
        default="pointnet",
    )
    train.add_argument(
        "--mirror-tta",
        action="store_true",
        help="Fuse predictions from geometry-only transverse mirrored blocks.",
    )
    train.add_argument(
        "--mirror-tta-weight",
        choices=("none", "fixed", "geometry", "random", "confidence"),
        default="fixed",
        help="Fusion weighting or gating strategy for mirror TTA.",
    )
    train.add_argument(
        "--mirror-tta-geometry-evidence-threshold",
        type=float,
        default=0.0,
    )
    train.add_argument(
        "--mirror-tta-confidence-threshold",
        type=float,
        default=0.7,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "smoke":
        payload = run_smoke(args.output, seed=args.seed)
    elif args.command == "audit":
        payload = audit_prepared_dataset(args.prepared_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif args.command == "train":
        payload = train_model(
            prepared_root=args.prepared_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            points_per_block=args.points_per_block,
            train_blocks_per_scene=args.train_blocks_per_scene,
            eval_blocks_per_scene=args.eval_blocks_per_scene,
            test_blocks_per_scene=args.test_blocks_per_scene,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device_name=args.device,
            sampling_mode=args.sampling_mode,
            loss_name=args.loss,
            focal_gamma=args.focal_gamma,
            ohem_ratio=args.ohem_ratio,
            ohem_min_points=args.ohem_min_points,
            occlusion_rate=args.occlusion_rate,
            occlusion_strategy=args.occlusion_strategy,
            eval_occlusion_rates=args.eval_occlusion_rates,
            eval_occlusion_strategy=args.eval_occlusion_strategy,
            symmetry_mode=args.symmetry_mode,
            symmetry_weight=args.symmetry_weight,
            symmetry_warmup_epochs=args.symmetry_warmup_epochs,
            topology_weight=args.topology_weight,
            topology_mass_weighting=args.topology_mass_weighting,
            backbone=args.backbone,
            mirror_tta=args.mirror_tta,
            mirror_tta_weight_mode=args.mirror_tta_weight,
            mirror_tta_geometry_evidence_threshold=(
                args.mirror_tta_geometry_evidence_threshold
            ),
            mirror_tta_confidence_threshold=(
                args.mirror_tta_confidence_threshold
            ),
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
