# Reproduction Guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## 2. Prepare BrPCD

Obtain the official data from:

`https://github.com/wxiong12/SEU-Digital-Bridge`

Then inspect and run:

```bash
python scripts/brpcd/prepare_brpcd.py --help
python scripts/brpcd/audit_brpcd.py --help
```

Use the same train, validation, and test split as the manuscript.

## 3. Train the 3D Baseline

```bash
python scripts/brpcd/train_pointnet_baseline.py --help
```

Run seeds 42, 43, and 44. Keep the controlled missing-rate order and the same
block sampler for every method at a given rate.

## 4. Build Virtual Views and Train the 2D Branch

```bash
python scripts/brpcd/build_geometry_projection_v3.py --help
python scripts/brpcd/train_projection_unet_geometry.py --help
python scripts/brpcd/train_projection_unet_aligned.py --help
```

The 2D input is geometry-rendered. It is not a real RGB image.

## 5. Evaluate Fusion

```bash
python scripts/brpcd/evaluate_geometry_2d3d_fusion.py --help
python scripts/brpcd/evaluate_coverage_penalty.py --help
```

Run `side` and `side_left`, with the fusion weight settings reported in the
paper. Do not reuse a complete-cloud prediction after removing points.

## 6. Export Statistics and Tables

```bash
python scripts/brpcd/export_block_statistics.py --help
python scripts/analyze_block_statistics.py --help
python scripts/brpcd/build_coverage_penalty_table.py --help
python scripts/brpcd/build_main_table.py --help
python scripts/brpcd/build_semanticbridge_external_table.py --help
```

Compact outputs from these stages are retained in `outputs/` and
`results/raw/`.

## 7. Rebuild Figures

```bash
python scripts/brpcd/plot_main_and_external_figures.py --help
python scripts/brpcd/plot_fig1_and_fig3.py --help
python scripts/brpcd/export_fig5_cases.py --help
python scripts/brpcd/plot_fig5_cases.py --help
```

Generated publication files are under `outputs/figures/`.

## 8. SemanticBridge

```bash
python scripts/semanticbridge/prepare_semanticbridge.py --help
python scripts/brpcd/run_semanticbridge_external.py --help
```

SemanticBridge is used for trend-level external validation only. Do not
compare its absolute accuracy with BrPCD.
