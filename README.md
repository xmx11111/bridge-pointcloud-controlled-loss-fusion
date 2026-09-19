# Bridge Point-Cloud 2D-3D Fusion under Controlled Point Loss

This repository accompanies a study on controlled point loss, virtual-view
2D-3D fusion, and failure boundaries for bridge point-cloud component
segmentation. The paper is evaluation-driven. It does not claim a new network
architecture, a new complex fusion operator, or state-of-the-art accuracy.

The repository preserves the code and compact result artifacts needed to
inspect or rebuild the reported tables and figures. Raw datasets, model
checkpoints, virtual environments, third-party source copies, and unrelated
experimental branches are intentionally excluded.

## Paper Scope

The release covers:

- A remove-then-forward protocol at 0%, 25%, 50%, and 75% missing points.
- A lightweight PointNet 3D baseline for girder, pier, and deck segmentation.
- Plain virtual-view 2D-3D fusion using projection, U-Net segmentation,
  Z-buffer back-projection, and probability summation.
- A training-time force-alignment constraint with a bounded interpretation.
- Surviving-point and coverage-penalised evaluation conventions.
- BrPCD main results and SemanticBridge external validation.
- Negative results for OHEM and a PointNet++ pilot.

Deleted scope, mirror-TTA, symmetry, and geometry-gate branches are not part of
the current paper or the recommended reproduction path.

## Repository Layout

```text
src/bridge_seg/              Core point-cloud segmentation package
tests/                       Framework tests
configs/                     Configuration files
scripts/                     Paper-related analysis and training entry points
scripts/brpcd/               BrPCD preparation, projection, fusion, and figures
scripts/semanticbridge/      SemanticBridge preparation
results/tables/              Compact result-table sources, if applicable
results/raw/                 Selected raw fusion JSON and case-study data
results/run_metrics/         Small JSON/CSV metric files from run directories
results/figures/             Publication figures, if copied here
outputs/                     Paper drafts, summary tables, figures, metadata
docs/                        Data, protocol, reproduction, and review notes
data/                        Data placement instructions only
```

The authoritative result summaries are under `outputs/`, including
`main_table_v1.md`, `block_paired_statistics_v1.md`,
`semanticbridge_external_table_v1.md`, and the aligned-fusion reports.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

On PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q
```

The code was developed with PyTorch 2.8 in a CPU/GPU-capable environment.
Training experiments used CUDA when available.

## Data

Raw data are not redistributed here.

### BrPCD

Use the official BrPCD repository:

`https://github.com/wxiong12/SEU-Digital-Bridge`

The project uses the official `prepared_20k_v2` split. Expected scene fields
are `xyz`, `rgb`, `labels`, `center`, `extent`, and `scale`. Labels are:

```text
girder = 0
pier = 1
deck = 2
```

### SemanticBridge

Use the dataset and code linked from:

`https://github.com/mvg-inatech/3d_bridge_segmentation`

The study uses the real TLS portion. Original class IDs are mapped as follows:

```text
superstructure (4)             -> girder
abutment (3), pillar (8)       -> pier
top surface (5)                -> deck
```

Ground, high vegetation, railing, traffic sign, and unlabelled points are
excluded.

Several author-side scripts contain default dataset paths from the original
machine. Use the corresponding command-line arguments or mirror the expected
directory layout. See `docs/DATA.md` and `docs/REPRODUCIBILITY.md`.

## Principal Entry Points

Run `--help` for each script because some arguments and defaults reflect the
author-side directory layout.

```bash
python scripts/brpcd/prepare_brpcd.py --help
python scripts/brpcd/train_pointnet_baseline.py --help
python scripts/brpcd/build_geometry_projection_v3.py --help
python scripts/brpcd/train_projection_unet_geometry.py --help
python scripts/brpcd/train_projection_unet_aligned.py --help
python scripts/brpcd/evaluate_geometry_2d3d_fusion.py --help
python scripts/brpcd/export_block_statistics.py --help
python scripts/analyze_block_statistics.py --help
python scripts/brpcd/build_main_table.py --help
python scripts/brpcd/plot_main_and_external_figures.py --help
python scripts/brpcd/plot_fig5_cases.py --help
```

The BridgeNetv2 adapter expects the official BridgeNetv2 model code at runtime.
The third-party source is not vendored in this repository.

## Controlled Point Loss

Points are removed before the model forward pass. The model never sees the
complete cloud when a missing rate greater than zero is evaluated. Metrics are
reported under two conventions:

- Surviving points: only points retained after removal are scored.
- Coverage-penalised: every point in the original reference block is scored,
  and removed points are counted as errors.

The difference is the coverage loss. The paper treats coverage-penalised
accuracy as the primary convention for robustness and retains surviving-point
accuracy as a companion. See `docs/CONTROLLED_POINT_LOSS.md`.

## Results and Evidence Boundaries

The release keeps both positive and negative evidence. In particular:

- Plain fusion improves BrPCD mIoU at all four tested missing rates under the
  surviving-point convention.
- The gain is concentrated in the deck class on BrPCD.
- Force alignment gives only a small, configuration-dependent gain at 0%-50%
  missing under `side_left`; its block-level intervals cross zero.
- The force-alignment gain does not reproduce stably on SemanticBridge.
- OHEM and the current PointNet++ pilot provide no stable improvement.

Do not interpret these artifacts as evidence for severe-sparsity
generalisation, SOTA performance, real-image fusion, or broad cross-bridge-type
transfer.

## Reproducibility Notes

See:

- `docs/DATA.md`
- `docs/CONTROLLED_POINT_LOSS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/RESULT_ARTIFACTS.md`
- `docs/EXCLUDED_ASSETS.md`
- `docs/LICENSE_STATUS.md`

The compact results can be inspected without retraining. Full end-to-end
reproduction requires the public datasets, compatible hardware, and
recreation of the training checkpoints.

## Citation and License

Author names:

```text
Mingxuan Xiao
Lidu Zhao
```

The final manuscript citation, article metadata, and repository URL remain
subject to journal submission. No license has been selected yet. Do not assume
permission to reuse, redistribute, or sublicense the code until the authors
add a license file.
