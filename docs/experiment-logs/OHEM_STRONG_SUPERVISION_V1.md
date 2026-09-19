# OHEM Strong-Supervision Loss Results v1

Date: 2026-09-18

## Protocol

- Dataset: BrPCD `prepared_20k_v2`, P1 split, test family `c-bridge4`
- Seeds: 42, 43, 44
- Training: 60 epochs, 1024 points per block, class-balanced sampling,
  topology weight 0.5, identical data pipeline and evaluation rates
  (0 / 25 / 50 / 75 % missing) for both arms
- Only the segmentation loss differs between the two arms

New arm (`ohem_weighted`), ported from the arch-bridge strong-supervision
module:

1. class weights are recomputed from the valid points of each batch as
   inverse frequencies (no hard-coded class count or weight array);
2. only the hardest 20 % of valid points are kept (`ohem_ratio=0.2`,
   `min_ohem_points=100`, automatic fallback to all points below that).

Control arm (`focal_weighted`) is the existing mainline recipe.

## Mean Results (3 seeds)

| Arm | 0 % | 25 % | 50 % | 75 % |
|-----|-----|------|------|------|
| focal weight (control) | 0.4827 | 0.3952 | 0.3204 | 0.2468 |
| weighted CE + OHEM | 0.4036 | 0.3503 | 0.2943 | 0.2554 |

Per-class IoU:

| Arm | rate | girder | pier | deck |
|-----|------|--------|------|------|
| focal weight | 0 % | 0.4037 | 0.8508 | 0.1934 |
| focal weight | 75 % | 0.2541 | 0.4803 | 0.0058 |
| weighted CE + OHEM | 0 % | **0.0044** | 0.8317 | **0.3748** |
| weighted CE + OHEM | 75 % | **0.0206** | 0.7054 | 0.0403 |

## Paired Per-Seed Difference (OHEM - control)

Both arms share the same seed, and therefore the same test block sampling,
so each row is a matched comparison.

| Seed | 0 % | 25 % | 50 % | 75 % |
|------|-----|------|------|------|
| 42 | -0.0777 | -0.0741 | -0.1159 | -0.1392 |
| 43 | +0.0668 | +0.0724 | +0.0530 | +0.0663 |
| 44 | -0.2263 | -0.1332 | -0.0154 | +0.0990 |
| mean | -0.0791 | -0.0450 | -0.0261 | +0.0087 |

## What Actually Happens

The OHEM arm does not simply lose accuracy; it changes which component the
model is willing to predict. Across all three seeds the `girder` IoU
collapses to essentially zero (0.0131 / 0.0000 / 0.0000) while `deck`
roughly doubles (0.3618 / 0.4327 / 0.3297 against 0.1141 / 0.0000 / 0.4662).

The mechanism is visible in the loss construction. Batch-frequency inverse
weighting multiplies the rarest class by the largest factor, and OHEM then
keeps only the hardest points, which after reweighting are dominated by that
same rare class. Under a 60-epoch budget the model resolves the conflict by
abandoning `girder` rather than by improving the component hierarchy.

Isolating the two components at seed 42 confirms that each one costs accuracy
on its own:

| Variant | 0 % | 25 % | 50 % | 75 % | deck IoU (0 %) |
|---------|-----|------|------|------|----------------|
| focal weight (control) | 0.4767 | 0.4481 | 0.4227 | 0.4006 | 0.1141 |
| inverse-frequency CE only (`ohem_ratio=1.0`) | 0.4260 | 0.3563 | 0.2991 | 0.2280 | 0.0000 |
| inverse-frequency CE + OHEM (`ohem_ratio=0.2`) | 0.2551 | 0.2446 | 0.2457 | 0.2441 | 0.0055 |

The control arm reproduces the archived baseline checkpoint exactly
(`0.4767 / 0.4481 / 0.4227 / 0.4006`), so the new loss path does not perturb
the existing recipe.

## Robustness to the Topology Weight

The comparison above uses `topology_weight=0.5`. A seed-42 sanity check at
`topology_weight=1.0`, which is the weight quoted in the archived main table,
gives the same direction:

| Variant (seed 42, 20 epochs) | 0 % | 25 % | 50 % | 75 % |
|------------------------------|-----|------|------|------|
| focal weight, weight 1.0 | 0.5706 | 0.4235 | 0.3417 | 0.2667 |
| weighted CE + OHEM, weight 1.0 | 0.2746 | 0.2297 | 0.2132 | 0.1981 |

The negative conclusion is therefore not an artefact of the topology weight
choice.

## Decision

- Do not adopt the OHEM strong-supervision loss as a drop-in improvement on
  this dataset: it is worse at 0 %, 25 % and 50 % missing, and its only
  advantage (75 % missing, +0.0087 mean) is smaller than the seed spread.
- Do not report it as a component-segmentation method. It trades the
  majority component away and therefore fails the paper's own acceptance
  criterion, which requires all three components to stay usable.
- Keep it as a documented negative result and as an optional
  minority-class-recall variant if a future task cares about `deck` at the
  cost of `girder`.
- Seed variance remains large in this split (control clean mIoU spans
  0.3768 to 0.5945 across seeds 42-44). Any future loss comparison needs the
  paired, same-seed protocol used here.

## Code Changes

- `src/bridge_seg/losses.py`: `StrongSupervisionLoss`, `LOSS_NAMES` entry
  `ohem_weighted`, `build_segmentation_loss` options
- `src/bridge_seg/train.py`: `ohem_ratio` / `ohem_min_points` pass-through and
  configuration recording
- `src/bridge_seg/cli.py`: `--ohem-ratio`, `--ohem-min-points`
- `tests/test_losses.py`: five new unit tests for the loss

## Artifacts

- `runs/seg_tuning_topo05_e60_seed42|43|44` (control)
- `runs/seg_tuning_ohem_e60_seed42|43|44` (OHEM arm)
- `runs/seg_tuning_topo05_seed42_ctrl` (reproducibility check)
- `runs/seg_tuning_ohem100_seed42` (weighted CE without OHEM)
