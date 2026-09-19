# Main Table Reconciliation v1

Date: 2026-09-18

## Problem

Two different configurations have both been reported under the name
"topology training":

| Source | Run directory | topology_weight | epochs | 0 % mIoU |
|--------|---------------|-----------------|--------|----------|
| `PAPER_PLAN_V3.md` §4.1 row 1 | `runs/topology_v1/topology_order_balanced_focal_seed42|1
|44` | 1.0 | 20 | 0.5096 (3-seed mean) |
| 2026-09-18 re-run | `runs/seg_tuning_topo05_seed42` | 0.5 | 20 | 0.4767 (seed 42) |

The directory name `seg_tuning_topo05_*` states weight 0.5, while the archived
row that the plan quotes is weight 1.0. Writing both into one table without a
label would mix recipes.

## Verified Numbers

Archived weight-1.0 arm, recomputed from the stored per-seed metrics:

| rate | mIoU | girder | deck | per seed |
|------|------|--------|------|----------|
| 0 % | 0.5096 | 0.4339 | 0.2400 | 0.5706 / 0.5058 / 0.4525 |
| 25 % | 0.4235 | 0.3592 | 0.1475 | 0.4235 / 0.4692 / 0.3778 |
| 50 % | 0.3484 | 0.2971 | 0.0966 | 0.3417 / 0.3794 / 0.3241 |
| 75 % | 0.2617 | 0.2486 | 0.0410 | 0.2667 / 0.2909 / 0.2274 |

Matched weight-0.5 arm run today with the current code, seeds 42-44,
60 epochs:

| rate | mIoU | girder | pier | deck |
|------|------|--------|------|------|
| 0 % | 0.4827 | 0.4037 | 0.8508 | 0.1934 |
| 25 % | 0.3952 | 0.3370 | 0.7470 | 0.1016 |
| 50 % | 0.3204 | 0.2873 | 0.6317 | 0.0422 |
| 75 % | 0.2468 | 0.2541 | 0.4803 | 0.0058 |

The weight-0.5 arm reproduces the archived seed-42 checkpoint exactly at both
20 and 60 epochs (`0.4767 / 0.4481 / 0.4227 / 0.4006`), so the epoch budget is
not what separates the two rows.

## Consequence for the Paper

- Pin one topology weight for the main table and label it in the caption.
  Weight 1.0 gives the higher clean number; weight 0.5 is what the newer run
  directories use.
- Do not compare a weight-1.0 row against a weight-0.5 row in the same table.
- Any claim that depends on the difference between 0.5096 and 0.4767 is
  currently a recipe difference, not an experimental result.

## Related Finding

The OHEM strong-supervision comparison (`OHEM_STRONG_SUPERVISION_V1.md`) was
run at weight 0.5 with a matched control. A weight-1.0 sanity check at seed 42
reproduces the same direction: `0.2746 / 0.2297 / 0.2132 / 0.1981` against
`0.5706 / 0.4235 / 0.3417 / 0.2667`. The negative conclusion therefore does
not depend on the topology weight.

## 2D-3D Fusion Line

The archived 2D-3D report quotes `0.4767 -> 0.5620` (`+0.085`). The `0.4767`
baseline is `runs/seg_tuning_topo05_seed42`, which is the weight-0.5
single-seed checkpoint, not the weight-1.0 main-table baseline.

Re-measured on weight-1.0 seeds 42-44 with the same evaluator and the same
360,448 test points:

| Method | mIoU | girder | pier | deck |
|--------|------|--------|------|------|
| PointNet, main table | 0.5103 | 0.4362 | 0.8494 | 0.2452 |
| fusion, side, alpha = 0.25 | 0.5488 | 0.4172 | 0.8715 | 0.3578 |
| fusion, side + side_left, alpha = 0.25 | 0.5501 | 0.4013 | 0.8669 | 0.3821 |

The direction holds (all three seeds positive, mean `+0.0385` and `+0.0398`),
but the gain is roughly half the quoted headline. Details in
`GEOMETRY_2D3D_MAINLINE_CHECK_V1.md`.

## Summary of the Three Recipe Clashes

1. "topology" covers both weight 1.0 and weight 0.5 checkpoints.
2. The OHEM comparison is controlled at weight 0.5; the main table is
   weight 1.0.
3. The 2D-3D fusion headline is anchored on the weight-0.5 seed-42
   checkpoint while the rest of the table is weight 1.0, three seeds.

Before the tables are written, pin the topology weight and the seed set, then
restate every method against that single recipe.
