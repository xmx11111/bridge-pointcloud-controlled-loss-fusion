# 2D-3D Side-View Fusion on the Main Table Recipe v1

Date: 2026-09-18

## Why This Check Exists

The archived 2D-3D report quotes a gain of `0.085` point-level mIoU
(`0.4767 -> 0.5620`). That number was measured against
`runs/seg_tuning_topo05_seed42`, a `topology_weight=0.5` single-seed
checkpoint, while the paper's main table uses `topology_weight=1.0` seeds
42-44. The gain therefore had to be re-measured on the main-table recipe.

## Protocol

- 3D backbone: `runs/topology_v1/topology_order_balanced_focal_seed{42,43,44}`
  (`topology_weight=1.0`, main-table checkpoints)
- 2D branch: `runs/projection_geom_512_radius0_unet_seed42`
- Projections: `projection_geom_512_v6_radius0`
- Same evaluator, same 360,448 sampled test points, sampling seed 2042,
  32 blocks per scene, so the 3D and fused rows are paired
- Fusion rule: `argmax(P_3D + alpha * P_2D)`, only Z-buffer visible points
  receive 2D evidence, invisible points fall back to the 3D prediction

## Results (mean over seeds 42-44)

Side view only (visible fraction 21.41 %):

| Method | mIoU | girder | pier | deck |
|--------|------|--------|------|------|
| PointNet (main table) | 0.5103 | 0.4362 | 0.8494 | 0.2452 |
| 2D side, visible points only | 0.6253 | 0.4869 | 0.8858 | 0.5033 |
| fusion, alpha = 0.25 | **0.5488** | 0.4172 | 0.8715 | 0.3578 |
| fusion, alpha = 1 | 0.5470 | 0.4136 | 0.8712 | 0.3561 |

Per-seed paired difference for `alpha = 0.25`:
`+0.0349 / +0.0487 / +0.0320`, mean `+0.0385`, all three seeds positive.

Side plus left yaw (visible fraction 42.86 %):

| Method | mIoU | girder | pier | deck |
|--------|------|--------|------|------|
| PointNet (main table) | 0.5103 | 0.4362 | 0.8494 | 0.2452 |
| fusion, alpha = 0.25 | **0.5501** | 0.4013 | 0.8669 | 0.3821 |
| fusion, alpha = 1 | 0.5453 | 0.3921 | 0.8634 | 0.3803 |

Per-seed paired difference for `alpha = 0.25`:
`+0.0221 / +0.0461 / +0.0513`, mean `+0.0398`, all three seeds positive.

## What Changes, What Does Not

- The direction survives: 2D-3D side fusion still improves the main-table
  recipe, and every seed is positive at both view settings.
- The size shrinks from `+0.085` to about `+0.038`. The earlier headline was
  inflated by a weaker 3D baseline, not by a different fusion rule.
- The absolute best single number also improves: seed 42 with `side`,
  `alpha = 0.25` reaches `0.6055`, above the previous best `0.5706`.
- The class trade-off is the same one seen elsewhere in this project:
  `deck` IoU rises from `0.2452` to `0.3578-0.3821` while `girder` falls
  from `0.4362` to `0.4013-0.4172`. The gain is not uniform across
  components and must be reported per class.
- The 2D branch alone scores `0.6253` on the points it can see, with
  `deck` IoU `0.5033`. Two thirds of the fused-model error on `deck` comes
  from the three quarters of points that no side view can see.

## Reporting Rule

Quote the fusion gain against the same recipe as the rest of the table:

> On the main-table recipe, side-view 2D-3D fusion raises mean point-level
> mIoU from 0.5103 to 0.5488 (alpha = 0.25, three seeds, all positive),
> with the gain concentrated in `deck`.

Do not quote `+0.085`, and do not compare `0.5620` against the `0.5103`
main-table baseline; the two numbers come from different 3D checkpoints.

## Artifacts

- `outputs/2D-3D_topo1_512_side_seed{42,43,44}.json`
- `outputs/2D-3D_topo1_512_side_left_seed{42,43,44}.json`
- evaluator: `C:\Users\肖\Documents\Codex\2026-09-18\ji-2\work\brpcd\evaluate_geometry_2d3d_fusion.py`

## Occlusion Sweep

The evaluator now takes `--missing-rate` and `--occlusion-strategy` and applies
`bridge_seg.occlusion` per block. The per-block seed does not depend on the
rate, so the removed sets are nested and the rates stay comparable.

Paired results on the same 360448 / 270336 / 180224 / 90112 points. The 3D and
2D checkpoints share a seed (seed 42 with seed 42, and so on), so the numbers
below use three independent 2D models rather than one shared model:

| View | Row | 0 % | 25 % | 50 % | 75 % |
|------|-----|-----|------|------|------|
| side | topology only (paired) | 0.5103 | 0.4303 | 0.3629 | 0.2700 |
| side | + 2D, alpha = 0.25 | 0.5508 | 0.4850 | 0.4251 | 0.3450 |
| side + side_left | topology only (paired) | 0.5103 | 0.4303 | 0.3629 | 0.2700 |
| side + side_left | + 2D, alpha = 0.25 | 0.5488 | 0.4988 | 0.4544 | 0.3945 |

Paired gain of the fusion over its own baseline:

| View | 0 % | 25 % | 50 % | 75 % |
|------|-----|------|------|------|
| side, alpha = 0.25 | +0.0405 | +0.0547 | +0.0622 | +0.0750 |
| side + side_left, alpha = 0.25 | +0.0385 | +0.0685 | +0.0915 | +0.1245 |

This is the strongest form of the result so far: the fusion advantage grows
with the missing rate instead of staying flat, which is the behaviour the
2D evidence is supposed to produce, and it is largest for the two-view
configuration.

### 2D Seed Robustness

The 2D branch was originally trained on seed 42 only, and every fused row reused
that single model. Seeds 43 and 44 were trained afterwards with the identical
recipe and land at comparable quality:

| 2D seed | best val pixel mIoU | test pixel mIoU | girder | pier | deck |
|---------|--------------------|-----------------|--------|------|------|
| 42 | 0.7459 | 0.5023 | 0.4006 | 0.7785 | 0.3279 |
| 43 | 0.7332 | 0.4816 | 0.4301 | 0.7052 | 0.3096 |
| 44 | 0.7408 | 0.4620 | 0.3775 | 0.6756 | 0.3330 |

Re-running the whole sweep with matched 3D/2D seeds moves every fused row by at
most 0.003 (`side` 0.5488 -> 0.5508 at 0 %, 0.4998 -> 0.4988 for
`side + side_left` at 25 %). The fusion gain is therefore not a property of one
lucky 2D checkpoint.

## Fixed Bug: Occlusion Order

The first version of the sweep computed the 3D prediction on the full block and
then scored only the surviving points. That inflated every non-zero rate
(0.5173 at 25 % missing instead of 0.4303) because occlusion by construction
removes the peripheral, hardest points.

Occlusion is now applied before the forward pass. The evaluator then
reproduces the training pipeline exactly: seed 42 at 25 % missing gives
`0.42346` in both, on the same 270336 points.
