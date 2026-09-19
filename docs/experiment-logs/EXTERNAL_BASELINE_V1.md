# BridgeNetv2 External Baseline V1

Date: 2026-09-18

## Purpose

This block adds the external comparison requested by the paper plan. It adapts
the official BridgeNetv2 backbone to the project's P1 split, sampling budget,
loss budget, and occlusion protocol. It is an external baseline, not a claim
that the published BridgeNetv2 synthetic-data result has been reproduced.

## Protocol

- Dataset: `prepared_20k_v2`
- Test family: `c-bridge4`
- Seeds: 42, 43, 44
- Training: 20 epochs, batch size 8, 8 blocks per scene
- Evaluation: 32 blocks per scene at 0/25/50/75% missing points
- Input: the first three per-block normalized XYZ channels
- Loss: class-weighted focal NLL on BridgeNetv2's native log-probability output
- Occlusion: viewpoint point removal followed by forward inference
- Fixed cardinality: surviving points are repeated to 1024 points because
  BridgeNetv2's hierarchical kNN requires a fixed input size; predictions are
  sliced back to the original surviving points before metrics

## Implementation Fixes

The first formal attempt collapsed to `pier`. Two interface errors were found:

1. BridgeNetv2 already returns `log_softmax`, while the adapter passed that
   output into a raw-logit focal cross-entropy. This applied a second softmax.
2. Training and validation flattened `[B, C, N]` into `[-1, C]` before the
   loss, which mixed the class and point axes.

The corrected adapter computes weighted focal NLL directly from
log-probabilities and keeps the batch/class/point axes intact. After the fix,
all three seeds learned all three classes.

## Results

### Three-seed mIoU

| Method | 0% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| PointNet topology only | 0.5096 | 0.4235 | 0.3484 | 0.2617 |
| **BridgeNetv2 external** | **0.5076 +/- 0.0761** | **0.4515 +/- 0.0740** | **0.4185 +/- 0.0800** | **0.3726 +/- 0.0490** |
| Main fusion, side, alpha=1 | 0.5485 | 0.4827 | 0.4228 | 0.3429 |
| Main fusion, side+side_left, alpha=1 | 0.5411 | 0.4927 | 0.4493 | 0.3902 |

The external baseline is effectively tied with the PointNet control at 0%
missing (`-0.0020`) and becomes stronger as points are removed:
`+0.0280`, `+0.0701`, and `+0.1109` at 25/50/75%.

The main 2D-3D fusion remains better than BridgeNetv2 at every rate. The best
main row is side fusion at 0/25/50% and side+side_left at 75%. At 75% missing,
the margin of the best main row over BridgeNetv2 is `+0.0176` mIoU.

### Per-class IoU

| Condition | Metric | girder | pier | deck |
|---|---|---:|---:|---:|
| BridgeNetv2, 0% | mean +/- sd | 0.4485 +/- 0.0645 | 0.9088 +/- 0.0577 | 0.1655 +/- 0.1104 |
| BridgeNetv2, 75% | mean +/- sd | 0.3162 +/- 0.0214 | 0.7904 +/- 0.1149 | 0.0111 +/- 0.0176 |
| Main fusion, side+left, 75% | mean only | 0.2723 | 0.6143 | 0.2840 |

BridgeNetv2 is strongest on `pier`, but its `deck` IoU is much lower than the
main fusion result under severe missing points. The clean `deck` result is also
highly seed-dependent.

### Per-seed mIoU

| Seed | 0% | 25% | 50% | 75% |
|---:|---:|---:|---:|---:|
| 42 | 0.4198 | 0.3779 | 0.3452 | 0.3173 |
| 43 | 0.5545 | 0.4508 | 0.4064 | 0.3897 |
| 44 | 0.5485 | 0.5258 | 0.5038 | 0.4107 |

Seed 42 is the weakest clean seed, while seeds 43 and 44 are consistent. This
variability is why the seed-42-only pilot was not promoted to the formal table.

## Interpretation

The supported claim is narrow:

> BridgeNetv2 can be adapted to the same P1 split and training budget, but it
> does not beat the proposed 2D-3D fusion at any occlusion rate.

The result also shows that BridgeNetv2 is relatively robust to the controlled
point-removal protocol. Its loss advantage over the PointNet control grows with
missing rate, but it still trails the fused method. This is useful evidence
that the main result is not explained merely by comparing against a weak
PointNet control.

## Scope and Limitations

- The published BridgeNetv2 weights and synthetic masonry-bridge data are not
  used; this is a matched-data adaptation.
- Fixed-cardinality padding is necessary for this backbone. It repeats
  surviving points and is not identical to a model with a native mask channel.
- The result covers one P1 family and three seeds.
- The external baseline is an appendix comparison, not a new main-table row
  claim.

## Artifacts

- Script: `work/brpcd/train_bridgenetv2_baseline.py`
- Seed 42: `runs/bridgenetv2_baseline_v2_seed42/`
- Seed 43: `runs/bridgenetv2_baseline_v2_seed43/`
- Seed 44: `runs/bridgenetv2_baseline_v2_seed44/`
- Unified table: `work/main_table_v1.{md,json,csv}`
- Table builder: `work/brpcd/build_main_table.py`
