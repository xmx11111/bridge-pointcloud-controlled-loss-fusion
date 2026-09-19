# PointNet++ Pilot V1

Date: 2026-09-17

## Protocol

- Dataset: BrPCD `prepared_20k_v2`
- Split: P1 random-family scene split, test family `c-bridge4`
- Seeds: 42, 43, 44
- Training: 20 epochs, 1024 points per block, batch 16,
  class-balanced sampling, weighted focal loss
- Backbone: pure-PyTorch PointNet++-style hierarchical encoder
- Methods: baseline; `deck > girder > pier` topology order;
  topology order with geometry-only transverse mirror TTA
- TTA uses no test labels and reuses the same block and occlusion seeds

## PointNet++ Three-Seed Results

| Method | 0% mIoU | 25% mIoU | 50% mIoU | 75% mIoU |
|---|---:|---:|---:|---:|
| PointNet++ baseline | 0.0877 +/- 0.0257 | 0.0866 +/- 0.0266 | 0.0855 +/- 0.0248 | 0.0848 +/- 0.0195 |
| PointNet++ topology order | 0.1654 +/- 0.0890 | 0.1631 +/- 0.0874 | 0.1622 +/- 0.0878 | 0.1652 +/- 0.1000 |
| PointNet++ topology + mirror TTA | 0.1671 +/- 0.0870 | 0.1671 +/- 0.0881 | 0.1677 +/- 0.1089 | 0.1594 +/- 0.0852 |

## Per-Class IoU at 0% Missing

| Method | girder | pier | deck |
|---|---:|---:|---:|
| PointNet++ baseline | 0.1433 +/- 0.0380 | 0.0013 +/- 0.0010 | 0.1185 +/- 0.0721 |
| PointNet++ topology order | 0.1020 +/- 0.0585 | 0.2440 +/- 0.4126 | 0.1503 +/- 0.0892 |
| PointNet++ topology + mirror TTA | 0.1023 +/- 0.0601 | 0.2434 +/- 0.4135 | 0.1556 +/- 0.0941 |

## Backbone Comparison

| Backbone / method | 0% mIoU | 25% mIoU | 50% mIoU | 75% mIoU |
|---|---:|---:|---:|---:|
| PointNet baseline | 0.4176 +/- 0.0124 | 0.3728 +/- 0.0714 | 0.3124 +/- 0.0976 | 0.2535 +/- 0.1216 |
| PointNet topology order | 0.5096 +/- 0.0591 | 0.4235 +/- 0.0457 | 0.3484 +/- 0.0283 | 0.2617 +/- 0.0321 |
| PointNet topology + mirror TTA | 0.5200 +/- 0.0304 | 0.4765 +/- 0.0319 | 0.4319 +/- 0.0388 | 0.3878 +/- 0.0514 |

## Paired Seed Differences

Negative values mean the first named method is worse.

| Comparison | 0% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| topology - baseline | +0.0777 +/- 0.0880 | +0.0764 +/- 0.0885 | +0.0767 +/- 0.0881 | +0.0803 +/- 0.1066 |
| mirror TTA - topology | +0.0017 +/- 0.0019 | +0.0040 +/- 0.0019 | +0.0055 +/- 0.0224 | -0.0058 +/- 0.0285 |
| mirror TTA - baseline | +0.0794 +/- 0.0861 | +0.0805 +/- 0.0897 | +0.0821 +/- 0.1104 | +0.0746 +/- 0.0964 |

## Decision

REJECT for the mainline.

The strongest PointNet++ variant remains far below the PointNet topology + mirror TTA comparator in clean mIoU. The PointNet++ topology group also has very large seed variance, including a seed where `pier` is not learned and another where it dominates.

This is a negative result for the current pure-PyTorch implementation and training recipe, not a general claim that PointNet++ is unsuitable for bridge segmentation.

## Artifact Paths

- Combined queue summary: `C:\Users\肖\Documents\Codex\2026-09-16\ni\runs\pointnetpp_pilot_v1\pilot_summary.csv`
- Pilot manifest: `C:\Users\肖\Documents\Codex\2026-09-16\ni\runs\pointnetpp_pilot_v1\pilot_manifest.json`
- Machine-readable analysis: `C:\Users\肖\Documents\Codex\2026-09-16\ni\work\pointnetpp_pilot_analysis.json`
