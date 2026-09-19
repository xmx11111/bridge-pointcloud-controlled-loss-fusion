# SemanticBridge Cross-Dataset Pilot v1

Date: 2026-09-17

## Data Preparation

- Official real-data archives downloaded and MD5-verified.
- TLS: 20 real bridge scenes, 15 train and 5 test.
- MLS: 7 paired domain-shift scenes.
- Pilot mapping:
  - `superstructure -> girder`
  - `pillar -> pier`
  - `abutment -> pier`
  - `top surface/deck -> deck`
- Environment and unlabeled classes were discarded.
- Each scene was sampled to 20,000 points with balanced class counts.
- Split used for the pilot: 13 train, 2 validation, 5 official TLS test scenes.

## Structural Check

The expected vertical order `deck > girder > pier` holds in:

- 13/13 training bridges;
- 2/2 validation bridges;
- 5/5 official test bridges.

## Pilot Results

The current model and label mapping do not transfer directly:

| Method | 0% mIoU | 25% mIoU | 50% mIoU | 75% mIoU |
|--------|---------|----------|----------|----------|
| Baseline | 0.2615 | 0.2971 | 0.3260 | 0.3327 |
| Baseline + geometry-only mirror TTA | 0.2260 | 0.2643 | 0.2794 | 0.3046 |
| Topology order | 0.2207 | 0.2716 | 0.2878 | 0.3117 |
| Topology order + geometry-only mirror TTA | 0.2087 | 0.2392 | 0.2596 | 0.2741 |

## Interpretation

- The structural order exists in SemanticBridge, so the failure is not a
  missing physical prior.
- A direct `superstructure -> girder` mapping is too coarse and likely
  collapses multiple structural roles into one class.
- The global vertical proxy and all-point PCA are not sufficient for this
  sensor/geometry domain.
- The local PointNet backbone and BrPCD-tuned training budget are not
  appropriate for this real dataset.
- This pilot therefore cannot serve as external validation yet.

## Decision

Do not use the current BrPCD-tuned pipeline as a universal cross-dataset
method. Keep SemanticBridge as a real-world external benchmark, but treat it
as a separate adaptation task requiring:

1. a careful mapping from the official nine classes;
2. a dataset-native coordinate frame estimate;
3. a stronger or dataset-adapted backbone;
4. comparisons against the official SemanticBridge baselines.
