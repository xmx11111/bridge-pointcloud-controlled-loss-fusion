# Paired Bootstrap Statistics V1

Date: 2026-09-18

Method: paired within family/seed; 95% CI by bootstrapping over folds.
The eligible scope has only two families, so its CI is exploratory.

| Comparison | 0% mean [95% CI] | 25% | 50% | 75% | Positive folds |
|---|---:|---:|---:|---:|---|
| in_scope_fixed_minus_none | +0.0068 [+0.0032, +0.0104] | +0.0028 [+0.0004, +0.0052] | -0.0001 [-0.0006, +0.0004] | +0.0008 [-0.0007, +0.0023] | 2, 2, 1, 1 |
| out_scope_fixed_minus_none | -0.0439 [-0.0668, -0.0212] | -0.0348 [-0.0502, -0.0193] | -0.0341 [-0.0451, -0.0232] | -0.0272 [-0.0343, -0.0186] | 0, 0, 0, 0 |
| in_scope_geometry_minus_none | +0.0066 [+0.0032, +0.0100] | +0.0028 [+0.0004, +0.0053] | +0.0002 [+0.0000, +0.0004] | +0.0032 [+0.0023, +0.0041] | 2, 2, 2, 2 |
| in_scope_augmentation_fixed_minus_none | -0.0200 [-0.0501, +0.0101] | -0.0059 [-0.0220, +0.0102] | +0.0042 [-0.0040, +0.0124] | +0.0099 [+0.0032, +0.0167] | 1, 1, 1, 2 |
| scope_fixed_minus_always_none | +0.0023 [+0.0000, +0.0057] | +0.0009 [+0.0000, +0.0027] | -0.0000 [-0.0003, +0.0002] | +0.0003 [-0.0003, +0.0011] | 2, 2, 1, 1 |

## Artifact Paths

- `C:\Users\肖\Documents\Codex\2026-09-16\ni\work\paired_statistics.json`
