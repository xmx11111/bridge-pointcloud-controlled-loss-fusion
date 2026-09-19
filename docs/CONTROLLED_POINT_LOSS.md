# Controlled Point Loss

## Protocol

1. Select an evaluation block and a view configuration.
2. Determine points that are not visible from the view.
3. Remove the prescribed fraction of points before any model forward pass.
4. Perform a fresh forward pass on the surviving subset.
5. Score the result under one of the two reporting conventions below.

Missing rates are 0%, 25%, 50%, and 75%. Results use seeds 42, 43, and 44.

## Reporting Conventions

### Surviving Points

Only points retained after removal are scored. This is a conditional accuracy:
it answers how well the model segments the points that remain visible.

### Coverage-Penalised Accuracy

Every point in the original reference block is scored. Removed points are
counted as errors. This convention reflects end-to-end coverage and decreases
as point loss increases.

The difference between the two conventions is the coverage loss caused by
missing points.

## Why Order Matters

The project audit compared:

- prediction on the complete cloud followed by scoring only surviving points;
- removal followed by a fresh forward pass.

At 25% missing, the two procedures produced mIoU values of 0.5173 and 0.4303.
The first is inflated because the model has already accessed the complete
geometry. All paper results use remove-then-forward inference.

## Block-Level Statistics

Seed values are averaged within each evaluation block. The pipeline then uses
10,000 paired bootstrap resamples and a two-sided sign test over 352 blocks.
This tests whether a gain is driven by a small number of blocks or one seed.
It does not remove the limitation that training uses only three seeds and
blocks within a scene share spatial structure.
