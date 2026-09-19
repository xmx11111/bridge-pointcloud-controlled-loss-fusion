# Excluded Assets

The following assets are intentionally excluded from the compact repository:

- trained `.pt`, `.pth`, and checkpoint files;
- `work/.venv-stats` and all Python virtual environments;
- downloaded BrPCD and SemanticBridge datasets;
- third-party BridgeNetv2 source and data mirrors;
- cloned GitHub skills and external repositories;
- unrelated TTA, scope, symmetry, LOBO, and legacy exploratory branches;
- large intermediate NPZ tensors and rendering caches;
- document-rendering caches and temporary images;
- editor, OS, and cache files.

The repository retains small JSON/CSV metrics, summary tables, manuscript
figures, selected raw fusion JSON, the controlled-loss case-study data, and the
scripts needed to understand the result pipeline.

Checkpoints can be recreated from the public datasets and documented training
scripts. If the authors choose to release checkpoints later, use a dedicated
external artifact service rather than adding them to the Git history.
