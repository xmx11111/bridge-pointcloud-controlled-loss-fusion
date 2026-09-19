# Result Artifact Map

| Paper item | Primary file |
|---|---|
| Controlled occlusion audit | `outputs/PAPER_V2_AEI_EN.md`; `docs/experiment-logs/GEOMETRY_2D3D_MAINLINE_CHECK_V1.md` |
| BrPCD unified main table | `outputs/main_table_v1.md/.json/.csv` |
| Block-level paired statistics | `outputs/block_paired_statistics_v1.md/.json` |
| Force-alignment comparison | `outputs/aligned_fusion_table_v1.md` |
| Force-alignment per-class results | `outputs/aligned_fusion_perclass_v1.md` |
| Force-alignment weight sensitivity | `outputs/aligned_fusion_weight_sensitivity_v1.md` |
| Coverage-penalised comparison | `outputs/coverage_penalty_table_v1.md/.json` |
| SemanticBridge external validation | `outputs/semanticbridge_external_table_v1.md/.json` |
| BridgeNetv2 matched baseline | `outputs/EXTERNAL_BASELINE_V1.md` |
| Fig. 1 method overview | `outputs/figures/fig1_method_overview.png/.pdf/.svg` |
| Fig. 2 BrPCD curves | `outputs/figures/fig2_brpcd_curves.png/.pdf` |
| Fig. 3 per-class IoU | `outputs/figures/fig3_per_class_iou.png/.pdf` |
| Fig. 4 external validation | `outputs/figures/fig4_semanticbridge_curves.png/.pdf` |
| Fig. 5 success/failure cases | `outputs/figures/fig5_cases.png/.pdf` |
| Fig. 5 case data | `outputs/fig5_case_metrics_v1.json`; `results/raw/fig5_case_data_v1.npz` |
| OHEM negative result | `docs/experiment-logs/OHEM_STRONG_SUPERVISION_V1.md` |
| PointNet++ negative result | `docs/experiment-logs/POINTNETPP_PILOT_V1.md` |

The `results/run_metrics/` directory contains compact JSON and CSV files
mirroring the original `runs/` directory. Model checkpoints are not included.
