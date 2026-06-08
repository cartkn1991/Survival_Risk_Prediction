# Epoch 26 downstream analyses

All outputs from repeating the **aesurv_final (WHI C≈0.64)** workflow using the joint epoch-26 checkpoint.

## Model

- Weights: `joint_model_epoch26.pt`
- C-index: FHS train **0.862**, val **0.832**, WHI **0.699** (`cindex_summary.json`)

## Re-run everything

```bash
export PYTHONPATH=.
python run_pipeline.py --device cuda
```

Individual stages:

```bash
python run_pipeline.py --stage downstream --device cuda
python run_pipeline.py --stage gwas
python run_pipeline.py --stage ewas
python run_pipeline.py --stage figures
```

## Output layout

| Folder | Contents |
|--------|----------|
| `bio_relevance/clocks/` | Clock C-index JSON/PDF, Cox models, comparison plots |
| `bio_relevance/lifestyle/` | Merged risk table, validation JSON, KM/linkage figures |
| `bio_relevance/lifestyle/figures/` | Lipid/glucose plots |
| `bio_relevance/cell_chromatin/` | Cell fraction correlations, context35 |
| `bio_relevance/pathway/` | g:Profiler enrichment |
| `bio_relevance/kegg_gsea/` | KEGG GSEA |
| `feature_importance/` | Gradients, `dann_W.npy`, top-feature CSVs |
| `feature_importance/significance/` | FDR / MAD feature selection |
| `feature_importance/shap/` | Integrated-gradient SHAP |
| `feature_importance/gwas/` | GWAS input TSVs (from merged log_h) |
| `scores/` | Cached joint JL projections (`joint_proj_pooled.npz`) |

## Notes

- Lifestyle harmonization is **cohort-only**; copied from the prior run (`lifestyle_harmonized.parquet`).
- RFE/ablation on masked raw features still target the frozen bundle architecture; use significance + SHAP for joint model feature sets.
- Primary comparison baseline (old model): `feature_importance/bio_relevance/` at repo root.
