# Lifestyle validation for AESurv-DANN-Aux log-hazard

Independent epidemiologic checks on the **subject-level** predicted log-hazard (`log_h`) from the frozen [`models/aesurv_final`](../../../models/aesurv_final) bundle. Lifestyle variables were **not** used during model training.

## Prerequisites

1. **dbGaP exports** — one row per analysis sample (same individuals as the methylation/SNP parquets):
   - `external_phenotypes/fhs_lifestyle_raw.csv`
   - `external_phenotypes/whi_lifestyle_raw.csv`

2. **Variable map** — copy [`bio_relevance/lifestyle_variable_map.example.json`](../../../bio_relevance/lifestyle_variable_map.example.json) to `bio_relevance/lifestyle_variable_map.json` and set your dbGaP column names.

3. **Model artifacts** (defaults used by scripts):
   - `models/aesurv_final/` (`dann_encoder.pt`, `aesurv_head.pt`, `dann_preprocess.npz`)
   - `vae_cox_cache/jl_proj/proj_FHS_b24e34c31a.npz`
   - `vae_cox_cache/jl_proj/proj_WHI_b159d5bf88.npz`

## ID keys for merge

| Cohort | Primary ID in omics parquet | dbGaP export should include |
|--------|------------------------------|-----------------------------|
| FHS (n≈3,209) | `Share_ID` | `Share_ID` or `IID` |
| WHI (n≈510) | `sample_ID` | `dbGaP_Subject_ID` or `SUBJID`; optional map via `whi_meth_ids.csv` → `SAMPLE_ID` |

Target merge rate: **≥85%** of omics samples per cohort.

## Pipeline commands

From repo root (`D:\SNP_datasets`):

```powershell
# 1) Harmonize dbGaP CSVs → canonical columns
python bio_relevance/harmonize_lifestyle.py --map bio_relevance/lifestyle_variable_map.json

# 2) Attach log_h + survival metadata + lifestyle
python bio_relevance/attach_dann_risk.py --lifestyle-parquet feature_importance/bio_relevance/lifestyle/lifestyle_harmonized.parquet

# 3) Statistics, JSON summary, all figures
python bio_relevance/lifestyle_risk_validation.py --fhs-only
```

### FHS DAF Cox file (current analysis)

```powershell
python bio_relevance/import_fhs_daf_lifestyle.py --daf-csv "F:\FHS_phenotypic data\FHS_DAF_COX\FHS_all_daf\All_data_cox.csv"
python bio_relevance/attach_dann_risk.py
python bio_relevance/lifestyle_risk_validation.py --fhs-only
# Or linkage figures only:
python bio_relevance/lifestyle_logh_linkage_figure.py --mode both
```

### Demo mode (no dbGaP files yet)

Generates **synthetic** lifestyle for pipeline testing only (not for publication):

```powershell
python bio_relevance/harmonize_lifestyle.py --demo --seed 7
python bio_relevance/attach_dann_risk.py
python bio_relevance/lifestyle_risk_validation.py
```

## Outputs

| File | Description |
|------|-------------|
| `lifestyle_harmonized.parquet` | Canonical lifestyle per subject |
| `lifestyle_harmonization.json` | Recode rules, missingness, winsor bounds |
| `lifestyle_risk_merged.parquet` | `log_h`, survival, demographics, lifestyle |
| `merge_qc.json` | Merge rates by cohort and ID column |
| `lifestyle_validation_summary.json` | Concordance, Cox models, residual risk, never-smoker subset |
| `figures/lifestyle_validation_supp.pdf` | 2×2 supplementary figure (concordance, Cox, KM) |
| `figures/lifestyle_confounding_figure.pdf` | Raw vs adjusted alcohol–log_h by batch (confounding) |
| `figures/lifestyle_logh_association_simple.pdf` | **Simple:** 1×3 raw lifestyle vs log_h (LOESS, batch colors) |
| `figures/lifestyle_logh_linkage_main.pdf` | **Main:** raw + adjusted 3 traits, heatmap, Cox C-index, residual-risk callout |

### Figure panel guide (`lifestyle_logh_linkage_main.pdf`)

| Panel | Content |
|-------|---------|
| Row 1 | Raw scatter: cigarettes, alcohol/occasion, sleep vs log_h (ρ annotated) |
| Row 2 | Adjusted residuals (age, sex, batch removed); partial ρ annotated |
| G | Heatmap: raw vs partial Spearman ρ for all three traits |
| H | Cox Harrell C-index M0–M3 (lifestyle vs log_h vs both) |
| I | Text: log_h_resid still predicts mortality after lifestyle adjustment |

## Interpretation

- **Concordance:** higher `log_h` with smoking, alcohol, short sleep → face validity.
- **Cox M3 vs M1:** `log_h` adds prognostic value beyond lifestyle.
- **`log_h_resid`:** risk after regressing out lifestyle + age/sex/batch; KM/Cox on residual tests omics signal **beyond** measured behavior.
- **Never-smokers:** persistence argues the score is not only a smoking proxy.

See also [`feature_importance/MODEL_FEATURE_USE_MANUSCRIPT.md`](../../MODEL_FEATURE_USE_MANUSCRIPT.md) (Lifestyle validation subsection).
