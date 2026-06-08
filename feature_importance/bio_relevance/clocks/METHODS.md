# Methylation clock mortality benchmark — methods

This document describes how each predictor was scored and how mortality discrimination (Harrell C-index) and age-adjusted Cox models were computed. Machine-readable version: `clock_methods.json`.

## Cohorts

| Cohort | N | Methylation+SNP NPZ | Metadata parquet |
|--------|---|---------------------|------------------|
| FHS (training) | 3,209 | `FHS_methylation_with_snp_1milfeatures_combined_training.npz` | `FHS_methylation_with_snp_1milfeatures.parquet` |
| WHI (external test) | 510 | `WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz` | `WHI_methylation_with_snp_merged_1milfeatures.parquet` |

Methylation values are **beta** (0–1) per CpG probe, aligned to NPZ column lists.

## Training supervision contrast (key for interpretation)

| Predictor | Trained to predict | Age at inference? | Omics at inference |
|-----------|-------------------|-------------------|-------------------|
| Horvath, Hannum, PhenoAge, **EpiClock** | Chronological age | No | Methylation only |
| GrimAge V2 | Mortality-calibrated DNAm age | **Yes** (+ sex) | Methylation + age + sex |
| DunedinPACE | Pace of aging | No | Methylation only |
| **AESurv `log_h`** | Survival (time-to-event) | **No** | Methylation + SNP |

**EpiClock** is trained with the same paradigm as Horvath (methylation → age). **AESurv `log_h`** is the mortality-hazard output: trained on survival with optional auxiliary age loss during training, but **chronological age is not an input when producing `log_h`**.

## Clock scoring formulas

Coefficients: `feature_importance/annot/clock_lists/` (from [biolearn](https://github.com/bio-learn/biolearn) unless noted).

### Horvath (353 CpGs)

1. Linear predictor: \( \text{LP} = b_0 + \sum_j \beta_j \cdot \text{coef}_j \)
2. Age (years): `anti_trafo(LP + 0.696)` where `anti_trafo(x) = (1+20)*exp(x)-1` if x&lt;0 else `(1+20)*x+20`

### Hannum (71 CpGs)

Age (years) = intercept + Σ (beta × coef).

### PhenoAge (513 CpGs)

Linear methylation age (biolearn): intercept + Σ (beta × coef). **No** Gompertz transform.

### DunedinPACE (173 CpGs)

Linear: intercept + Σ (beta × coef). Note: reference implementation uses Gold-standard quantile normalization before scoring; **not applied** in this benchmark.

### GrimAge V2

1. Fit linear DNAm proxies per protein component (CpGs + optional intercept/age per component).
2. Linear COX score: `COX = Σ (component × weight) + b_age×Age + b_female×Female`.
3. `GrimAge = ((COX - m_cox)/sd_cox) × sd_age + m_age` (biolearn `GrimageModel`).

### EpiClock (custom, ~6,762 CpGs)

`EpiClock = 32.14246737 + Σ (beta × Coefficient Estimate)` from  
`G:\All_GEO_epiclock_datasets\4k data training final\6k probes.csv`  
(same age-supervised training philosophy as Horvath).

### AESurv `log_h`

Frozen bundle `models/aesurv_final`: DANN encoder on JL-projected **methylation + SNP** → AESurv Cox head → `log_h`. Projections: `vae_cox_cache/jl_proj/proj_FHS_*.npz`, `proj_WHI_*.npz`.

### Age acceleration

`{Clock}_accel = Clock − chronological_age` (except GrimAge_accel in Cox analysis, which uses regression residual on age).

## Harrell C-index

- Implementation: `train_dann_survival.harrell_c_index`
- **Direction:** higher score → higher mortality risk (shorter survival)
- **FHS:** stratified random split 85% train / 15% validation (`random_state=42`); primary metric = **validation** C-index
- **WHI:** full cohort (external test), no split

## Age-adjusted Cox models

Script: `bio_relevance/clock_mortality_age_adjusted_cox.py`  
Output: `clock_cox_age_adjusted.json`, `clock_cox_age_adjusted_{fhs,whi}.pdf`

All continuous predictors are **z-scored** per cohort. `lifelines.CoxPHFitter` with `penalizer=0.01`. C-index computed on Cox **partial hazard** linear predictor (Harrell).

| Model | Formula |
|-------|---------|
| M0 | age |
| M1 | log_h |
| M2 | GrimAge |
| M3 | GrimAge_accel (GrimAge residual after regressing on age) |
| M4 | age + log_h |
| M5 | age + GrimAge_accel |
| M6 | age + GrimAge |

**Likelihood ratio tests (LRT):** nested comparison vs M0 to test incremental value beyond chronological age.

### Systematic incremental validity (all predictors)

For each predictor, fit `age + predictor (+ sex)` and report the **age-adjusted HR** for the predictor (per 1 SD) plus an LRT vs `age` alone. Output: `clock_cox_incremental_validity.json`, forest plots `clock_cox_incremental_forest_{fhs,whi}.pdf`.

This is the appropriate test when raw Harrell C-index is dominated by chronological age (especially for age-trained clocks).

Forest plots (traditional layout; separate file per split; no combined panels):

**Raw clock scores** (Cox: age + predictor + sex in FHS):

- `clock_cox_incremental_forest_fhs_train.pdf` / `.png`
- `clock_cox_incremental_forest_fhs_val.pdf` / `.png`
- `clock_cox_incremental_forest_whi.pdf` / `.png`

**Age acceleration** (Cox: age + accel + sex in FHS; GrimAge accel = regression residual):

- `clock_cox_accel_forest_fhs_train.pdf` / `.png`
- `clock_cox_accel_forest_fhs_val.pdf` / `.png`
- `clock_cox_accel_forest_whi.pdf` / `.png`

FHS uses the same 85% train / 15% validation split as the C-index benchmark (`seed=42`, stratified on event).

**Head-to-head (AESurv vs each clock):** joint model `age + clock + log_h (+ sex in FHS)` with LRT vs `age + clock` and vs `age + log_h`. JSON: `clock_cox_logh_beyond_clocks.json`; forests: `clock_cox_logh_beyond_clocks_{fhs_train,fhs_val,whi}.pdf`.

```powershell
python bio_relevance/clock_mortality_age_adjusted_cox.py --head-to-head-only
```

Redraw only: `python bio_relevance/clock_mortality_age_adjusted_cox.py --plots-only`

## Reproducibility

```powershell
cd D:\SNP_datasets
$env:PYTHONPATH="D:\SNP_datasets"
python bio_relevance/epigenetic_clock_mortality_benchmark.py --device cuda
python bio_relevance/clock_mortality_age_adjusted_cox.py
```

## Output files

| File | Description |
|------|-------------|
| `clock_scores_fhs.parquet` / `clock_scores_whi.parquet` | Per-sample scores |
| `clock_mortality_cindex.json` | Harrell C-index all/train/val |
| `clock_mortality_cindex_*.pdf` | Bar charts |
| `clock_methods.json` | Machine-readable methods |
| `clock_cox_age_adjusted.json` | Cox + LRT results |
| `METHODS.md` | This document |
