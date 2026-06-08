# GWAS pipeline: mortality, AESurv risk, and full SNP discovery

This folder contains phenotype/covariate helpers for external tools **and** an in-repo
Python pipeline for **FHS discovery → WHI replication → outlier stratification**.

## A. External GWAS inputs (PLINK / SAIGE)

### 1. Prepare phenotype and covariate files

First, ensure you have built the merged risk + lifestyle table:

```powershell
cd D:\SNP_datasets
$env:PYTHONPATH="D:\SNP_datasets"
python bio_relevance/attach_dann_risk.py --device cuda
```

Then create GWAS-ready tables:

```powershell
python gwas/prepare_gwas_inputs.py `
  --merged-parquet feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet `
  --out-dir gwas
```

Outputs:

- `gwas/fhs_phenotypes.tsv` — FHS: FID/IID, survival time/event, AESurv `log_h`, traits
- `gwas/fhs_covariates.tsv` — FHS: age, sex, batch
- `gwas/whi_phenotypes.tsv`, `gwas/whi_covariates.tsv` — WHI analogues

These can be joined with PLINK genotype files using FID/IID.

See section 2 below for example PLINK 2 commands.

---

## B. Full SNP pipeline (recommended)

**Script:** `gwas/run_full_gwas_pipeline.py`

**Design (matches mini-GWAS outlier logic, but discovery on all FHS):**

| Stage | Cohort | What it does |
|-------|--------|--------------|
| 1. Discovery | FHS (~3209) | Linear GWAS on **age+sex-adjusted `log_h` residual** for all SNPs in the NPZ bundle |
| 2. Replication | WHI (~510) | Test clumped FHS lead SNPs; same model; fixed-effect meta |
| 3. Stratification | FHS (+ pooled groups) | Mean allele dosage in **resilient / middle / accelerated** (top/bottom 10% residual) |
| 4. Enrichment | FHS hits | Hypergeometric overlap with ~11k AESurv gradient-significant SNPs |

Shared utilities live in `gwas/gwas_common.py` (residual phenotype, batched OLS, clumping, plots).

### Run all stages

```powershell
cd D:\SNP_datasets
$env:PYTHONPATH="D:\SNP_datasets"
python gwas/run_full_gwas_pipeline.py --stage all
```

### Smoke test (first 20k SNPs)

```powershell
python gwas/run_full_gwas_pipeline.py --stage all --snp-limit 20000
```

### Run individual stages

```powershell
python gwas/run_full_gwas_pipeline.py --stage discovery
python gwas/run_full_gwas_pipeline.py --stage replication
python gwas/run_full_gwas_pipeline.py --stage stratification
python gwas/run_full_gwas_pipeline.py --stage enrichment
```

### Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--out-dir` | `feature_importance/gwas` | All pipeline outputs |
| `--discovery-p` | `1e-5` | Suggestive threshold for lead selection |
| `--replication-p` | `0.05` | WHI p-value + same sign as FHS |
| `--max-leads` | `500` | Max independent lead SNPs (1 Mb clumping) |
| `--residual-q-lo/hi` | `0.10 / 0.90` | Outlier group cutoffs (same as mini-GWAS) |
| `--batch-size` | `4096` | SNPs per discovery batch |
| `--force` | off | Recompute even if cached CSVs exist |

### Outputs (`feature_importance/gwas/` by default)

| File | Description |
|------|-------------|
| `discovery_fhs_results.csv` | All FHS SNP association results |
| `discovery_fhs_manhattan.png`, `discovery_fhs_qq.png` | Discovery plots |
| `discovery_summary.json` | λ_GC, top hits, thresholds |
| `discovery_lead_snps.csv` | Clumped lead SNPs for replication |
| `replication_results.csv` | FHS + WHI + meta for each lead |
| `replicated_hits.csv` | Subset passing replication criteria |
| `outlier_groups.csv`, `outlier_groups.png` | Pooled group assignments |
| `stratification_dosage.csv`, `stratification_dosage.png` | Dosage by group for replicated SNPs |
| `enrichment_summary.json` | Overlap with `significant_risk.csv` SNPs |

### Inputs (defaults)

- FHS/WHI NPZ bundles under `vae_cox_cache/bundles/`
- SNP column lists (`*_snp_genotypes.snp_columns.txt`)
- AESurv risk: `runs/aesurv_aux_grid/age12.00_cell1.00/aesurv_aux_risk.npz`
- Gradient SNP panel: `feature_importance/significance/significant_risk.csv`

### Interpretation notes

- Discovery on **`log_h` residual** uses the same AESurv model scores as training (in-sample on FHS). WHI replication partially addresses this; mortality GWAS via external tools (section A) is a co-primary sensitivity.
- Outlier stratification is **post hoc** on replicated loci — not a discovery design.
- Mini-GWAS (`minigwas_outliers.py`) remains useful as supplementary extreme-concordance analysis.

---

### Cross-compare log_h vs mortality hits

```powershell
python gwas/cross_compare_gwas_hits.py
python gwas/run_fhs_snp_cox.py --snp-csv feature_importance/gwas/cross_compare_union.csv --snp-col snp
```

Outputs: `cross_compare_union.csv`, `cross_compare_pairs.csv`, `fhs_snp_cox_results.csv`

---

## C. Suggested external GWAS commands (examples)

### Linear GWAS on AESurv `log_h` and traits (PLINK 2)

```bash
plink2 \
  --pfile FHS_genotypes \
  --pheno gwas/fhs_phenotypes.tsv \
  --pheno-name log_h \
  --covar gwas/fhs_covariates.tsv \
  --covar-name age,sex,batch \
  --glm hide-covar \
  --out gwas/fhs_logh_gwas
```

### Survival GWAS on mortality (R / SAIGE-survival)

Use `gwas/fhs_phenotypes.tsv` and `gwas/fhs_covariates.tsv` with `time` and `event` columns.

---

## D. Downstream MR and fine-mapping

GWAS summary statistics from either pipeline arm can feed TwoSampleMR, SuSiE/FINEMAP, etc.
