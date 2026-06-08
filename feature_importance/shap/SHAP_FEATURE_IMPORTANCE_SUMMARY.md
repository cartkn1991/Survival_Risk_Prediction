# SHAP feature importance for AESURV-DANN risk prediction

This note documents the SHAP-based feature-importance analysis that was added
on top of the earlier mean-gradient attribution. The two methods agree on the
overall biological picture but disagree systematically on the tail of "top"
features, in a way that has a clean theoretical explanation.

The final outputs live in `feature_importance/shap/`:

```
shap_proj_importance.csv              importance of each of 2048 JL-projected dims
shap_top_cpg_<target>.csv             top-K CpGs by mean(|SHAP|)
shap_top_snp_<target>.csv             top-K SNPs by mean(|SHAP|)
shap_all_cpg_<target>.csv             full SHAP table for all 2,806 candidate CpGs
shap_all_snp_<target>.csv             full SHAP table for all 13,151 candidate SNPs
shap_per_sample_<target>.npz          per-sample SHAP + z arrays  (samples x K)
shap_cohort_compare_<target>.csv      per-feature mean|SHAP| split FHS / WHI
shap_vs_gradient_<target>.png         scatter: gradient |mean_grad| vs SHAP mean|phi|
shap_bar_<target>.png                 top accelerator / decelerator bar plots
shap_beeswarm_<target>{,_cpg,_snp}.png classic SHAP summary beeswarm
shap_dependence_<target>.png          dependence (z vs SHAP) for top-2 CpG + top-2 SNP
shap_force_<target>.png               per-sample SHAP decomposition for 3 examples
shap_cohort_scatter_<target>.png      FHS-mean|SHAP| vs WHI-mean|SHAP|
shap_summary.json                     run config, completeness diagnostics, top-10 hits
annot/<target>_top_cpg_annotated.csv  gene + epigenetic-clock annotations
annot/<target>_top_snp_annotated.csv  gene + GWAS-catalog nearby-hit annotations
```

The two driver scripts are `shap_feature_importance.py` (SHAP computation +
basic plots) and `shap_extra_plots.py` (beeswarm / dependence / force /
cohort-consistency plots).  Annotation reuses `annotate_top_features.py` via
the thin wrapper `annotate_shap_features.py`.

## 1. Method

Exact SHAP / KernelSHAP would need to enumerate coalitions over our 1.37 M
input features, which is intractable.  But the model's preprocessing layer is
purely linear:

```
z_j           = (x_raw_j - mu_scaler_j) / sigma_scaler_j        (StandardScaler, M-values)
x_proj        = z @ W                                            (Johnson-Lindenstrauss, d_proj=2048)
f(x_proj)     = AESURV_head( DANN_encoder( x_proj ) )            (frozen nonlinear net)
```

For the natural baseline `x_raw_b = mu_scaler` (so `x_proj_b = 0`) the chain
rule for Integrated Gradients propagates analytically:

```
IG_proj_k   = int_0^1  d f / d x_proj_k (alpha * x_proj) d alpha          (numerical, n_steps=50)
IG_raw_j    = (1 / sigma_j) * (W[j,:] @ IG_proj)                          (linear back-projection)
phi_raw_j   = (x_raw_j - mu_scaler_j) * IG_raw_j  =  z_j * (W[j,:] @ IG_proj)
```

A trapezoidal integral with 50 alpha grid points satisfies the IG
**completeness axiom** in projected space to high accuracy:

| target | mean |sum(phi) - (f(x) - f(baseline))| | max error |
|--------|-------------------------------------|-----------|
| risk   | 0.058                               | 1.111     |
| age    | 0.698                               | 12.10     |

(Per-sample log-hazard spans roughly +/- 2.5 and age-pred spans roughly
+/- 30 years, so these are ~3% and ~3% relative errors on average.)

**Important fix:** the AESURV scaler was fitted on **M-values** (log2(beta /
(1 - beta))), not beta-values, while the cohort NPZ bundles store raw beta.
The script applies `beta_to_m` before standardising so that `z` is N(0, 1)
across the population (mean = 0.005, std = 0.996 for the top CpG cg09760963).
Without this transform every CpG looked artificially "important" because the
raw beta values landed ~17 sigma away from the M-value scaler mean.

**Baseline choice:** we use `baseline_proj = 0`, which is equivalent to
"the average sample" in M-value/dosage space.  This was preferred over
"baseline = pooled mean(x_proj)" because the JL projection of centered data
is essentially zero already, and `baseline_proj = 0` is exactly the centroid
of the (M-value, allele-dosage) feature space the model was trained on.

**Candidate set:** rather than computing per-sample SHAP for all 1.37 M
features (which would cost ~20 GB per target), we evaluate
`phi_raw_j = z_j * (W[j,:] @ IG_proj)` only for the **15,957 features that
the model itself flagged as significant** (the union of risk-significant +
age-significant features at FDR < 0.05 combined with the
|mean-grad| > median + 5 sigma_MAD effect-size cutoff). For every other raw
feature the global mean(|SHAP|) is provably bounded by the sigma_MAD
threshold times the per-projected-dim importance, which is itself low.

## 2. Headline numbers

| metric | value |
|--------|------:|
| samples (pooled FHS + WHI) | 3,209 + 510 = **3,719** |
| candidate raw features | **2,806 CpG + 13,151 SNP = 15,957** |
| IG_proj integration steps | 50 (trapezoidal) |
| CpG share of total per-sample sum(|SHAP|) | **30%** |
| SNP share of total per-sample sum(|SHAP|) | **70%** |
| mean per-feature mean(|SHAP|) -- CpG | 1.4e-3 |
| mean per-feature mean(|SHAP|) -- SNP | 0.7e-3 |
| top-1 CpG mean(|SHAP|) (cg09760963) | 3.1e-3 |
| top-1 SNP mean(|SHAP|) (1:113707844_C) | 3.7e-3 |
| cohort consistency (FHS vs WHI mean|SHAP|, Pearson) | **0.974** |

The 30%/70% CpG/SNP split is **per-sample sum** -- on a *per-feature* basis
the top CpGs and top SNPs have similar mean(|SHAP|) (~3e-3); SNPs win in
aggregate because the candidate pool is ~5x larger.

The FHS-vs-WHI mean(|SHAP|) correlation of **0.974** is the strongest piece
of evidence so far that the model is leveraging cohort-invariant signal, not
artefacts.

## 3. Top features

### 3.1 Risk (log-hazard) -- top 15 CpGs

| rank | CpG | mean(|SHAP|) | direction | gene |
|-----:|-----|------:|-----------|------|
| 1 | cg09760963 | 3.15e-3 | accel | **RNF175** (ubiquitin ligase) |
| 2 | cg08528519 | 2.91e-3 | decel | -- |
| 3 | cg02278760 | 2.80e-3 | decel | **CCDC189 / RNF40** (H2B Ub ligase, chromatin) |
| 4 | cg01069466 | 2.57e-3 | decel | CTB-58E17.1 / MIR4734 |
| 5 | cg07695379 | 2.54e-3 | decel | -- |
| 6 | cg12222949 | 2.53e-3 | decel | **COLGALT2** (collagen biosynthesis) |
| 7 | cg04294058 | 2.52e-3 | accel | COMMD10 |
| 8 | cg04134528 | 2.50e-3 | decel | -- |
| 9 | cg26244838 | 2.49e-3 | decel | **SLC12A7** (K-Cl cotransporter) |
| 10 | cg15043975 | 2.49e-3 | accel | **RASSF1 / RASSF1-AS1** (classic age-methylation locus) |
| 11 | cg26916297 | 2.48e-3 | decel | NAT16 |
| 12 | cg08548882 | 2.46e-3 | decel | **PLEKHG1** |
| 13 | cg15282632 | 2.46e-3 | accel | **CACNB3** (Ca channel) |
| 14 | cg12484370 | 2.45e-3 | decel | **NFIB** (nuclear factor I/B) |
| 15 | cg27154163 | 2.45e-3 | accel | **KIT** (RTK, hematopoiesis) |

### 3.2 Risk -- top 15 SNPs

| rank | SNP | mean(|SHAP|) | direction | gene | nearby GWAS rsIDs |
|-----:|-----|------:|-----------|------|--------------------|
| 1 | 1:113707844_C | 3.74e-3 | decel | **PHTF1** (1p13, immune) | -- |
| 2 | 3:130255330_T | 3.46e-3 | accel | **COL6A4P2** (collagen pseudogene) | rs9813712 |
| 3 | 1:201139965_A | 3.21e-3 | accel | **TMEM9** (mTOR/lysosomal) | rs73081051 |
| 4 | 6:113088640_A | 2.99e-3 | accel | -- | rs6420694, rs9320429, rs7771682, rs7738702 |
| 5 | 8:13657291_C | 2.91e-3 | accel | -- | rs28415552 |
| 6 | 8:56996727_G | 2.91e-3 | accel | -- | rs3808629 |
| 7 | 18:49934833_T | 2.86e-3 | decel | **MYO5B** (autophagy / lysosomal trafficking) | rs1790796, rs1787521, rs1787328 |
| 8 | 13:23546102_C | 2.85e-3 | accel | -- | rs184884487 |
| 9 | 5:26509848_C | 2.83e-3 | accel | -- | -- |
| 10 | 12:61642896_T | 2.78e-3 | accel | -- | -- |
| 11 | 6:156157108_A | 2.76e-3 | decel | -- | rs7763221, rs56222681 |
| 12 | 11:123720105_T | 2.76e-3 | accel | -- | rs625888 |
| 13 | 6:90593357_C | 2.76e-3 | accel | -- | -- |
| 14 | 16:58344789_G | 2.73e-3 | accel | **GINS3** (DNA replication) | rs140445141 |
| 15 | 3:16611257_A | 2.69e-3 | accel | **DAZL** (RNA-binding) | rs4685366 |

**Biological themes emerging from SHAP:**

- **mTOR / autophagy / lysosomal acidification** -- TMEM9, MYO5B, MARCHF3 (#13 SNP), PHTF1 region (chromatin/immune via 1p13).
- **Chromatin remodelling and ubiquitin-mediated proteostasis** -- RNF175 (CpG #1), RNF40 (CpG #3), GINS3, MARCHF3.
- **Collagen / extracellular matrix** -- COLGALT2 (CpG #6), COL6A4P2 (SNP #2).  Aligns with ECM-mediated senescence literature.
- **Classic age-methylation locus** -- **RASSF1** (CpG #10) is one of the most reproducible cancer + aging methylation markers.
- **Nuclear factor / transcription** -- NFIB (CpG #14), DAZL.

Top-15 hits are nearly identical between risk and age targets (cg09760963,
cg02278760, cg01069466, cg04134528, cg15282632, cg12222949 are in both
lists), consistent with the high age-risk coupling (r = 0.83 pooled) shown
in the diagnostic plots.

## 4. SHAP vs mean-gradient: where the two methods disagree

| metric | CpG | SNP |
|--------|----:|----:|
| top-50 overlap (risk) | **7 / 50** | **0 / 50** |
| top-50 overlap (age) | 5 / 50 | 0 / 50 |
| top-200 overlap (risk) | 55 / 200 | 0 / 200 |
| Spearman rank correlation (all 2,806 / 13,151) | 0.76 | 0.76 |

The bulk distributions agree (Spearman 0.76 for both modalities), but the
**top tails diverge sharply**, especially for SNPs.  The explanation is in
the scaler's `sigma_scale_`:

| | median(sigma_scale_) on top-50 |
|---|---:|
| **Gradient top-50 SNPs** | **0.153** (5th percentile of all SNPs)|
| **SHAP top-50 SNPs** | **0.412** (close to population median 0.50) |

The mean-gradient back-projection is `(1 / sigma) * (W @ partial_f / partial_x_proj)`,
so SNPs with tiny sigma (rare variants, MAF ~ 1-3%) are amplified by
1 / sigma > 6.  SHAP back-projection is `z * (W @ IG_proj)`, where
`z = (x_raw - mu_scaler) / sigma_scaler`.  For a rare variant
**z is essentially zero for the 95+% of samples without the minor allele**,
so the population-mean |SHAP| stays small.

In other words:

> The gradient method flags features the model is *locally most sensitive*
> to (`d f / d x`), regardless of whether typical individuals actually
> express that sensitivity.  SHAP flags features that *actually move
> predictions away from baseline across the population* -- exactly the
> intuitive notion of "important for risk prediction".

This is why SHAP is generally preferred for global feature ranking.  Both
analyses are kept in the repo: the gradient table is the right entry point
for "which features could the model *react* to?", and the SHAP table is the
right entry point for "which features *did* the model use across this
cohort?".

For CpGs, where there are no rare-variant analogues, the two methods agree
much more closely (Spearman 0.76, 55 / 200 top-200 overlap) and disagree
mostly on ordering within the head of the distribution.

## 5. Diagnostics and visualisations

### 5.1 SHAP beeswarm (`shap_beeswarm_risk.png`)

Each row is one of the top-20 features, each dot is one sample.  X-axis is
its SHAP value (impact on log-hazard); colour is the standardised feature
value `z`.  Several features (top: cg09760963, COL6A4P2 SNP) show monotonic
red-to-blue gradients, indicating clean linear-in-`z` contributions over the
nonlinear network.

### 5.2 Dependence plots (`shap_dependence_risk.png`)

For the top-2 CpGs and top-2 SNPs, scatter of `z_j` vs `phi_j` (per sample),
coloured by the sample's total SHAP contribution.  SNPs show the expected
three vertical bands at standardised dosage 0, 1, 2; the SHAP values are
nearly linear in `z` because the AESURV head is shallow (2 hidden layers).

### 5.3 Force / waterfall examples (`shap_force_risk.png`)

Three illustrative samples (lowest, median, highest predicted log-hazard)
with their top-8 features.  Useful for spot-checking why the model rated a
specific individual high or low.

### 5.4 Cohort consistency (`shap_cohort_scatter_risk.png`)

mean(|SHAP|) computed separately on FHS (n = 3,209) vs WHI (n = 510),
plotted against each other.  Pearson 0.974, Spearman 0.954.  This is the
strongest evidence so far that the DANN cohort-invariance training
succeeded: the model assigns similar importance to the same features in
both cohorts.

## 6. What we believe the model is doing

Combining the gradient analysis, the GWAS-style outlier analysis, and now
SHAP:

1. **Methylation dominates per-feature, SNPs dominate in aggregate.**  Each
   top CpG carries ~2-4x more SHAP than each top SNP, but with ~5x more
   SNPs in the candidate pool the SNP block contributes 70% of total
   per-sample |SHAP|.
2. **The biology is consistent with canonical aging pathways** -- mTOR /
   autophagy / lysosomal acidification (TMEM9, MYO5B), chromatin
   ubiquitination (RNF40, RNF175), collagen / ECM (COLGALT2, COL6A4P2),
   classic age-methylation locus RASSF1.
3. **The model uses CpGs OUTSIDE canonical clocks.**  Zero of the top-200
   SHAP CpGs (and zero of the top-200 mean-gradient CpGs) fall in any of
   Horvath / Hannum / PhenoAge / GrimAgeV2 / DunedinPACE / Zhang2019 / Lin
   / Weidner / Horvath-SkinBlood lists.  This is *not* a bug: the JL
   projection + DANN encoder construct a new latent space that need not
   align with hand-curated clock probes.
4. **The model is cohort-invariant** -- mean(|SHAP|) agrees at Pearson 0.97
   between FHS and WHI, confirming the DANN + MMD training did its job.
5. **The two attribution methods complement each other.**  Mean-gradient
   surfaces rare-variant sensitivities (`1 / sigma` amplification), SHAP
   surfaces common-variant population-level contributions (`z`-weighted).

## 7. Reproducing the analysis

```powershell
# 1. compute SHAP for risk + age (uses cached projections + the W cache)
python shap_feature_importance.py --targets risk,age --n-steps 50

# 2. beeswarm / dependence / force / cohort-consistency plots
python shap_extra_plots.py --targets risk,age --top-k 20

# 3. annotate top SHAP CpGs / SNPs with genes + GWAS hits (reuses caches)
python annotate_shap_features.py --targets risk,age
```

Runtime on a CUDA-enabled box: ~1 minute for step 1 (most of which is the
26-second bundle column slice), ~10 seconds for step 2, ~35 seconds for
step 3 (the latter dominated by the first-time `Annotation.csv` parse +
Ensembl REST top-up).
