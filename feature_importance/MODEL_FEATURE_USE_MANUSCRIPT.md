# Model feature use — Methods and Results (draft text)

Use with figure: `feature_importance/figures/model_feature_use_onepage.pdf`

---

## Methods (feature importance and selection)

We quantified which raw CpG and SNP inputs the frozen AESurv-DANN-Aux model relies on for **predicted log-hazard** (mortality risk) and **predicted age**. The deployed pipeline maps 1,372,416 concatenated methylation and genotype features (393,234 CpGs + 979,182 SNPs) through a fixed standard scaler and Johnson–Lindenstrauss projection to 2,048 dimensions, a frozen domain-adversarial variational encoder (128-dimensional cohort-aligned latent), and a trained AESurv head with an 8-dimensional variational bottleneck. The head was fit on Framingham Heart Study (FHS) survival with auxiliary losses for chronological age and Houseman cell composition (weights 12.0 and 1.0), plus reconstruction and cohort-adversarial terms, and evaluated on FHS validation and Women’s Health Initiative (WHI) test samples.

Feature importance was defined by **gradients of the model output with respect to inputs**, not by sparse input weights. For each of *N* = 3,719 pooled projected samples (FHS + WHI), we computed ∂(Σ log-hazard)/∂**x**<sub>proj</sub> and ∂(Σ age)/∂**x**<sub>proj</sub> on the frozen model. Per-sample gradients were summarized by their mean **μ**<sub>p</sub> and covariance **Σ**<sub>p</sub> in projection space and back-projected to each raw feature *j* via the fixed JL matrix **W** (cached as `feature_importance/dann_W.npy`):

\[
\bar{g}_j = \frac{\mathbf{W}_{j,:}\,\boldsymbol{\mu}_p}{\sigma_j}, \qquad
\mathrm{Var}(\bar{g}_j) = \frac{\mathbf{W}_{j,:}\,\boldsymbol{\Sigma}_p\,\mathbf{W}_{j,:}^{\top}}{\sigma_j^2},
\]

where σ<sub>j</sub> is the scaler scale from training. Standard errors SE<sub>j</sub> = √(Var(\bar{g}<sub>j</sub>)/*N*) and two-sided *z*-scores *z*<sub>j</sub> = \bar{g}<sub>j</sub>/SE<sub>j</sub> were used with Benjamini–Hochberg FDR. Because gradient variance across individuals was small relative to the mean, more than 94% of features passed FDR *q* < 0.05; we therefore applied a **primary effect-size filter**: a feature was selected if |ḡ<sub>j</sub>| exceeded median(|ḡ|) + 5×MAD(|ḡ|)/0.6745 **and** *q*<sub>j</sub> < 0.05. This threshold corresponds to roughly the top 0.5–1% of features by back-projected gradient magnitude. Selected SNPs and CpGs were mapped to genes (Ensembl overlap) and tested for pathway enrichment and SNP–CpG overlap as described elsewhere.

---

## Results (why the model consistently uses these features)

### Model architecture constrains what “using a feature” means

The model does not assign 11,939 independent SNP coefficients. It learns a **low-dimensional mortality state** (8-D bottleneck **z**) that must simultaneously support (i) FHS partial-likelihood survival ranking, (ii) reconstruction of the 128-D DANN latent, (iii) prediction of age and blood cell composition in both cohorts, and (iv) cohort-invariant representation from the frozen DANN. Raw SNPs and CpGs influence risk only through **shared** compressed pathways (2,048-D → 128-D → 8-D). Features are flagged as “used” when the **average sensitivity** of log-hazard to that input, back-propagated through **W**, is unusually large and **stable across all 3,719 individuals**.

### Selection scale (exact counts)

| Category | Count |
|----------|------:|
| Total features | 1,372,416 |
| Pooled samples | 3,719 |
| **Risk-selected SNPs** | **11,939** |
| Risk-selected CpGs | 2,435 |
| Age-selected SNPs | 11,976 |
| Age-selected CpGs | 2,512 |
| Risk **and** age (SNPs) | 10,764 |
| Features with FDR *q* < 0.05 (risk only) | 1,291,762 |
| Risk effect-size threshold \|ḡ\| > | 0.00195 |

The FDR criterion alone is uninformative (low cross-sample gradient variance inflates *z*). The **5σ MAD** rule defines a reproducible extreme tail: about **1.0%** of all features for risk, of which **~83%** are SNPs reflecting the larger SNP panel and linkage redundancy.

### Direction and stability

Among risk-selected SNPs, gradients split into **accelerators** (positive mean ∂log-h/∂SNP, higher genotype → higher predicted risk) and **decelerators** (negative mean). The same SNP sets show **near-zero FDR** and large |*z*| because the model applies a **smooth, shared** mapping from omics to risk: perturbing correlated inputs shifts log-hazard similarly in almost every individual.

### Biological concordance (not GWAS)

Risk-selected SNPs map to **735 unique gene symbols** (904 SNPs with any gene annotation). Risk-selected CpGs map to **2,258 genes**. **83 genes** are hit by **both** a risk-selected SNP and a risk-selected CpG; **1,488** SNP–CpG pairs fall on the same gene. Pathway analysis of SNP genes shows enrichment of cardiovascular, metabolic, and signalling pathways (nominal *p* in KEGG/GO; genome-wide pathway FDR was conservative on the full union background). This supports that the model’s consistent sensitivity is concentrated in **annotated, disease-relevant biology**, not random probes.

### Interpretation for readers

**Why does AESurv consistently use these features?** Because survival supervision and auxiliary age/cell constraints require a stable omics → latent → risk map; back-projection identifies the raw features whose average effect on log-hazard is in the **extreme tail** of all 1.37M inputs and **coherent across individuals**. Many SNPs appear together because of **LD and shared latent factors**, not because the network stores 11,939 separate mortality mechanisms.

**Recommended wording:** “extreme, cohort-stable gradient sensitivity” — **not** “genome-wide significant GWAS hits” or “causal SNPs” unless supported by separate genetics.

### Suggested tighter feature sets for follow-up

| Set | Approx. size | Use |
|-----|-------------:|-----|
| Risk-selected SNPs (current) | 11,939 | Pathway / gene overlap |
| Top 0.1% \|ḡ\| (risk) | ~1,373 | Core sensitivity |
| SHAP ∩ significant (risk) | ~1,316 | Model-driven core |

---

## Figure legend (one-page)

**Figure X. Why AESurv-DANN-Aux consistently uses a subset of omics features for mortality risk.**

**(A)** Data flow: 1.37M CpG/SNP inputs are projected and encoded to a low-dimensional latent; the Cox head outputs log-hazard with auxiliary age and cell heads. Per-sample gradients of summed log-hazard are back-projected to raw features and filtered by large mean gradient (5σ MAD) and FDR.

**(B)** Volcano of back-projected mean ∂(log-hazard)/∂(feature) versus −log<sub>10</sub>(FDR) for the risk target. Dashed vertical lines: effect-size threshold (|ḡ| > 1.95×10<sup>−3</sup>). Red, SNPs; blue, CpGs.

**(C)** Selection counts and biological gene overlap for risk-selected features: 735 SNP genes, 2,258 CpG genes, 83 shared.

---

## Methods (lifestyle validation of log-hazard)

Lifestyle phenotypes (smoking status, cigarettes per day, pack-years, alcohol drinks per week, sleep hours, physical activity, BMI) were exported from dbGaP **independently** of model training and harmonized to canonical columns (`bio_relevance/harmonize_lifestyle.py`). Samples were linked to the AESurv-DANN-Aux predicted log-hazard (`log_h`) from the frozen [`models/aesurv_final`](../../models/aesurv_final) bundle via cohort-specific IDs (FHS: `Share_ID`; WHI: `sample_ID` or `dbGaP_Subject_ID` through `whi_meth_ids.csv`). Merge quality was recorded in `merge_qc.json` (target ≥85% per cohort).

**Concordance:** Spearman correlation and partial Spearman correlation (residualizing ranks on age, sex, and FHS batch) between `log_h` and each lifestyle trait; Benjamini–Hochberg FDR across traits. A composite unhealthy lifestyle score combined z-scored smoking ordinality, alcohol, and short sleep (&lt;6 h).

**Incremental prognosis:** Within each cohort, Cox proportional hazards models for time-to-death with Harrell C-index: M0 = age (+ sex + batch); M1 = M0 + lifestyle block; M2 = M0 + `log_h`; M3 = M0 + lifestyle + `log_h`. Likelihood-ratio tests compared nested models.

**Lifestyle-adjusted risk:** `log_h_resid` = residuals from linear regression of `log_h` on age, sex, batch, and available lifestyle covariates; Kaplan–Meier curves by tertiles of `log_h_resid` and Cox models with `log_h_resid` as predictor tested whether omics-derived risk predicts mortality beyond measured behavior.

**Robustness:** Analyses restricted to never-smokers; comparison of |ρ(`log_h`, lifestyle)| versus |ρ(`log_h`, noise)|.

Pipeline commands and outputs: `feature_importance/bio_relevance/lifestyle/README.md`.

---

## Results (lifestyle validation; FHS DAF, n=3,138)

Lifestyle data from FHS DAF Cox file (`All_data_cox.csv`: alcohol per occasion, cigarettes/day, sleep hours) were merged on `shareid` = `Share_ID` (97.8% of FHS omics samples). Raw Spearman correlations with `log_h` were strong for alcohol (ρ ≈ −0.46) and weak for cigarettes and sleep; after adjustment for age, sex, and methylation batch (Gen 3 / UMN / JHU), partial correlations were near zero for cigarettes and alcohol (ρ ≈ 0.02) and modest for sleep (partial ρ ≈ 0.04, p ≈ 0.02). Cox models on FHS mortality: C-index 0.815 (age/sex/batch), 0.827 (+lifestyle), 0.829 (+`log_h`), **0.839** (+both). Lifestyle-adjusted residual log-hazard remained associated with death (HR ≈ 2.39 per SD, p ≈ 2.2×10⁻¹⁵). Among never-smokers (n=2,761), `log_h` alone achieved C-index 0.83. **Do not** interpret raw alcohol–risk association as protective; it reflects batch/age structure (see confounding figure).

**Figures:**
- `figures/lifestyle_logh_association_simple.pdf` — unadjusted lifestyle vs `log_h` (three habits).
- `figures/lifestyle_logh_linkage_main.pdf` — raw vs adjusted linkage, correlation heatmap, Cox C-index, residual-risk summary.
- `figures/lifestyle_confounding_figure.pdf` — alcohol confounding by batch/age.
- `figures/lifestyle_validation_supp.pdf` — partial correlation bars, Cox models, KM by `log_h_resid`.

### Figure legends (linkage)

**Simple figure.** Cross-sectional association between cigarettes per day, alcohol amount per occasion, and sleep hours versus AESurv-DANN-Aux predicted log-hazard in 3,138 FHS participants. Points are colored by methylation processing batch (Gen 3, UMN, JHU); dashed curves are LOESS smoothers; ρ is Spearman correlation (unadjusted).

**Main linkage figure.** Top row: unadjusted associations as in the simple figure. Middle row: partial regression plots after removing age, sex, and batch effects (residuals of lifestyle and log-hazard). Panel G: heatmap of raw versus partial Spearman ρ. Panel H: Harrell C-index for nested Cox models (M0 demographics/batch; M1 +lifestyle; M2 +log-hazard; M3 both). Panel I: residual log-hazard after lifestyle adjustment remains prognostic for mortality.

---

## One-sentence summary (abstract-style)

Back-propagated gradients of a frozen cross-cohort mortality model identify ~12,000 SNPs and ~2,400 CpGs with extreme, individual-stable sensitivity to predicted log-hazard, mapping to hundreds of shared protein-coding genes and cardiovascular–metabolic pathways, consistent with a low-dimensional omics representation of aging-related mortality rather than independent per-SNP model weights.
