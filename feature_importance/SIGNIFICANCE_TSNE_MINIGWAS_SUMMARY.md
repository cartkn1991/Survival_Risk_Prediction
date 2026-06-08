# Statistical cutoff, latent t-SNE, and outlier mini-GWAS

This report ties together three add-on analyses on top of the deployed
AESURV-DANN-Aux survival model:

1. A **statistical cutoff** (effect-size + FDR) replacing the original top-200
   feature-importance ranking.
2. A **latent-space t-SNE comparison** of the raw projection vs the
   deterministic vs the stochastic latent representations.
3. A **mini-GWAS** on AESURV-significant SNPs contrasting "resilient"
   (lower-risk-than-expected-for-age) vs "accelerated"
   (higher-risk-than-expected-for-age) outlier individuals.

All artefacts are reproducible from the three scripts:

| step | script | inputs | outputs |
|---|---|---|---|
| (1) cutoff | `select_significant_features.py` | model bundle + JL projections | `feature_importance/significance/` |
| (2) t-SNE  | `tsne_latent_compare.py`         | model bundle + JL projections + risk NPZ | `plots/latent_tsne_*` |
| (3) GWAS   | `minigwas_outliers.py`           | (1) output + per-cohort NPZ bundles      | `feature_importance/minigwas/` |

---

## 1. Statistical cutoff for feature importance

### Why a new cutoff?

The original report used a fixed `top-K = 200`. To replace that with a
defensible **statistical** rule we propagate per-sample gradient mean *and*
variance through the linear back-projection (see math in
`select_significant_features.py` docstring). For each of the 1.37 M raw
features we then have

```
mean_x[j]   = (W[j,:] @ mu_proj) / sigma[j]
var_x[j]    = (W[j,:] @ Cov_proj @ W[j,:].T) / sigma[j]^2
z[j]        = mean_x[j] / sqrt(var_x[j] / N)
q[j]        = Benjamini-Hochberg FDR on the two-sided p-values
```

Because the model is **highly consistent across samples** (small per-sample
variance relative to the mean), naive FDR is uninformative -- > 94 % of features
"pass" `q < 0.05`. We therefore combine it with a robust **effect-size
threshold** on the back-projected gradient magnitude:

> A feature is *selected* iff `|mean_x[j]| > median + k_sigma * sigma_MAD` **and**
> `q[j] < FDR`, where `sigma_MAD = MAD(|mean_x|) / 0.6745` is the Gaussian-consistent
> robust scale.  Default `k_sigma = 5` (a 5-sigma outlier of the bulk
> distribution) and `FDR = 0.05`.

### Selection counts

|                | total      | CpG       | SNP        |
|----------------|-----------:|----------:|-----------:|
| all features   | 1,372,416  | 393,234   | 979,182    |
| **AGE selected** at 5*sigma_MAD & q<0.05 | **14,488** | 2,512 | 11,976 |
| **RISK selected** at 5*sigma_MAD & q<0.05 | **14,374** | 2,435 | 11,939 |
| **selected in EITHER**  | 15,957 | 2,806 | 13,151 |
| **selected in BOTH**    | 12,905 | 2,141 | 10,764 |

Reference tallies at alternative thresholds (for transparency):

| threshold (AGE) | features |
|---|---:|
| q < 0.05 (BH) | 1,290,824 |
| q < 1e-6 | 1,171,222 |
| Bonferroni @ 5 % | 1,145,927 |
| top 1 % by |grad| | 13,724 |
| top 0.5 % by |grad| | 6,863 |
| top 0.1 % by |grad| | 1,373 |

The 5-sigma_MAD threshold sits between *top 0.5 %* and *top 1 %* and is the
defensible "outlier" cut-off used downstream.

### Files

```
feature_importance/significance/
  significance_age.npz       mean_x / se / z / p / q + selected mask
  significance_risk.npz      same for log-hazard
  significant_age.csv        14,488 rows -- features with |mean| > 5*sigma_MAD & q < 0.05
  significant_risk.csv       14,374 rows
  significant_either.csv     union  (15,957 rows)
  volcano_age.png            volcano with the selection thresholds drawn
  volcano_risk.png           volcano (risk)
  summary.json               all counts + thresholds
```

The volcano plots show the bulk distribution (gray) and the selected set
(blue=CpG, red=SNP). The horizontal width of the selected band reflects the
5-sigma_MAD effect-size cutoff (vertical dashed lines).

---

## 2. Latent-space t-SNE comparison

A 4-row x 4-column figure (`plots/latent_tsne_grid.png`) shows t-SNE embeddings
of every representation that data passes through inside the model, coloured by
the four phenotypes of interest:

| row | representation              | dim   | what it shows                    |
|----:|-----------------------------|------:|----------------------------------|
| 1   | raw JL projection           | 2048  | what the model **sees**          |
| 2   | DANN deterministic latent (mu_DANN) | 128  | cohort-invariant projection        |
| 3   | AESURV deterministic latent (mu_AE)  | 8    | mean of the survival bottleneck    |
| 4   | AESURV stochastic latent (z = mu + eps * sigma) | 8 | reparam-sampled bottleneck         |

| column | coloured by                  |
|--------|------------------------------|
| 1      | chronological age            |
| 2      | predicted log-hazard         |
| 3      | follow-up time (deaths circled) |
| 4      | cohort (FHS vs WHI)          |

Key observations from the resulting figure:

* The **raw JL projection** has a single, weakly structured cloud with only a
  faint age gradient -- the model has not yet organised the data.
* The **DANN deterministic latent** is already much more organised: cohort
  becomes mixed (as intended by the adversary + MMD) yet age starts to span the
  axis monotonically.
* The **AESURV deterministic latent** shows a striking diagonal organisation by
  age and risk in 8 dimensions -- this is where the model performs the bulk of
  its compression of the biology that drives survival.
* The **AESURV stochastic latent** is structurally the same as the
  deterministic latent but with small noise added by `eps * sigma`, confirming
  that the VAE bottleneck does not collapse and remains close to its mean.

A 1-row "summary" figure is also written for each colouring
(`latent_tsne_by_age.png`, `latent_tsne_by_risk.png`, `latent_tsne_by_time.png`,
`latent_tsne_by_cohort.png`) for use in slides. The raw 2-D coordinates are
stored in `plots/latent_tsne_embeddings.npz` for re-plotting.

---

## 3. Mini-GWAS: resilient vs accelerated outliers

### Problem with the "low risk + old" / "high risk + young" definition

Predicted risk and age are extremely correlated in the pooled set:
**r(age, risk) = 0.83 (FHS r=0.82, WHI r=0.74)**. A global quantile-based
definition therefore collapses to ~3 samples per group. The well-known
*epigenetic-age-acceleration* trick fixes this: regress risk on age + cohort
and use the **residual** as the outlier statistic.

Fit (on the pooled set):

```
risk = -3.508 + 0.0545 * age + 0.027 * cohort
```

The age-decoupled outlier groups (top/bottom 10 % of residual) are:

| group        | definition                                       | n (FHS / WHI) |
|--------------|--------------------------------------------------|--------------:|
| resilient    | predicted risk **lower** than age would predict | 372 (368 / 4) |
| accelerated  | predicted risk **higher** than age would predict| 372 (363 / 9) |

`group_definition.png` plots the scatter with the two cohort fits and the two
outlier tails. (Within-cohort and global definitions remain available via
`--grouping within-cohort` and `--grouping global`.)

### Test setup

For every AESURV-RISK-significant SNP (11,939) with MAF >= 0.01 and at least 3
minor alleles per group, fit by IRLS:

```
logit( P(accelerated) )  =  beta_0  +  beta_g * dose  +  beta_c * cohort
```

10,941 SNPs pass QC (the other 998 are too rare in this 744-sample subset).

### Inflation / calibration

QQ plot (`minigwas_qq.png`): **lambda_GC = 1.089**. The test is essentially well
calibrated, with only the mild inflation expected of a model-guided focused scan.

### Top hits

No SNP reaches genome-wide Bonferroni (cutoff `p < 4.6e-6` after correcting for
10,941 tests) or BH FDR 5 % -- expected because the case+control pool is only
744 individuals. Several variants reach **nominal-suggestive** significance,
mostly inside or near genes with plausible aging / TGF-beta / cell-cycle roles.

| rank | SNP                  | chr | OR (acc.) |    p     | gene (Ensembl) | model also AGE-sig? |
|-----:|----------------------|-----|----------:|---------:|---------------|----------------------|
| 1    | 9:31514251_A         | 9   | 3.34 | 1.8e-4 | -            | yes |
| 2    | 12:115151255_T       | 12  | 0.28 | 2.0e-3 | -            | yes |
| 3    | 5:21722174_T         | 5   | 3.11 | 2.5e-3 | GUSBP1       | yes |
| 4    | 2:192722603_A        | 2   | 0.44 | 2.9e-3 | -            | yes |
| 5    | 4:15317738_A         | 4   | 0.39 | 3.0e-3 | C1QTNF7-AS1  | yes |
| 6    | 2:181566271_T        | 2   | 5.04 | 3.5e-3 | CERKL        | yes |
| 7    | 2:33238426_C         | 2   | 3.89 | 3.6e-3 | LTBP1        | yes |
| 8    | 11:44759309_C        | 11  | 0.38 | 3.7e-3 | TSPAN18      | yes |
| 9    | 1:179995624_T        | 1   | 2.34 | 3.8e-3 | CEP350       | yes |
| 10   | 12:74569580_G        | 12  | 0.17 | 4.1e-3 | -            | yes |
| 11   | 3:69766936_A         | 3   | 3.50 | 4.2e-3 | MITF         | yes |
| 12   | 12:59290104_G        | 12  | 6.00 | 4.4e-3 | -            | no  |
| 13   | 5:21721984_C         | 5   | 2.67 | 5.0e-3 | GUSBP1       | yes |
| 14   | 6:112031098_C        | 6   | 2.01 | 5.0e-3 | -            | yes |
| 15   | 1:75765991_A         | 1   | 0.33 | 5.1e-3 | ACADM        | yes |
| 16   | 3:56327103_G         | 3   | 4.75 | 5.2e-3 | ERC2         | yes |
| 17   | 4:73550401_C         | 4   | 0.53 | 5.3e-3 | -            | yes |
| 18   | 5:126991637_T        | 5   | 0.59 | 5.3e-3 | MARCHF3      | yes |
| 19   | 19:40626760_C        | 19  | 2.33 | 5.3e-3 | LTBP4        | yes |
| 20   | 3:16925357_T         | 3   | 0.37 | 5.4e-3 | -            | yes |

**Pattern observed**: two of the top 20 hits are in **LTBP1 / LTBP4** -- both
latent TGF-beta-binding proteins, a pathway strongly implicated in cellular
senescence and tissue aging. Other hits include **MITF** (oxidative stress
regulation in immune cells), **ACADM** (mitochondrial beta-oxidation, lethal
inborn-error genes), **CEP350** (centrosomal / cell-cycle), and **CERKL**
(ceramide kinase-like, lipid signalling). 19/20 are *also* in the
age-significant model set, suggesting the model is consistent about which
variants drive its age and risk predictions.

### Files

```
feature_importance/minigwas/
  group_definition.png    age vs predicted log-hazard with the two cohort fits & outlier tails
  group_definition.csv    sample-level group labels
  minigwas_results.csv    full per-SNP table (10,941 rows): OR, SE, p, q, genotypes, gene
  minigwas_top_hits.csv   top 100 by p, with live Ensembl gene annotation
  minigwas_manhattan.png  chromosomal Manhattan plot with top-8 labels and Bonferroni line
  minigwas_qq.png         QQ plot with lambda_GC
  summary.json            n_resolved, n_after_qc, group counts, top 10 list
```

---

## How to reproduce

```bash
# (1) Statistical cutoff (~22 s wall, requires feature_importance/dann_W.npy in cache)
python select_significant_features.py --fdr 0.05 --k-sigma 5.0

# (2) Latent-space t-SNE (~40 s)
python tsne_latent_compare.py

# (3) Mini-GWAS (~25 s + ~25 s Ensembl REST on first run)
python minigwas_outliers.py --grouping residual --residual-q-lo 0.10 --residual-q-hi 0.90
```
