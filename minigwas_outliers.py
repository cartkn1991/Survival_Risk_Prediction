#!/usr/bin/env python3
"""Mini-GWAS on AESURV-significant SNPs: resilient (low-risk, old) vs accelerated
(high-risk, young).

This is an *outlier-contrast* case-control GWAS *restricted* to the SNP set the
AESURV-DANN-Aux model itself flagged as important.

Outlier groups (combined FHS + WHI):

    Resilient    = (predicted_risk <= risk_q_lo)   AND  (age >= age_q_hi)
    Accelerated  = (predicted_risk >= risk_q_hi)   AND  (age <= age_q_lo)

Default quantiles are 25% / 75%, which produces an interpretable, comparable
number of cases and controls.  Per SNP, we fit

    logit( P(accelerated) )  =  beta0  +  beta_g * dose  +  beta_c * cohort

via ``statsmodels`` Logit and report OR, SE, Wald p, BH-FDR-adjusted q.

Inputs / dependencies:
  --sig-csv          significant_risk.csv (or --sig-age-csv) from
                     select_significant_features.py
  --fhs-npz/--whi-npz  per-cohort cpgall/snpall NPZ bundles (have X_snp + event + time)
  --fhs-snp-txt / --whi-snp-txt   per-cohort SNP column names (one per line)
  --fhs-meta-pq / --whi-meta-pq   parquet metadata for true age
  --risk-npz         aesurv_aux_risk.npz from the training run

Outputs (under ``--out-dir``, default ``feature_importance/minigwas/``):
  minigwas_results.csv      per-SNP OR/SE/p/q + n_cases/n_controls
  minigwas_top_hits.csv     SNPs surviving genome-wide & BH cutoffs
  minigwas_manhattan.png    Manhattan-like plot (-log10 p vs SNP index sorted by genome position)
  group_definition.png      scatter age vs risk with the two outlier groups highlighted
  group_definition.csv      sample assignments
  summary.json              cohort/group tallies, top hits, parameters

Post-hoc figure (optional)::

  python feature_importance/minigwas/plot_minigwas_effects.py
    -> minigwas_effects_forest.png  (SNPs vs beta + CI; color = model accelerator/decelerator)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


_THRIFT_LIMIT = 2_147_483_647
_SNP_RE = re.compile(r"^(chr)?([0-9XYMTxymt]+):([0-9]+)")


def _open_pq(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(str(path), thrift_string_size_limit=_THRIFT_LIMIT,
                              thrift_container_size_limit=_THRIFT_LIMIT)
    except TypeError:
        return pq.ParquetFile(str(path))


def _read_age(parquet_path: Path, id_col: str) -> np.ndarray:
    pf = _open_pq(parquet_path)
    df = pf.read(columns=[id_col, "age"]).to_pandas()
    a = df["age"].to_numpy(dtype=np.float32)
    if np.isnan(a).any():
        a = np.where(np.isnan(a), np.nanmean(a), a)
    return a


def _parse_chr_pos(snp_name: str) -> Tuple[int, int]:
    """1, X, Y -> integer chromosome (X=23, Y=24, MT=25, other -> 26)."""
    m = _SNP_RE.match(snp_name)
    if not m:
        return 26, -1
    chrom_raw = m.group(2).upper()
    pos = int(m.group(3))
    if chrom_raw.isdigit():
        return int(chrom_raw), pos
    if chrom_raw == "X":
        return 23, pos
    if chrom_raw == "Y":
        return 24, pos
    return 25, pos


def _read_txt_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n").rstrip("\r") for line in fh if line.strip()]


def logit_irls(X: np.ndarray, y: np.ndarray, *, max_iter: int = 25,
               tol: float = 1e-6, ridge: float = 1e-8
               ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Lightweight logistic regression via IRLS / Newton-Raphson.

    Returns ``(beta, se, converged)`` where ``beta`` are coefficients (length p)
    and ``se`` are Wald standard errors.  Adds a tiny ridge to the Hessian for
    numerical stability with rare-allele predictors.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    last_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        # avoid degenerate weights
        w = np.clip(w, 1e-9, 0.25)
        XW = X * w[:, None]
        H = X.T @ XW + ridge * np.eye(p)
        score = X.T @ (y - mu)
        try:
            delta = np.linalg.solve(H, score)
        except np.linalg.LinAlgError:
            return beta, np.full(p, np.nan), False
        beta_new = beta + delta
        ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
        if abs(ll - last_ll) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
        last_ll = ll
    # Final Hessian for Wald SE
    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-9, 0.25)
    XW = X * w[:, None]
    H = X.T @ XW + ridge * np.eye(p)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return beta, np.full(p, np.nan), False
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, converged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sig-csv", type=str,
                   default="feature_importance/significance/significant_risk.csv",
                   help="Significant features (risk target) from select_significant_features.py")
    p.add_argument("--sig-age-csv", type=str,
                   default="feature_importance/significance/significant_age.csv",
                   help="Significant features for the age target -- used to add 'sig_age' annotation.")
    p.add_argument("--fhs-npz", type=str,
                   default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--whi-npz", type=str,
                   default="vae_cox_cache/bundles/WHI_raw_WHI_methylation_with_snp_merged_1milfeatures_combined_training_WHI_methylation_with_snp_merged_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-snp-txt", type=str,
                   default="FHS_methylation_with_snp_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--whi-snp-txt", type=str,
                   default="WHI_methylation_with_snp_merged_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--fhs-meta-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-meta-pq", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--whi-id-col", type=str, default="sample_ID")
    p.add_argument("--risk-npz", type=str,
                   default="runs/aesurv_aux_grid/age12.00_cell1.00/aesurv_aux_risk.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--grouping", type=str, default="residual",
                   choices=["residual", "within-cohort", "global"],
                   help="How to define resilient and accelerated groups. "
                        "'residual' regresses predicted_risk on age+cohort and takes the tails of the "
                        "residuals (age-decoupled). 'within-cohort' uses raw quantiles per cohort. "
                        "'global' pools and takes raw quantiles (collapses when age and risk are co-linear).")
    p.add_argument("--residual-q-lo", type=float, default=0.10,
                   help="In residual mode: lower quantile of the residual -> resilient (lower risk than expected for age).")
    p.add_argument("--residual-q-hi", type=float, default=0.90,
                   help="In residual mode: upper quantile of the residual -> accelerated (higher risk than expected for age).")
    p.add_argument("--risk-q-lo", type=float, default=0.25, help="Lower quantile for low risk (raw modes).")
    p.add_argument("--risk-q-hi", type=float, default=0.75, help="Upper quantile for high risk (raw modes).")
    p.add_argument("--age-q-lo",  type=float, default=0.25, help="Lower quantile for young (raw modes).")
    p.add_argument("--age-q-hi",  type=float, default=0.75, help="Upper quantile for old (raw modes).")
    p.add_argument("--maf-min",   type=float, default=0.01,
                   help="Minimum minor allele frequency in the case+control pool.")
    p.add_argument("--min-mac-per-group", type=int, default=3,
                   help="Minimum minor-allele *count* in each of resilient/accelerated groups.")
    p.add_argument("--missing-thresh", type=float, default=0.10,
                   help="Drop SNPs with > this fraction missing genotypes after restriction.")
    p.add_argument("--annot-csv", type=str, default="feature_importance/annot/snp_gene_cache.csv",
                   help="Optional gene-annotation cache (snp -> gene); merged onto top hits if present.")
    p.add_argument("--annot-gwas-csv", type=str, default="feature_importance/annot/snp_gwas_cache.csv",
                   help="Optional GWAS catalog cache; merged onto top hits if present.")
    p.add_argument("--out-dir", type=str, default="feature_importance/minigwas")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- 1.  Significant SNP list
    print(f"[1/7] Loading significant SNPs from {args.sig_csv} ...")
    sig = pd.read_csv(args.sig_csv)
    sig_snp = sig[sig["kind"] == "snp"].copy()
    sig_age = pd.DataFrame()
    if Path(args.sig_age_csv).exists():
        sig_age = pd.read_csv(args.sig_age_csv)
        sig_age = sig_age[sig_age["kind"] == "snp"]
    sig_age_set = set(sig_age["feature"].astype(str)) if not sig_age.empty else set()
    sig_snp_names = sig_snp["feature"].astype(str).tolist()
    print(f"  RISK-significant SNPs: {len(sig_snp_names):,}  (AGE-overlap: {len(set(sig_snp_names) & sig_age_set):,})")

    # ---- 2.  Build column-index maps per cohort
    print(f"\n[2/7] Resolving SNP column indices in each cohort ...")
    fhs_snp_names = _read_txt_list(Path(args.fhs_snp_txt))
    whi_snp_names = _read_txt_list(Path(args.whi_snp_txt))
    fhs_map = {s: i for i, s in enumerate(fhs_snp_names)}
    whi_map = {s: i for i, s in enumerate(whi_snp_names)}
    cols_fhs = []; cols_whi = []; resolved = []
    for name in sig_snp_names:
        i_f = fhs_map.get(name); i_w = whi_map.get(name)
        if i_f is None or i_w is None:
            continue
        cols_fhs.append(i_f); cols_whi.append(i_w); resolved.append(name)
    cols_fhs = np.asarray(cols_fhs, dtype=np.int64)
    cols_whi = np.asarray(cols_whi, dtype=np.int64)
    resolved = np.asarray(resolved, dtype=object)
    print(f"  resolved {resolved.size:,} / {len(sig_snp_names):,} SNPs in both cohorts")
    if resolved.size == 0:
        raise SystemExit("No significant SNPs resolved to genotype columns; aborting.")

    # ---- 3.  Memory-map genotype matrices and slice the columns of interest
    print(f"\n[3/7] Memory-mapping genotype bundles + slicing columns ...")
    t0 = time.time()
    fhs_z = np.load(args.fhs_npz, allow_pickle=False, mmap_mode="r")
    whi_z = np.load(args.whi_npz, allow_pickle=False, mmap_mode="r")
    X_snp_fhs_full = fhs_z["X_snp"]    # (3209, ~1.04M) int8
    X_snp_whi_full = whi_z["X_snp"]    # (510,  ~1.04M) int8
    print(f"  FHS X_snp={X_snp_fhs_full.shape}  WHI X_snp={X_snp_whi_full.shape}")
    # Sort columns ascending for sequential disk access, then unscramble after read
    order_f = np.argsort(cols_fhs); order_w = np.argsort(cols_whi)
    G_fhs_sorted = X_snp_fhs_full[:, cols_fhs[order_f]].astype(np.float32)
    G_whi_sorted = X_snp_whi_full[:, cols_whi[order_w]].astype(np.float32)
    # Unscramble back to the order of ``resolved``
    inv_f = np.argsort(order_f); inv_w = np.argsort(order_w)
    G_fhs = G_fhs_sorted[:, inv_f]
    G_whi = G_whi_sorted[:, inv_w]
    print(f"  loaded G_fhs={G_fhs.shape}  G_whi={G_whi.shape}   in {time.time() - t0:.1f}s")

    # Treat negative / >2 values as missing (typical for int8 PLINK encoding)
    G_fhs = np.where((G_fhs >= 0) & (G_fhs <= 2), G_fhs, np.nan)
    G_whi = np.where((G_whi >= 0) & (G_whi <= 2), G_whi, np.nan)

    # ---- 4.  Age + predicted risk per sample (full FHS + WHI)
    print(f"\n[4/7] Loading age + risk per sample ...")
    age_fhs = _read_age(Path(args.fhs_meta_pq), args.fhs_id_col)
    age_whi = _read_age(Path(args.whi_meta_pq), args.whi_id_col)
    n_fhs = G_fhs.shape[0]; n_whi = G_whi.shape[0]
    if age_fhs.shape[0] != n_fhs:
        raise SystemExit(f"FHS age {age_fhs.shape} != bundle rows {n_fhs}")
    if age_whi.shape[0] != n_whi:
        raise SystemExit(f"WHI age {age_whi.shape} != bundle rows {n_whi}")

    rd = dict(np.load(args.risk_npz, allow_pickle=False))
    risk_whi = rd["risk_whi_test"]; assert risk_whi.shape[0] == n_whi

    # Reconstruct FHS risk in original order via the same train/val split
    event_arr_full = fhs_z["event"].astype(np.int32)
    idx = np.arange(n_fhs)
    strat = event_arr_full if event_arr_full.sum() >= 2 else None
    tr_idx, va_idx = train_test_split(idx, test_size=args.val_frac, random_state=args.seed,
                                      stratify=strat, shuffle=True)
    risk_fhs_full = np.zeros(n_fhs, dtype=np.float32)
    if len(tr_idx) == rd["risk_fhs_train"].size and len(va_idx) == rd["risk_fhs_val"].size:
        risk_fhs_full[tr_idx] = rd["risk_fhs_train"]
        risk_fhs_full[va_idx] = rd["risk_fhs_val"]
        print(f"  FHS risk recovered via reproduced split (val={len(va_idx)})")
    else:
        print(f"  WARNING: split shapes mismatch (tr={len(tr_idx)},val={len(va_idx)}) vs "
              f"risk_fhs_train={rd['risk_fhs_train'].size}; falling back to train+val concat order.")
        risk_fhs_full[:rd["risk_fhs_train"].size] = rd["risk_fhs_train"]
        risk_fhs_full[rd["risk_fhs_train"].size:rd["risk_fhs_train"].size + rd["risk_fhs_val"].size] = rd["risk_fhs_val"]

    # Pool samples
    age_all = np.concatenate([age_fhs, age_whi], axis=0).astype(np.float32)
    risk_all = np.concatenate([risk_fhs_full, risk_whi], axis=0).astype(np.float32)
    cohort_all = np.concatenate([np.zeros(n_fhs, dtype=np.int8), np.ones(n_whi, dtype=np.int8)])
    G_all = np.concatenate([G_fhs, G_whi], axis=0)  # (3719, K)
    print(f"  pooled samples N={age_all.size}   genotype matrix={G_all.shape}")

    # ---- 5.  Define outlier groups
    print(f"\n[5/7] Defining resilient and accelerated groups  (mode={args.grouping}) ...")
    grouping_info: Dict[str, float | str] = {"mode": args.grouping}
    if args.grouping == "residual":
        # Regress predicted risk on age + cohort, take residual tails
        X_cov = np.column_stack([np.ones(age_all.size, dtype=np.float64),
                                  age_all.astype(np.float64),
                                  cohort_all.astype(np.float64)])
        coef, *_ = np.linalg.lstsq(X_cov, risk_all.astype(np.float64), rcond=None)
        risk_fit = X_cov @ coef
        risk_resid = risk_all.astype(np.float64) - risk_fit
        resid_lo = float(np.quantile(risk_resid, args.residual_q_lo))
        resid_hi = float(np.quantile(risk_resid, args.residual_q_hi))
        print(f"  fit: risk = {coef[0]:.3f} + {coef[1]:.4f}*age + {coef[2]:.3f}*cohort  ")
        print(f"  residual cutoffs: q{args.residual_q_lo*100:.0f}={resid_lo:.3f}  q{args.residual_q_hi*100:.0f}={resid_hi:.3f}")
        resilient_mask   = risk_resid <= resid_lo                       # lower risk than expected (good)
        accelerated_mask = risk_resid >= resid_hi                       # higher risk than expected (bad)
        grouping_info.update(intercept=float(coef[0]), beta_age=float(coef[1]), beta_cohort=float(coef[2]),
                             resid_lo=resid_lo, resid_hi=resid_hi)
    elif args.grouping == "within-cohort":
        resilient_mask = np.zeros(age_all.size, dtype=bool)
        accelerated_mask = np.zeros(age_all.size, dtype=bool)
        for c, c_label in [(0, "FHS"), (1, "WHI")]:
            m = cohort_all == c
            risk_lo_c = float(np.quantile(risk_all[m], args.risk_q_lo))
            risk_hi_c = float(np.quantile(risk_all[m], args.risk_q_hi))
            age_lo_c  = float(np.quantile(age_all[m],  args.age_q_lo))
            age_hi_c  = float(np.quantile(age_all[m],  args.age_q_hi))
            print(f"  [{c_label}]  risk_lo={risk_lo_c:.3f}  risk_hi={risk_hi_c:.3f}  "
                  f"age_lo={age_lo_c:.1f}  age_hi={age_hi_c:.1f}")
            resilient_mask |= m & (risk_all <= risk_lo_c) & (age_all >= age_hi_c)
            accelerated_mask |= m & (risk_all >= risk_hi_c) & (age_all <= age_lo_c)
    else:  # global
        risk_lo = float(np.quantile(risk_all, args.risk_q_lo))
        risk_hi = float(np.quantile(risk_all, args.risk_q_hi))
        age_lo = float(np.quantile(age_all, args.age_q_lo))
        age_hi = float(np.quantile(age_all, args.age_q_hi))
        print(f"  quantiles: risk_lo={risk_lo:.3f}  risk_hi={risk_hi:.3f}  "
              f"age_lo={age_lo:.1f}  age_hi={age_hi:.1f}")
        grouping_info.update(risk_lo=risk_lo, risk_hi=risk_hi, age_lo=age_lo, age_hi=age_hi)
        resilient_mask   = (risk_all <= risk_lo) & (age_all >= age_hi)
        accelerated_mask = (risk_all >= risk_hi) & (age_all <= age_lo)

    keep_mask = resilient_mask | accelerated_mask
    y = accelerated_mask.astype(np.int32)
    cohort_keep = cohort_all[keep_mask]
    n_acc = int(accelerated_mask.sum()); n_res = int(resilient_mask.sum())
    print(f"  resilient (less risk than expected)   : {n_res}  ({(cohort_all[resilient_mask] == 0).sum()} FHS / "
          f"{(cohort_all[resilient_mask] == 1).sum()} WHI)")
    print(f"  accelerated (more risk than expected) : {n_acc}  ({(cohort_all[accelerated_mask] == 0).sum()} FHS / "
          f"{(cohort_all[accelerated_mask] == 1).sum()} WHI)")

    # Save the group plot + assignment csv
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(age_all[~keep_mask], risk_all[~keep_mask], s=10, c="#d0d0d0",
               linewidths=0, alpha=0.45, label=f"other (n={int((~keep_mask).sum())})")
    ax.scatter(age_all[resilient_mask], risk_all[resilient_mask], s=22, c="#27ae60",
               edgecolors="black", linewidths=0.35,
               label=f"resilient: less risk than expected (n={n_res})")
    ax.scatter(age_all[accelerated_mask], risk_all[accelerated_mask], s=22, c="#c0392b",
               edgecolors="black", linewidths=0.35,
               label=f"accelerated: more risk than expected (n={n_acc})")
    if args.grouping == "residual":
        ages_line = np.linspace(age_all.min(), age_all.max(), 50)
        for c, c_label, c_color in [(0, "FHS", "#3498db"), (1, "WHI", "#e67e22")]:
            risk_fit_line = grouping_info["intercept"] + grouping_info["beta_age"] * ages_line + grouping_info["beta_cohort"] * c
            ax.plot(ages_line, risk_fit_line, color=c_color, lw=1.2, ls="--",
                    label=f"linear fit {c_label}")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted log-hazard (AESURV)")
    title = (
        f"Outlier-group definition for mini-GWAS  (grouping = {args.grouping})\n"
        if args.grouping != "residual" else
        f"Outlier-group definition for mini-GWAS  (age-residualised risk)\n"
        f"resilient = residual < q{args.residual_q_lo*100:.0f},  accelerated = residual > q{args.residual_q_hi*100:.0f}"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "group_definition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'group_definition.png'}")

    pd.DataFrame({
        "sample_idx_pooled": np.arange(age_all.size, dtype=np.int32),
        "cohort": np.where(cohort_all == 0, "FHS", "WHI"),
        "age":    age_all,
        "predicted_risk": risk_all,
        "group":  np.where(resilient_mask, "resilient",
                  np.where(accelerated_mask, "accelerated", "other")),
    }).to_csv(out_dir / "group_definition.csv", index=False)

    if n_acc < 5 or n_res < 5:
        raise SystemExit("Too few outliers to fit logistic models (need >=5 in each group).")

    # ---- 6.  Per-SNP logistic regression
    print(f"\n[6/7] Fitting per-SNP logistic regression ...")
    # Subset matrix to the case+control rows for efficiency
    G_use = G_all[keep_mask, :].astype(np.float32)   # (N_use, K)
    cohort_use = cohort_keep.astype(np.float32)
    n_use, K = G_use.shape
    print(f"  matrix N={n_use}  SNPs K={K}")

    results = {
        "snp": [],
        "n_used": [],
        "maf": [],
        "or_acc": [],
        "beta": [],
        "se": [],
        "z": [],
        "p": [],
        "beta_cohort": [],
        "cases_AA": [], "cases_AB": [], "cases_BB": [],
        "ctrls_AA": [], "ctrls_AB": [], "ctrls_BB": [],
    }
    t0 = time.time(); n_skipped_maf = 0; n_skipped_miss = 0; n_skipped_var = 0; n_failed = 0
    intercept = np.ones(n_use, dtype=np.float32)
    cov_cohort = cohort_use

    for j in range(K):
        g = G_use[:, j]
        good = ~np.isnan(g)
        if good.sum() < max(20, int(0.6 * n_use)):
            n_skipped_miss += 1
            continue
        gn = g[good]
        if (1 - good.mean()) > args.missing_thresh:
            n_skipped_miss += 1
            continue
        mean_g = float(np.mean(gn))
        maf = min(mean_g / 2.0, 1.0 - mean_g / 2.0)
        if maf < args.maf_min:
            n_skipped_maf += 1
            continue
        if np.ptp(gn) == 0:
            n_skipped_var += 1
            continue
        y_j = (accelerated_mask[keep_mask])[good].astype(np.float32)
        cohort_j = cov_cohort[good]
        if y_j.sum() < 3 or (1 - y_j).sum() < 3:
            n_skipped_var += 1
            continue
        # Require enough minor-allele count in each group (dominant collapsed)
        # Use the count of *minor* alleles, not the dose, to make this robust to
        # which allele is the reference in the encoding.
        af = float(np.mean(gn) / 2.0)
        minor_is_alt = af <= 0.5
        dose_minor = gn if minor_is_alt else (2.0 - gn)
        mac_cases = int(dose_minor[y_j == 1].sum())
        mac_ctrls = int(dose_minor[y_j == 0].sum())
        if min(mac_cases, mac_ctrls) < args.min_mac_per_group:
            n_skipped_maf += 1
            continue
        X = np.column_stack([np.ones(gn.size, dtype=np.float64),
                              gn.astype(np.float64),
                              cohort_j.astype(np.float64)])
        beta_v, se_v, conv = logit_irls(X, y_j.astype(np.float64))
        if not conv or not np.isfinite(beta_v[1]) or not np.isfinite(se_v[1]) or se_v[1] <= 0:
            n_failed += 1
            continue
        beta = float(beta_v[1]); se = float(se_v[1])
        zval = beta / se
        pval = float(2.0 * stats.norm.sf(abs(zval)))
        beta_c = float(beta_v[2])
        # genotype tallies
        cases = y_j == 1; ctrls = y_j == 0
        c_aa = int(((gn == 0) & cases).sum()); c_ab = int(((gn == 1) & cases).sum()); c_bb = int(((gn == 2) & cases).sum())
        x_aa = int(((gn == 0) & ctrls).sum()); x_ab = int(((gn == 1) & ctrls).sum()); x_bb = int(((gn == 2) & ctrls).sum())
        results["snp"].append(resolved[j])
        results["n_used"].append(int(good.sum()))
        results["maf"].append(maf)
        results["or_acc"].append(float(np.exp(beta)))
        results["beta"].append(beta)
        results["se"].append(se)
        results["z"].append(zval)
        results["p"].append(pval)
        results["beta_cohort"].append(beta_c)
        results["cases_AA"].append(c_aa); results["cases_AB"].append(c_ab); results["cases_BB"].append(c_bb)
        results["ctrls_AA"].append(x_aa); results["ctrls_AB"].append(x_ab); results["ctrls_BB"].append(x_bb)

    elapsed = time.time() - t0
    n_fit = len(results["snp"])
    print(f"  fit {n_fit:,} models in {elapsed:.1f}s  "
          f"(skipped maf={n_skipped_maf:,}  miss={n_skipped_miss:,}  "
          f"variance={n_skipped_var:,}  failed={n_failed:,})")

    df = pd.DataFrame(results)
    if df.empty:
        raise SystemExit("No SNPs passed QC; aborting.")
    # BH FDR
    m = len(df)
    order = df["p"].sort_values().index.to_numpy()
    p_sorted = df.loc[order, "p"].to_numpy()
    ranks = np.arange(1, m + 1, dtype=np.float64)
    bh = p_sorted * m / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1].clip(0, 1)
    q = np.empty(m, dtype=np.float64); q[order] = bh
    df["q"] = q
    # chrom / pos for plotting
    chrom_pos = df["snp"].map(_parse_chr_pos)
    df["chrom"] = chrom_pos.map(lambda t: t[0])
    df["pos"] = chrom_pos.map(lambda t: t[1])
    df["sig_age"] = df["snp"].isin(sig_age_set)
    df = df.sort_values(["p", "chrom", "pos"], kind="mergesort").reset_index(drop=True)

    # Optional gene annotation
    if Path(args.annot_csv).exists():
        try:
            ann = pd.read_csv(args.annot_csv)
            snp_col_ann = [c for c in ann.columns if c.lower() in ("feature", "feature_id", "snp")]
            if snp_col_ann:
                ann = ann.rename(columns={snp_col_ann[0]: "snp"})
                gene_cols = [c for c in ann.columns if "gene" in c.lower()]
                ann_keep = ann[["snp"] + gene_cols].drop_duplicates("snp", keep="first")
                df = df.merge(ann_keep, on="snp", how="left")
                print(f"  merged gene annotations from {args.annot_csv}")
        except Exception as e:
            print(f"  could not merge annotations from {args.annot_csv}: {e}")
    # Optional GWAS catalog annotation
    if Path(args.annot_gwas_csv).exists():
        try:
            gw = pd.read_csv(args.annot_gwas_csv)
            snp_col = [c for c in gw.columns if c.lower() in ("feature", "feature_id", "snp")]
            if snp_col:
                gw = gw.rename(columns={snp_col[0]: "snp"})
                keep = ["snp"] + [c for c in gw.columns if c != "snp"]
                gw_keep = gw[keep].drop_duplicates("snp", keep="first")
                df = df.merge(gw_keep, on="snp", how="left")
                print(f"  merged GWAS catalog from {args.annot_gwas_csv}")
        except Exception as e:
            print(f"  could not merge GWAS from {args.annot_gwas_csv}: {e}")

    df.to_csv(out_dir / "minigwas_results.csv", index=False)
    print(f"  wrote {out_dir / 'minigwas_results.csv'}")

    bonferroni = 0.05 / m
    top = df[df["q"] < 0.05].copy()
    print(f"  m={m:,}  Bonferroni cutoff p<{bonferroni:.2e}  passed: {int((df['p'] < bonferroni).sum()):,}")
    print(f"  BH FDR<0.05 passed: {len(top):,}")

    # Always emit a "top suggestive" list (top 100 by p) for downstream annotation
    # even if nothing survives Bonferroni / BH (typical with N<1000 outlier scans).
    n_show = max(len(top), 100)
    top_show = df.head(n_show).copy()

    # Live Ensembl gene annotation for the top hits (incremental cache update)
    try:
        from annotate_top_features import fetch_snp_genes
        existing = {}
        if Path(args.annot_csv).exists():
            cache_df = pd.read_csv(args.annot_csv)
            id_col = next((c for c in cache_df.columns if c.lower() in ("feature_id", "feature", "snp")), None)
            gene_col = next((c for c in cache_df.columns if "gene" in c.lower()), None)
            if id_col is not None and gene_col is not None:
                existing = dict(zip(cache_df[id_col].astype(str), cache_df[gene_col].astype(str)))
        top_snp_ids = top_show["snp"].astype(str).tolist()
        snp_gene_cache = fetch_snp_genes(top_snp_ids, existing, out_csv=Path(args.annot_csv),
                                         sleep_sec=0.05)
        live_gene = top_show["snp"].map(lambda s: snp_gene_cache.get(str(s), "")).astype(str)
        if "gene_ensembl" in top_show.columns:
            existing_gene = top_show["gene_ensembl"].fillna("").astype(str)
            # Use existing if non-empty/non-"nan"; otherwise live cache lookup
            valid = existing_gene.where(
                (existing_gene != "") & (existing_gene.str.lower() != "nan"), other=""
            )
            top_show["gene_ensembl"] = valid.where(valid != "", other=live_gene)
        else:
            top_show["gene_ensembl"] = live_gene
        print(f"  live Ensembl annotation: filled "
              f"{int((top_show['gene_ensembl'].fillna('') != '').sum())}/{len(top_show)} top hits")
    except Exception as e:
        print(f"  live Ensembl annotation failed: {e}; skipping")

    top_show.to_csv(out_dir / "minigwas_top_hits.csv", index=False)
    print(f"  wrote {out_dir / 'minigwas_top_hits.csv'}  (n={len(top_show):,})")

    # ---- 7.  Manhattan plot
    print(f"\n[7/7] Rendering Manhattan plot ...")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    df_plot = df.sort_values(["chrom", "pos"]).reset_index(drop=True)
    # Build x positions stacked across chromosomes
    chroms = sorted(df_plot["chrom"].unique())
    x_offsets = {}; cum = 0; tick_pos = []; tick_lab = []
    palette = ["#34495e", "#7f8c8d"]
    for i, ch in enumerate(chroms):
        mask = df_plot["chrom"] == ch
        df_ch = df_plot[mask]
        x = cum + np.arange(len(df_ch))
        ax.scatter(x, -np.log10(df_ch["p"].clip(lower=1e-300)),
                   c=palette[i % 2], s=6, linewidths=0, alpha=0.85)
        x_offsets[ch] = (cum, cum + len(df_ch))
        tick_pos.append(cum + len(df_ch) / 2)
        # Friendly chromosome label
        if ch <= 22: lab = str(int(ch))
        elif ch == 23: lab = "X"
        elif ch == 24: lab = "Y"
        elif ch == 25: lab = "MT"
        else: lab = "?"
        tick_lab.append(lab)
        cum += len(df_ch)
    # Annotate top 8 hits with SNP name
    top_show = df.nsmallest(8, "p")
    for _, row in top_show.iterrows():
        ch = row["chrom"]
        ch_df = df_plot[df_plot["chrom"] == ch].reset_index(drop=True)
        try:
            rel_idx = ch_df.index[ch_df["snp"] == row["snp"]][0]
        except IndexError:
            continue
        x_pos = x_offsets[ch][0] + rel_idx
        ax.annotate(row["snp"], xy=(x_pos, -np.log10(max(row["p"], 1e-300))),
                    xytext=(0, 9), textcoords="offset points", fontsize=7.5,
                    ha="center", color="#1a1a1a")
    ax.axhline(-np.log10(bonferroni), color="#c0392b", lw=0.8, ls="--",
               label=f"Bonferroni (p={bonferroni:.1e})")
    ax.axhline(-np.log10(0.05), color="#7f8c8d", lw=0.8, ls=":",
               label="p=0.05")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(f"Mini-GWAS: resilient (low-risk old, n={n_res}) vs "
                 f"accelerated (high-risk young, n={n_acc})\n"
                 f"restricted to {K:,} AESURV-significant SNPs (RISK target)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "minigwas_manhattan.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'minigwas_manhattan.png'}")

    # QQ plot for distribution diagnostic
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    p_obs = df["p"].sort_values().to_numpy()
    p_exp = (np.arange(1, len(p_obs) + 1) - 0.5) / len(p_obs)
    ax.scatter(-np.log10(p_exp), -np.log10(p_obs.clip(min=1e-300)),
               s=6, c="#2c3e50", linewidths=0)
    mx = max(-np.log10(p_exp.min()), -np.log10(p_obs.min().clip(min=1e-300)))
    ax.plot([0, mx], [0, mx], color="#c0392b", lw=1)
    # genomic inflation lambda (from chi2)
    chi2 = stats.chi2.isf(p_obs.clip(min=1e-300), df=1)
    lam = float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))
    ax.set_title(f"QQ plot of mini-GWAS p-values   (lambda_GC = {lam:.3f})", fontsize=11)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "minigwas_qq.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'minigwas_qq.png'}")

    # ---- Summary
    summary = {
        "n_significant_input_snps": int(len(sig_snp_names)),
        "n_resolved_in_both_cohorts": int(resolved.size),
        "n_after_qc": int(m),
        "n_resilient": int(n_res),
        "n_accelerated": int(n_acc),
        "grouping": grouping_info,
        "qc_thresholds": {
            "maf_min": args.maf_min, "missing_thresh": args.missing_thresh,
        },
        "bonferroni_p": bonferroni,
        "n_pass_bonferroni": int((df["p"] < bonferroni).sum()),
        "n_pass_bh_fdr_5pc": int((df["q"] < 0.05).sum()),
        "lambda_gc": lam,
        "top10": df.head(10)[["snp", "chrom", "pos", "or_acc", "p", "q", "sig_age"]].to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    print(f"\nDone. Mini-GWAS outputs in {out_dir}/")


if __name__ == "__main__":
    main()
