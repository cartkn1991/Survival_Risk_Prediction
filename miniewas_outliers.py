#!/usr/bin/env python3
"""Mini-EWAS on AESURV-significant CpGs: resilient vs accelerated outlier groups.

Mirrors minigwas_outliers.py but tests model-selected CpGs with:

    logit( P(accelerated) ) = beta0 + beta_meth * methylation + beta_cohort * cohort

Outputs (under --out-dir):
  miniewas_results.csv
  miniewas_top_hits.csv
  miniewas_manhattan.png
  miniewas_qq.png
  group_definition.png / group_definition.csv
  summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from gwas.gwas_common import (
    annotate_cpg_df,
    load_cpg_annotation,
    load_pooled_logh_from_parquet,
    read_age_sex,
    read_txt_list,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def logit_irls(X: np.ndarray, y: np.ndarray, *, max_iter: int = 25, ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, bool]:
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    last_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-9, 0.25)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(p)
        score = X.T @ (y - mu)
        try:
            beta = beta + np.linalg.solve(H, score)
        except np.linalg.LinAlgError:
            return beta, np.full(p, np.nan), False
        ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
        if abs(ll - last_ll) < 1e-6:
            converged = True
            break
        last_ll = ll
    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-9, 0.25)
    H = X.T @ (X * w[:, None]) + ridge * np.eye(p)
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        return beta, np.full(p, np.nan), False
    return beta, se, converged


def _read_age(parquet_path: Path, id_col: str) -> np.ndarray:
    from gwas.gwas_common import open_pq
    pf = open_pq(parquet_path)
    age = pf.read(columns=[id_col, "age"]).to_pandas()["age"].to_numpy(dtype=np.float32)
    if np.isnan(age).any():
        age = np.where(np.isnan(age), np.nanmean(age), age)
    return age


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sig-csv", default="feature_importance/significance/significant_risk.csv")
    p.add_argument("--sig-age-csv", default="feature_importance/significance/significant_age.csv")
    p.add_argument("--fhs-npz", default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--whi-npz", default="vae_cox_cache/bundles/WHI_raw_WHI_methylation_with_snp_merged_1milfeatures_combined_training_WHI_methylation_with_snp_merged_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-cpg-txt", default="FHS_methylation_with_snp_1milfeatures_cpg_columns.txt")
    p.add_argument("--whi-cpg-txt", default="WHI_methylation_with_snp_merged_1milfeatures_cpg_columns.txt")
    p.add_argument("--fhs-meta-pq", default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-meta-pq", default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", default="Share_ID")
    p.add_argument("--whi-id-col", default="sample_ID")
    p.add_argument("--risk-parquet", default="")
    p.add_argument("--risk-npz", default="")
    p.add_argument("--annot-csv", default="Annotation.csv")
    p.add_argument("--grouping", default="residual", choices=["residual", "within-cohort", "global"])
    p.add_argument("--residual-q-lo", type=float, default=0.10)
    p.add_argument("--residual-q-hi", type=float, default=0.90)
    p.add_argument("--min-std", type=float, default=0.01)
    p.add_argument("--missing-thresh", type=float, default=0.10)
    p.add_argument("--out-dir", default="feature_importance/miniewas")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = pd.read_csv(args.sig_csv)
    sig_cpg = sig.loc[sig["kind"] == "cpg", "feature"].astype(str).tolist()
    sig_age_set = set()
    if Path(args.sig_age_csv).exists():
        sig_age_set = set(pd.read_csv(args.sig_age_csv).query('kind=="cpg"')["feature"].astype(str))
    print(f"[mini-ewas] Selected CpGs: {len(sig_cpg):,}")

    fhs_names = read_txt_list(Path(args.fhs_cpg_txt))
    whi_names = read_txt_list(Path(args.whi_cpg_txt))
    fhs_map = {s: i for i, s in enumerate(fhs_names)}
    whi_map = {s: i for i, s in enumerate(whi_names)}
    cols_fhs, cols_whi, resolved = [], [], []
    for name in sig_cpg:
        i_f, i_w = fhs_map.get(name), whi_map.get(name)
        if i_f is None or i_w is None:
            continue
        cols_fhs.append(i_f)
        cols_whi.append(i_w)
        resolved.append(name)
    cols_fhs = np.asarray(cols_fhs, dtype=np.int64)
    cols_whi = np.asarray(cols_whi, dtype=np.int64)
    resolved = np.asarray(resolved, dtype=object)
    print(f"  resolved {resolved.size:,} CpGs in both cohorts")
    if resolved.size == 0:
        raise SystemExit("No CpGs resolved in both cohorts.")

    fhs_z = np.load(args.fhs_npz, mmap_mode="r")
    whi_z = np.load(args.whi_npz, mmap_mode="r")
    order_f, order_w = np.argsort(cols_fhs), np.argsort(cols_whi)
    G_fhs = fhs_z["X_meth"][:, cols_fhs[order_f]].astype(np.float32)[:, np.argsort(order_f)]
    G_whi = whi_z["X_meth"][:, cols_whi[order_w]].astype(np.float32)[:, np.argsort(order_w)]
    G_fhs = np.where(np.isfinite(G_fhs), G_fhs, np.nan)
    G_whi = np.where(np.isfinite(G_whi), G_whi, np.nan)

    age_fhs = _read_age(Path(args.fhs_meta_pq), args.fhs_id_col)
    age_whi = _read_age(Path(args.whi_meta_pq), args.whi_id_col)
    n_fhs, n_whi = G_fhs.shape[0], G_whi.shape[0]

    if args.risk_parquet:
        risk_all = load_pooled_logh_from_parquet(
            Path(args.fhs_meta_pq), args.fhs_id_col,
            Path(args.whi_meta_pq), args.whi_id_col,
            Path(args.risk_parquet), n_fhs, n_whi,
        )
    elif args.risk_npz:
        from gwas.gwas_common import load_fhs_risk_full
        risk_fhs, risk_whi = load_fhs_risk_full(Path(args.risk_npz), n_fhs, fhs_z["event"].astype(np.int32))
        risk_all = np.concatenate([risk_fhs, risk_whi])
    else:
        raise SystemExit("Provide --risk-parquet or --risk-npz")

    age_all = np.concatenate([age_fhs, age_whi])
    cohort_all = np.concatenate([np.zeros(n_fhs, np.int8), np.ones(n_whi, np.int8)])
    G_all = np.concatenate([G_fhs, G_whi], axis=0)

    # Outlier groups (residual mode default)
    grouping_info: Dict[str, float | str] = {"mode": args.grouping}
    if args.grouping == "residual":
        X_cov = np.column_stack([np.ones(len(age_all)), age_all.astype(np.float64), cohort_all.astype(np.float64)])
        coef, *_ = np.linalg.lstsq(X_cov, risk_all.astype(np.float64), rcond=None)
        risk_resid = risk_all.astype(np.float64) - X_cov @ coef
        lo, hi = np.quantile(risk_resid, args.residual_q_lo), np.quantile(risk_resid, args.residual_q_hi)
        resilient_mask = risk_resid <= lo
        accelerated_mask = risk_resid >= hi
        grouping_info.update(intercept=float(coef[0]), beta_age=float(coef[1]), beta_cohort=float(coef[2]), resid_lo=float(lo), resid_hi=float(hi))
    else:
        raise SystemExit("Only residual grouping implemented for mini-EWAS in this version.")

    keep_mask = resilient_mask | accelerated_mask
    y = accelerated_mask.astype(np.int32)
    n_res, n_acc = int(resilient_mask.sum()), int(accelerated_mask.sum())
    print(f"  resilient={n_res}  accelerated={n_acc}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(age_all[~keep_mask], risk_all[~keep_mask], s=10, c="#d0d0d0", alpha=0.45, label=f"other (n={int((~keep_mask).sum())})")
    ax.scatter(age_all[resilient_mask], risk_all[resilient_mask], s=22, c="#27ae60", label=f"resilient (n={n_res})")
    ax.scatter(age_all[accelerated_mask], risk_all[accelerated_mask], s=22, c="#c0392b", label=f"accelerated (n={n_acc})")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted log-hazard")
    ax.set_title("Mini-EWAS outlier groups (age-residualised risk)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "group_definition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "sample_idx_pooled": np.arange(len(age_all)),
        "cohort": np.where(cohort_all == 0, "FHS", "WHI"),
        "age": age_all, "predicted_risk": risk_all,
        "group": np.where(resilient_mask, "resilient", np.where(accelerated_mask, "accelerated", "other")),
    }).to_csv(out_dir / "group_definition.csv", index=False)

    G_use = G_all[keep_mask]
    cohort_use = cohort_all[keep_mask].astype(np.float32)
    y_use = y[keep_mask]
    n_use, K = G_use.shape
    intercept = np.ones(n_use, dtype=np.float32)

    results = {k: [] for k in ("cpg", "n_used", "mean_beta", "std_beta", "or_acc", "beta", "se", "z", "p", "beta_cohort")}
    t0 = time.time()
    for j in range(K):
        g = G_use[:, j]
        good = ~np.isnan(g)
        if good.sum() < max(20, int(0.6 * n_use)):
            continue
        gn = g[good]
        if (1 - good.mean()) > args.missing_thresh or np.nanstd(gn) < args.min_std:
            continue
        y_j = y_use[good].astype(np.float32)
        if y_j.sum() < 3 or (1 - y_j).sum() < 3:
            continue
        X = np.column_stack([np.ones(gn.size), gn.astype(np.float64), cohort_use[good].astype(np.float64)])
        beta_v, se_v, conv = logit_irls(X, y_j.astype(np.float64))
        if not conv or not np.isfinite(beta_v[1]) or se_v[1] <= 0:
            continue
        beta, se = float(beta_v[1]), float(se_v[1])
        zval = beta / se
        pval = float(2.0 * stats.norm.sf(abs(zval)))
        results["cpg"].append(resolved[j])
        results["n_used"].append(int(good.sum()))
        results["mean_beta"].append(float(np.mean(gn)))
        results["std_beta"].append(float(np.std(gn)))
        results["or_acc"].append(float(np.exp(beta)))
        results["beta"].append(beta)
        results["se"].append(se)
        results["z"].append(zval)
        results["p"].append(pval)
        results["beta_cohort"].append(float(beta_v[2]))

    df = pd.DataFrame(results)
    if df.empty:
        raise SystemExit("No CpGs passed QC.")
    m = len(df)
    order = df["p"].sort_values().index.to_numpy()
    p_sorted = df.loc[order, "p"].to_numpy()
    ranks = np.arange(1, m + 1, dtype=np.float64)
    bh = p_sorted * m / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1].clip(0, 1)
    q = np.empty(m)
    q[order] = bh
    df["q"] = q
    df["sig_age"] = df["cpg"].isin(sig_age_set)

    annot = load_cpg_annotation(Path(args.annot_csv))
    df = annotate_cpg_df(df, annot)
    df = df.sort_values(["p", "chrom", "pos"]).reset_index(drop=True)
    df.to_csv(out_dir / "miniewas_results.csv", index=False)

    bonf = 0.05 / m
    top_show = df.head(max(int((df["q"] < 0.05).sum()), 100)).copy()
    top_show.to_csv(out_dir / "miniewas_top_hits.csv", index=False)

    # Manhattan
    fig, ax = plt.subplots(figsize=(11, 5.2))
    dplot = df.sort_values(["chrom", "pos"])
    chroms = sorted(dplot["chrom"].unique())
    cum = 0
    palette = ["#8e44ad", "#9b59b6"]
    for i, ch in enumerate(chroms):
        sub = dplot[dplot["chrom"] == ch]
        x = cum + np.arange(len(sub))
        ax.scatter(x, -np.log10(sub["p"].clip(1e-300)), c=palette[i % 2], s=6, linewidths=0, alpha=0.85)
        cum += len(sub)
    ax.axhline(-np.log10(bonf), color="#c0392b", ls="--", lw=0.8, label=f"Bonferroni p={bonf:.1e}")
    ax.axhline(-np.log10(0.05), color="#7f8c8d", ls=":", lw=0.8, label="p=0.05")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(f"Mini-EWAS: resilient (n={n_res}) vs accelerated (n={n_acc}), {K:,} model CpGs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "miniewas_manhattan.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # QQ
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    p_obs = df["p"].sort_values().to_numpy()
    p_exp = (np.arange(1, len(p_obs) + 1) - 0.5) / len(p_obs)
    ax.scatter(-np.log10(p_exp), -np.log10(p_obs.clip(1e-300)), s=6, c="#2c3e50", linewidths=0)
    chi2 = stats.chi2.isf(p_obs.clip(1e-300), df=1)
    lam = float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))
    ax.set_title(f"Mini-EWAS QQ  lambda_GC={lam:.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "miniewas_qq.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_input_cpgs": len(sig_cpg),
        "n_resolved_both_cohorts": int(resolved.size),
        "n_after_qc": m,
        "n_resilient": n_res,
        "n_accelerated": n_acc,
        "grouping": grouping_info,
        "bonferroni_p": bonf,
        "n_pass_bonferroni": int((df["p"] < bonf).sum()),
        "n_pass_bh_fdr_5pc": int((df["q"] < 0.05).sum()),
        "n_suggestive_p_lt_0_05": int((df["p"] < 0.05).sum()),
        "lambda_gc": lam,
        "runtime_sec": time.time() - t0,
        "top10": df.head(10).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Mini-EWAS outputs in {out_dir}/  tested={m}  suggestive={summary['n_suggestive_p_lt_0_05']}")


if __name__ == "__main__":
    main()
