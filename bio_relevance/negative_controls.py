#!/usr/bin/env python3
"""Negative controls: batch variance, noise correlations, random gene-set enrichment null.

Outputs ``feature_importance/bio_relevance/negative_controls/``:

  batch_logh_eta_fhs.json     one-way ANOVA eta^2 (FHS): batch -> log_h, controlling age
  noise_correlation_null.json Pearson |r|(log_h, noise) vs |r|(log_h, age_pred)
  gprofiler_null_summary.json optional: mean # significant terms from random gene queries
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bio_relevance.model_scores import pooled_predictions
from bio_relevance.pathway_enrichment import (
    collect_foreground,
    load_epic_background,
    run_gprofiler,
    results_table_all,
)


def eta_squared_oneway(groups: list) -> float:
    """Eta^2 for one-way layout (list of 1-D arrays)."""
    all_y = np.concatenate(groups)
    grand_mean = all_y.mean()
    ss_tot = ((all_y - grand_mean) ** 2).sum()
    ss_between = 0.0
    for g in groups:
        g = np.asarray(g, dtype=np.float64)
        ss_between += len(g) * (g.mean() - grand_mean) ** 2
    if ss_tot <= 0:
        return 0.0
    return float(ss_between / ss_tot)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/negative_controls")
    p.add_argument("--bundle-dir", type=str, default="models/aesurv_final")
    p.add_argument("--proj-fhs-npz", type=str, default="vae_cox_cache/jl_proj/proj_FHS_b24e34c31a.npz")
    p.add_argument("--proj-whi-npz", type=str, default="vae_cox_cache/jl_proj/proj_WHI_b159d5bf88.npz")
    p.add_argument("--fhs-meta-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--annot-csv", type=str, default="Annotation.csv")
    p.add_argument("--n-gprofiler-null", type=int, default=12,
                   help="Random gene draws vs g:Profiler (0 skips network).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    import torch
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    log_h, age_p, n_fhs, n_whi = pooled_predictions(
        Path(args.bundle_dir), Path(args.proj_fhs_npz), Path(args.proj_whi_npz), device,
    )

    # --- Noise null: correlation with pure noise should be ~0 ---
    noise = rng.standard_normal(len(log_h))
    r_noise, _ = stats.pearsonr(log_h, noise)
    r_agep, _ = stats.pearsonr(log_h, age_p)
    (out / "noise_correlation_null.json").write_text(
        json.dumps({
            "r_logh_vs_gaussian_noise": float(r_noise),
            "r_logh_vs_age_pred": float(r_agep),
            "interpretation": "If biology drives risk, |r|(log_h, age_pred) should exceed spurious |r| vs noise.",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {out / 'noise_correlation_null.json'}")

    # --- FHS batch effect on log_h (age-adjusted residuals) ---
    df = pd.read_parquet(args.fhs_meta_pq, columns=["age", "batch"])
    n = min(len(df), n_fhs, len(log_h))
    age = df["age"].to_numpy(dtype=np.float64)[:n]
    batch = df["batch"].astype(str).to_numpy()[:n]
    y = log_h[:n].astype(np.float64)
    X = np.column_stack([np.ones(n), age])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_res = y - X @ coef
    batches = sorted(set(batch.tolist()))
    groups = [y_res[batch == b] for b in batches if np.sum(batch == b) > 5]
    eta = eta_squared_oneway(groups) if len(groups) >= 2 else float("nan")
    # omnibus p from one-way ANOVA
    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
    else:
        f_stat, p_anova = float("nan"), float("nan")
    (out / "batch_logh_eta_fhs.json").write_text(
        json.dumps({
            "n_fhs_used": n,
            "n_batches": len(batches),
            "eta_sq_batch_on_age_adjusted_logh": eta,
            "f_oneway": float(f_stat) if f_stat == f_stat else None,
            "p_oneway": float(p_anova) if p_anova == p_anova else None,
            "note": "log_h residualised on linear age within FHS; eta^2 is fraction of residual variance between batches.",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {out / 'batch_logh_eta_fhs.json'}")

    # --- g:Profiler null: random gene sets of same size as foreground ---
    if args.n_gprofiler_null <= 0:
        return
    fg = collect_foreground(
        Path("feature_importance/shap/annot/risk_top_cpg_annotated.csv"),
        Path("feature_importance/shap/annot/risk_top_snp_annotated.csv"),
        Path("feature_importance/minigwas/minigwas_results.csv"),
        top_n=300,
    )
    bg = load_epic_background(Path(args.annot_csv), max_n=20000, seed=args.seed)
    bg_arr = np.array(bg)
    fg_list = sorted(set(fg) & set(bg))
    k = max(10, min(100, len(fg_list)))
    null_counts = []
    for i in range(args.n_gprofiler_null):
        q = rng.choice(bg_arr, size=k, replace=False).tolist()
        resp = run_gprofiler(q, bg)
        if "error" in resp or "result" not in resp:
            null_counts.append(-1)
            continue
        df_t = results_table_all(resp)
        if df_t.empty:
            null_counts.append(0)
            continue
        col = "adjusted_p_value" if "adjusted_p_value" in df_t.columns else "p_value"
        null_counts.append(int((df_t[col].astype(float) < 0.05).sum()))
    real = run_gprofiler(fg_list[:k], bg)
    real_df = results_table_all(real)
    col = "adjusted_p_value" if not real_df.empty and "adjusted_p_value" in real_df.columns else "p_value"
    n_sig_real = int((real_df[col].astype(float) < 0.05).sum()) if not real_df.empty and col in real_df.columns else 0
    valid = [c for c in null_counts if c >= 0]
    summary = {
        "foreground_genes_used": k,
        "n_sig_terms_real_query": n_sig_real,
        "n_sig_terms_random_mean": float(np.mean(valid)) if valid else None,
        "n_sig_terms_random_std": float(np.std(valid)) if valid else None,
        "empirical_p_vs_null": float(np.mean(np.array(valid) >= n_sig_real)) if valid else None,
    }
    (out / "gprofiler_null_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out / 'gprofiler_null_summary.json'}")


if __name__ == "__main__":
    main()
