#!/usr/bin/env python3
"""Annotation + outlier stratification for replicated mortality SNPs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from annotate_top_features import fetch_gwas_nearby, fetch_snp_genes
from gwas.gwas_common import (
    define_outlier_groups,
    load_fhs_risk_full,
    load_genotype_columns,
    read_age_sex,
    resolve_snp_columns,
    write_json,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

DEFAULT_SNPS = ["12:46600556_T", "11:127103604_C"]


def _plot_dosage(strat: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(strat))
    w = 0.25
    ax.bar(x - w, strat["mean_dose_resilient"], width=w, label="resilient", color="#27ae60")
    ax.bar(x, strat["mean_dose_middle"], width=w, label="middle", color="#95a5a6")
    ax.bar(x + w, strat["mean_dose_accelerated"], width=w, label="accelerated", color="#c0392b")
    labels = [
        f"{r['snp']}\n(r={r['trend_r']:.2f}, p={r['trend_p']:.2e})"
        for _, r in strat.iterrows()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean allele dosage (FHS)")
    ax.set_title("Replicated mortality SNPs: dosage by outlier group", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snp-list", nargs="*", default=DEFAULT_SNPS)
    p.add_argument("--out-dir", default="feature_importance/gwas/replicated")
    p.add_argument("--sig-csv", default="feature_importance/significance/significant_risk.csv")
    p.add_argument("--annot-cache", default="feature_importance/annot/snp_gene_cache.csv")
    p.add_argument("--gwas-cache", default="feature_importance/annot/snp_gwas_cache.csv")
    p.add_argument("--replication-csv", default="feature_importance/gwas/snp_cox_replication.csv")
    p.add_argument("--fhs-npz",
                   default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--whi-npz",
                   default="vae_cox_cache/bundles/WHI_raw_WHI_methylation_with_snp_merged_1milfeatures_combined_training_WHI_methylation_with_snp_merged_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-snp-txt", default="FHS_methylation_with_snp_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--fhs-meta-pq", default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-meta-pq", default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", default="Share_ID")
    p.add_argument("--whi-id-col", default="sample_ID")
    p.add_argument("--risk-npz", default="runs/aesurv_aux_grid/age12.00_cell1.00/aesurv_aux_risk.npz")
    p.add_argument("--residual-q-lo", type=float, default=0.10)
    p.add_argument("--residual-q-hi", type=float, default=0.90)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snps = list(args.snp_list)
    print(f"[followup] SNPs: {snps}")

    # ---- Annotation
    cache: dict = {}
    if Path(args.annot_cache).exists():
        df_c = pd.read_csv(args.annot_cache)
        cache = dict(zip(df_c["feature_id"].astype(str), df_c["gene_ensembl"].astype(str)))
    gene_map = fetch_snp_genes(snps, cache, Path(args.annot_cache), sleep_sec=0.05)
    gwas_map = fetch_gwas_nearby(snps, Path(args.gwas_cache), radius_bp=5000, sleep_sec=0.05)

    sig_set = set()
    if Path(args.sig_csv).exists():
        sig = pd.read_csv(args.sig_csv)
        sig_set = set(sig.loc[sig["kind"] == "snp", "feature"].astype(str))

    rep = pd.read_csv(args.replication_csv) if Path(args.replication_csv).exists() else pd.DataFrame()
    annot_rows = []
    for snp in snps:
        r = rep[rep["snp"] == snp].iloc[0] if not rep.empty and (rep["snp"] == snp).any() else None
        annot_rows.append({
            "snp": snp,
            "gene_ensembl": gene_map.get(snp, ""),
            "gwas_rsids_near": gwas_map.get(snp, {}).get("rsids", ""),
            "gwas_n_hits_near": gwas_map.get(snp, {}).get("n_hits", 0),
            "in_aesurv_gradient_11k": snp in sig_set,
            "fhs_hr": float(r["fhs_hr_per_allele"]) if r is not None else np.nan,
            "fhs_p": float(r["fhs_p"]) if r is not None else np.nan,
            "whi_hr": float(r["whi_hr_per_allele"]) if r is not None else np.nan,
            "whi_p": float(r["whi_p"]) if r is not None else np.nan,
            "meta_hr": float(r["meta_hr"]) if r is not None else np.nan,
            "meta_p": float(r["meta_p"]) if r is not None else np.nan,
            "replicated": bool(r["replicated"]) if r is not None else False,
        })
    annot_df = pd.DataFrame(annot_rows)
    annot_df.to_csv(out_dir / "replicated_snp_annotation.csv", index=False)
    print(f"  wrote {out_dir / 'replicated_snp_annotation.csv'}")

    # ---- Outlier groups (pooled definition, FHS dosage)
    fhs_z = np.load(args.fhs_npz, mmap_mode="r")
    whi_z = np.load(args.whi_npz, mmap_mode="r")
    n_fhs, n_whi = fhs_z["X_snp"].shape[0], whi_z["X_snp"].shape[0]
    age_f, _ = read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    age_w, _ = read_age_sex(Path(args.whi_meta_pq), args.whi_id_col)
    event_f = fhs_z["event"].astype(np.int32)
    risk_f, risk_w = load_fhs_risk_full(Path(args.risk_npz), n_fhs, event_f)
    log_h = np.concatenate([risk_f, risk_w])
    age = np.concatenate([age_f, age_w])
    cohort = np.concatenate([np.zeros(n_fhs, dtype=np.int8), np.ones(n_whi, dtype=np.int8)])
    og = define_outlier_groups(log_h, age, cohort, residual_q_lo=args.residual_q_lo, residual_q_hi=args.residual_q_hi)

    groups = pd.DataFrame({
        "cohort": np.where(cohort == 0, "FHS", "WHI"),
        "age": age,
        "predicted_risk": log_h,
        "residual": og.residual,
        "group": np.where(
            og.resilient, "resilient",
            np.where(og.accelerated, "accelerated", np.where(og.middle, "middle", "other")),
        ),
    })
    groups.to_csv(out_dir / "outlier_groups_pooled.csv", index=False)

    fhs_groups = groups.loc[groups["cohort"] == "FHS", "group"].to_numpy()
    cols, resolved = resolve_snp_columns(snps, Path(args.fhs_snp_txt))
    G = load_genotype_columns(Path(args.fhs_npz), cols)
    G = np.where((G >= 0) & (G <= 2), G, np.nan)

    strat_rows = []
    for j, snp in enumerate(resolved):
        g = G[:, j]
        good = np.isfinite(g)
        gf, gf_grp = g[good], fhs_groups[good]
        means = {lbl: float(np.nanmean(gf[gf_grp == lbl])) if (gf_grp == lbl).any() else np.nan
                 for lbl in ("resilient", "middle", "accelerated")}
        ord_map = {"resilient": 0, "middle": 1, "accelerated": 2}
        y_ord = np.array([ord_map.get(x, np.nan) for x in gf_grp], dtype=np.float64)
        mask = np.isfinite(y_ord)
        if mask.sum() >= 20:
            trend_r, trend_p = stats.pearsonr(gf[mask], y_ord[mask])
        else:
            trend_r, trend_p = np.nan, np.nan
        beta_fhs = float(annot_df.loc[annot_df["snp"] == snp, "fhs_hr"].iloc[0])
        expected = means["accelerated"] > means["middle"] > means["resilient"] if beta_fhs > 1 else (
            means["accelerated"] < means["middle"] < means["resilient"]
        )
        strat_rows.append({
            "snp": snp,
            "mean_dose_resilient": means["resilient"],
            "mean_dose_middle": means["middle"],
            "mean_dose_accelerated": means["accelerated"],
            "trend_r": float(trend_r),
            "trend_p": float(trend_p),
            "expected_direction_in_tails": bool(expected),
        })

    strat = pd.DataFrame(strat_rows)
    strat.to_csv(out_dir / "replicated_snp_stratification.csv", index=False)
    _plot_dosage(strat, out_dir / "replicated_snp_stratification.png")
    print(f"  wrote {out_dir / 'replicated_snp_stratification.csv'}")

    write_json(out_dir / "replicated_snp_followup_summary.json", {
        "snps": snps,
        "group_definition": og.info,
        "annotation": annot_rows,
        "stratification": strat_rows,
    })
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
