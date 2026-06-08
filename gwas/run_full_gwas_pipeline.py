#!/usr/bin/env python3
"""FHS discovery -> WHI replication -> outlier-stratification GWAS pipeline.

Primary phenotype: age-adjusted AESurv log_h residual (log_h ~ age + sex within cohort).

Stages
------
1. discovery   — linear GWAS on all FHS SNPs (~1M)
2. replication — test clumped FHS lead SNPs in WHI; fixed-effect meta
3. stratification — allele dosage across resilient / middle / accelerated groups
4. enrichment  — overlap of discovery hits with AESurv gradient-significant SNPs

Example
-------
cd D:\\SNP_datasets
$env:PYTHONPATH="D:\\SNP_datasets"
python gwas/run_full_gwas_pipeline.py --stage all

Smoke test (first 20k SNPs only):
python gwas/run_full_gwas_pipeline.py --stage all --snp-limit 20000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from gwas.gwas_common import (
    batch_linear_gwas,
    bh_fdr,
    clump_leads,
    compute_logh_residual,
    define_outlier_groups,
    enrichment_vs_panel,
    fixed_effect_meta,
    genomic_lambda,
    load_fhs_risk_full,
    load_genotype_columns,
    read_age_sex,
    read_txt_list,
    resolve_snp_columns,
    write_json,
    _drop_constant_covariates,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def _plot_manhattan(df: pd.DataFrame, out_path: Path, title: str, p_line: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    d = df.sort_values(["chrom", "pos"]).reset_index(drop=True)
    chroms = sorted(d["chrom"].unique())
    cum = 0
    tick_pos, tick_lab = [], []
    palette = ["#34495e", "#7f8c8d"]
    for i, ch in enumerate(chroms):
        sub = d[d["chrom"] == ch]
        x = cum + np.arange(len(sub))
        ax.scatter(
            x,
            -np.log10(sub["p"].clip(lower=1e-300)),
            c=palette[i % 2],
            s=4,
            linewidths=0,
            alpha=0.85,
        )
        tick_pos.append(cum + len(sub) / 2)
        if ch <= 22:
            lab = str(int(ch))
        elif ch == 23:
            lab = "X"
        elif ch == 24:
            lab = "Y"
        else:
            lab = "?"
        tick_lab.append(lab)
        cum += len(sub)
    ax.axhline(-np.log10(p_line), color="#c0392b", lw=0.8, ls="--", label=f"p={p_line:g}")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_qq(pvals: np.ndarray, out_path: Path, lam: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    p_obs = np.sort(pvals)
    p_exp = (np.arange(1, len(p_obs) + 1) - 0.5) / len(p_obs)
    ax.scatter(-np.log10(p_exp), -np.log10(p_obs.clip(min=1e-300)), s=6, c="#2c3e50", linewidths=0)
    mx = max(-np.log10(p_exp.min()), -np.log10(p_obs.min().clip(min=1e-300)))
    ax.plot([0, mx], [0, mx], color="#c0392b", lw=1)
    ax.set_title(f"{title}   (lambda_GC = {lam:.3f})", fontsize=11)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_group_scatter(
    age: np.ndarray,
    log_h: np.ndarray,
    groups: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"other": "#d0d0d0", "middle": "#95a5a6", "resilient": "#27ae60", "accelerated": "#c0392b"}
    for g in ("other", "middle", "resilient", "accelerated"):
        m = groups["group"] == g
        if not m.any():
            continue
        ax.scatter(
            age[m], log_h[m], s=14 if g != "other" else 8, c=colors[g],
            alpha=0.55 if g == "other" else 0.85, linewidths=0.3 if g != "other" else 0,
            label=f"{g} (n={int(m.sum())})",
        )
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted log-hazard (AESurv)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_stratification(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(df)), 5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["mean_dose_resilient"], width=w, label="resilient", color="#27ae60")
    ax.bar(x, df["mean_dose_middle"], width=w, label="middle", color="#95a5a6")
    ax.bar(x + w, df["mean_dose_accelerated"], width=w, label="accelerated", color="#c0392b")
    ax.set_xticks(x)
    labels = df["snp"].astype(str).tolist()
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean allele dosage")
    ax.set_title("Replicated SNPs: dosage by outlier group (FHS primary)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_discovery(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    out_csv = out_dir / "discovery_fhs_results.csv"
    if out_csv.exists() and not args.force:
        print(f"[discovery] Loading cached {out_csv}")
        return pd.read_csv(out_csv)

    print("[discovery] Loading FHS phenotypes ...")
    fhs_z = np.load(args.fhs_npz, mmap_mode="r")
    n_fhs = int(fhs_z["X_snp"].shape[0])
    age, sex = read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    event = fhs_z["event"].astype(np.int32)
    risk_fhs, _ = load_fhs_risk_full(
        Path(args.risk_npz), n_fhs, event, seed=args.seed, val_frac=args.val_frac,
    )
    y, pheno_info = compute_logh_residual(risk_fhs, age, sex=sex)
    cov = np.column_stack([np.ones(n_fhs), age.astype(np.float64), sex.astype(np.float64)])

    snp_names = read_txt_list(Path(args.fhs_snp_txt))
    n_snps = len(snp_names)
    if args.snp_limit > 0:
        n_snps = min(n_snps, args.snp_limit)
        snp_names = snp_names[:n_snps]
    print(f"[discovery] FHS n={n_fhs}  SNPs={n_snps:,}  phenotype=age+sex adjusted log_h residual")

    X_snp = fhs_z["X_snp"]
    chunks: List[pd.DataFrame] = []
    t0 = time.time()
    for start in range(0, n_snps, args.batch_size):
        end = min(n_snps, start + args.batch_size)
        G = X_snp[:, start:end].astype(np.float32)
        batch_df = batch_linear_gwas(
            y, G, cov, snp_names[start:end],
            maf_min=args.maf_min, miss_max=args.miss_max,
        )
        chunks.append(batch_df)
        if (start // args.batch_size) % 20 == 0 or end == n_snps:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1e-6)
            print(f"  [{end:,}/{n_snps:,}] SNPs  {rate:.0f} SNP/s  valid={batch_df.shape[0]}")

    df = pd.concat(chunks, ignore_index=True)
    df["q"] = bh_fdr(df["p"].to_numpy())
    df = df.sort_values(["p", "chrom", "pos"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[discovery] Wrote {out_csv}  n={len(df):,}  in {time.time()-t0:.1f}s")

    lam = genomic_lambda(df["p"].to_numpy())
    _plot_manhattan(
        df, out_dir / "discovery_fhs_manhattan.png",
        f"FHS discovery GWAS: age-adjusted log_h residual (n={n_fhs})",
        args.discovery_p,
    )
    _plot_qq(df["p"].to_numpy(), out_dir / "discovery_fhs_qq.png", lam, "FHS discovery QQ")
    write_json(out_dir / "discovery_summary.json", {
        "n_samples": n_fhs,
        "n_snps_tested": int(len(df)),
        "phenotype": "log_h_residual_age_sex",
        "pheno_regression": pheno_info,
        "lambda_gc": lam,
        "discovery_p_threshold": args.discovery_p,
        "n_suggestive": int((df["p"] <= args.discovery_p).sum()),
        "n_fdr_5pc": int((df["q"] <= 0.05).sum()),
        "top10": df.head(10)[["snp", "chrom", "pos", "beta", "p", "q"]].to_dict(orient="records"),
    })
    return df


def run_replication(args: argparse.Namespace, out_dir: Path, discovery: pd.DataFrame) -> pd.DataFrame:
    out_csv = out_dir / "replication_results.csv"
    if out_csv.exists() and not args.force:
        print(f"[replication] Loading cached {out_csv}")
        return pd.read_csv(out_csv)

    leads = clump_leads(
        discovery, p_max=args.discovery_p, max_leads=args.max_leads, window_bp=args.clump_bp,
    )
    leads.to_csv(out_dir / "discovery_lead_snps.csv", index=False)
    print(f"[replication] Testing {len(leads):,} lead SNPs in WHI ...")

    whi_z = np.load(args.whi_npz, mmap_mode="r")
    n_whi = int(whi_z["X_snp"].shape[0])
    age_w, sex_w = read_age_sex(Path(args.whi_meta_pq), args.whi_id_col)
    rd = dict(np.load(args.risk_npz, allow_pickle=False))
    risk_whi = rd["risk_whi_test"].astype(np.float32)
    y_w, _ = compute_logh_residual(risk_whi, age_w, sex=sex_w)
    cov_w = np.column_stack([np.ones(n_whi), age_w.astype(np.float64), sex_w.astype(np.float64)])
    cov_w = _drop_constant_covariates(cov_w)

    fhs_map = {s: i for i, s in enumerate(read_txt_list(Path(args.fhs_snp_txt)))}
    whi_map = {s: i for i, s in enumerate(read_txt_list(Path(args.whi_snp_txt)))}

    rows = []
    for _, lead in leads.iterrows():
        snp = str(lead["snp"])
        i_f = fhs_map.get(snp)
        i_w = whi_map.get(snp)
        if i_f is None or i_w is None:
            continue
        G_w = whi_z["X_snp"][:, i_w : i_w + 1].astype(np.float32)
        whi_res = batch_linear_gwas(
            y_w, G_w, cov_w, [snp], maf_min=args.maf_min, miss_max=args.miss_max,
        )
        if whi_res.empty:
            continue
        wr = whi_res.iloc[0]
        meta_b, meta_se, meta_p = fixed_effect_meta(
            float(lead["beta"]), float(lead["se"]), float(wr["beta"]), float(wr["se"]),
        )
        same_sign = np.sign(lead["beta"]) == np.sign(wr["beta"])
        repl_ok = same_sign and float(wr["p"]) <= args.replication_p
        rows.append({
            "snp": snp,
            "chrom": int(lead["chrom"]),
            "pos": int(lead["pos"]),
            "beta_fhs": float(lead["beta"]),
            "se_fhs": float(lead["se"]),
            "p_fhs": float(lead["p"]),
            "q_fhs": float(lead.get("q", np.nan)),
            "beta_whi": float(wr["beta"]),
            "se_whi": float(wr["se"]),
            "p_whi": float(wr["p"]),
            "same_sign": bool(same_sign),
            "replicated": bool(repl_ok),
            "beta_meta": meta_b,
            "se_meta": meta_se,
            "p_meta": meta_p,
        })

    rep = pd.DataFrame(rows).sort_values("p_meta").reset_index(drop=True)
    rep.to_csv(out_csv, index=False)
    repl = rep[rep["replicated"]].copy()
    repl.to_csv(out_dir / "replicated_hits.csv", index=False)
    write_json(out_dir / "replication_summary.json", {
        "n_leads_tested": int(len(rep)),
        "n_replicated": int(len(repl)),
        "discovery_p": args.discovery_p,
        "replication_p": args.replication_p,
        "replicated_snps": repl["snp"].tolist(),
        "top_replicated": repl.head(10).to_dict(orient="records"),
    })
    print(f"[replication] Wrote {out_csv}  replicated={len(repl)}")
    return rep


def run_stratification(args: argparse.Namespace, out_dir: Path, replication: pd.DataFrame) -> pd.DataFrame:
    out_csv = out_dir / "stratification_dosage.csv"
    if out_csv.exists() and not args.force:
        print(f"[stratification] Loading cached {out_csv}")
        return pd.read_csv(out_csv)

    repl = replication[replication["replicated"]].copy()
    if repl.empty:
        repl = replication.nsmallest(min(10, len(replication)), "p_meta").copy()
        print("[stratification] No replicated hits; using top meta-p leads for exploratory plot.")
    if repl.empty:
        print("[stratification] Nothing to stratify.")
        return pd.DataFrame()

    print(f"[stratification] Dosage by group for {len(repl)} SNPs (FHS primary) ...")
    fhs_z = np.load(args.fhs_npz, mmap_mode="r")
    whi_z = np.load(args.whi_npz, mmap_mode="r")
    n_fhs, n_whi = fhs_z["X_snp"].shape[0], whi_z["X_snp"].shape[0]
    age_f, _ = read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    age_w, _ = read_age_sex(Path(args.whi_meta_pq), args.whi_id_col)
    event_f = fhs_z["event"].astype(np.int32)
    risk_f, risk_w = load_fhs_risk_full(
        Path(args.risk_npz), n_fhs, event_f, seed=args.seed, val_frac=args.val_frac,
    )
    log_h = np.concatenate([risk_f, risk_w])
    age = np.concatenate([age_f, age_w])
    cohort = np.concatenate([np.zeros(n_fhs, dtype=np.int8), np.ones(n_whi, dtype=np.int8)])
    og = define_outlier_groups(
        log_h, age, cohort,
        residual_q_lo=args.residual_q_lo, residual_q_hi=args.residual_q_hi,
    )
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
    groups.to_csv(out_dir / "outlier_groups.csv", index=False)
    _plot_group_scatter(
        age, log_h, groups, out_dir / "outlier_groups.png",
        "Outlier groups for stratification (pooled FHS+WHI, age-residualised log_h)",
    )

    fhs_groups = groups.loc[groups["cohort"] == "FHS"].reset_index(drop=True)
    grp_fhs = fhs_groups["group"].to_numpy()
    cols_f, resolved = resolve_snp_columns(repl["snp"].astype(str).tolist(), Path(args.fhs_snp_txt))
    G = load_genotype_columns(Path(args.fhs_npz), cols_f)
    G = np.where((G >= 0) & (G <= 2), G, np.nan)

    rows = []
    for j, snp in enumerate(resolved):
        g = G[:, j]
        good = np.isfinite(g)
        gf = g[good]
        gf_grp = grp_fhs[good]
        means = {}
        for label in ("resilient", "middle", "accelerated"):
            m = gf_grp == label
            means[label] = float(np.nanmean(gf[m])) if m.any() else np.nan
        # Cochran-Armitage trend: group ordinal 0,1,2
        ord_map = {"resilient": 0, "middle": 1, "accelerated": 2}
        y_ord = np.array([ord_map.get(x, np.nan) for x in gf_grp], dtype=np.float64)
        mask = np.isfinite(y_ord)
        if mask.sum() >= 20 and np.unique(y_ord[mask]).size >= 2:
            r, p_trend = stats.pearsonr(gf[mask], y_ord[mask])
        else:
            r, p_trend = np.nan, np.nan
        beta_fhs = float(repl.loc[repl["snp"] == snp, "beta_fhs"].iloc[0])
        risk_allele_higher_in_accel = (
            means["accelerated"] > means["middle"] > means["resilient"]
            if beta_fhs > 0
            else means["accelerated"] < means["middle"] < means["resilient"]
        )
        rows.append({
            "snp": snp,
            "beta_fhs": beta_fhs,
            "mean_dose_resilient": means["resilient"],
            "mean_dose_middle": means["middle"],
            "mean_dose_accelerated": means["accelerated"],
            "trend_r": float(r) if np.isfinite(r) else np.nan,
            "trend_p": float(p_trend) if np.isfinite(p_trend) else np.nan,
            "expected_direction_in_tails": bool(risk_allele_higher_in_accel),
        })

    strat = pd.DataFrame(rows)
    strat.to_csv(out_csv, index=False)
    _plot_stratification(strat, out_dir / "stratification_dosage.png")
    write_json(out_dir / "stratification_summary.json", {
        "group_definition": og.info,
        "n_fhs_resilient": int((groups["group"] == "resilient").sum()),
        "n_fhs_accelerated": int((groups["group"] == "accelerated").sum()),
        "n_snps": int(len(strat)),
        "n_expected_direction": int(strat["expected_direction_in_tails"].sum()) if not strat.empty else 0,
        "rows": strat.to_dict(orient="records"),
    })
    print(f"[stratification] Wrote {out_csv}")
    return strat


def run_enrichment(args: argparse.Namespace, out_dir: Path, discovery: pd.DataFrame) -> dict:
    out_json = out_dir / "enrichment_summary.json"
    if out_json.exists() and not args.force:
        print(f"[enrichment] Loading cached {out_json}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    sig = pd.read_csv(args.sig_csv)
    panel = set(sig.loc[sig["kind"] == "snp", "feature"].astype(str))
    hits = discovery.loc[discovery["p"] <= args.discovery_p, "snp"].astype(str).tolist()
    universe = len(read_txt_list(Path(args.fhs_snp_txt)))
    if args.snp_limit > 0:
        universe = min(universe, args.snp_limit)
    result = enrichment_vs_panel(hits, panel, universe)
    result["discovery_p_threshold"] = args.discovery_p
    result["panel_csv"] = args.sig_csv
    write_json(out_json, result)
    print(
        f"[enrichment] overlap={result['n_overlap']}  "
        f"fold={result['fold_enrichment']:.2f}  p={result['p_hypergeom']:.3g}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="FHS discovery -> WHI replication GWAS pipeline")
    p.add_argument("--stage", type=str, default="all",
                   choices=["all", "discovery", "replication", "stratification", "enrichment"])
    p.add_argument("--out-dir", type=str, default="feature_importance/gwas")
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
    p.add_argument("--sig-csv", type=str,
                   default="feature_importance/significance/significant_risk.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--snp-limit", type=int, default=0,
                   help="If >0, only first N SNPs (smoke test).")
    p.add_argument("--maf-min", type=float, default=0.01)
    p.add_argument("--miss-max", type=float, default=0.05)
    p.add_argument("--discovery-p", type=float, default=1e-5)
    p.add_argument("--replication-p", type=float, default=0.05)
    p.add_argument("--max-leads", type=int, default=500)
    p.add_argument("--clump-bp", type=int, default=1_000_000)
    p.add_argument("--residual-q-lo", type=float, default=0.10)
    p.add_argument("--residual-q-hi", type=float, default=0.90)
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = ["discovery", "replication", "stratification", "enrichment"]
    if args.stage != "all":
        stages = [args.stage]

    discovery = replication = None
    if "discovery" in stages or any(s in stages for s in ("replication", "enrichment")):
        if "discovery" in stages:
            discovery = run_discovery(args, out_dir)
        elif (out_dir / "discovery_fhs_results.csv").exists():
            discovery = pd.read_csv(out_dir / "discovery_fhs_results.csv")
        else:
            raise SystemExit("discovery_fhs_results.csv missing; run --stage discovery first.")

    if "replication" in stages or "stratification" in stages:
        if discovery is None:
            discovery = pd.read_csv(out_dir / "discovery_fhs_results.csv")
        replication = run_replication(args, out_dir, discovery)

    if "stratification" in stages:
        if replication is None:
            replication = pd.read_csv(out_dir / "replication_results.csv")
        run_stratification(args, out_dir, replication)

    if "enrichment" in stages:
        if discovery is None:
            discovery = pd.read_csv(out_dir / "discovery_fhs_results.csv")
        run_enrichment(args, out_dir, discovery)

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
