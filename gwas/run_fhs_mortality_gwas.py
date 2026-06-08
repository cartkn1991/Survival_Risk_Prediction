#!/usr/bin/env python3
"""Full-genome mortality GWAS in FHS using batched linear models.

Outcome: death event indicator, adjusted for age and sex.

Model (per SNP, approximating Cox):

    event ~ SNP_additive + age_z + sex_female

SNP effects are estimated via OLS with covariate adjustment, vectorised across
all SNPs using the shared `batch_linear_gwas` helper.

Outputs (under --out-dir, default feature_importance/gwas/):
  - mortality_fhs_results.csv      per-SNP beta/se/p + MAF
  - mortality_fhs_manhattan.png    Manhattan plot
  - mortality_fhs_qq.png           QQ plot with lambda_GC
  - mortality_fhs_summary.json     summary stats and top hits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gwas.gwas_common import (
    batch_linear_gwas,
    genomic_lambda,
    open_pq,
    read_age_sex,
    read_txt_list,
    write_json,
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


def main() -> None:
    p = argparse.ArgumentParser(description="FHS mortality GWAS using linear models (event ~ SNP + age + sex).")
    p.add_argument(
        "--fhs-npz",
        type=str,
        default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz",
    )
    p.add_argument("--fhs-snp-txt", type=str, default="FHS_methylation_with_snp_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--fhs-meta-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--out-dir", type=str, default="feature_importance/gwas")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--maf-min", type=float, default=0.01)
    p.add_argument("--miss-max", type=float, default=0.05)
    p.add_argument("--discovery-p", type=float, default=5e-8)
    p.add_argument("--snp-limit", type=int, default=0, help="If >0, only first N SNPs (smoke test).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[mortality] Loading FHS bundle + metadata ...")
    z = np.load(args.fhs_npz, mmap_mode="r")
    n_fhs = int(z["X_snp"].shape[0])
    event = z["event"].astype(np.float32)

    age, sex = read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    if age.shape[0] != n_fhs:
        raise SystemExit(f"Age length {age.shape[0]} != bundle rows {n_fhs}")

    age_z = (age - age.mean()) / (age.std() + 1e-6)
    # sex already in {0,1} from read_age_sex
    cov = np.column_stack([np.ones(n_fhs, dtype=np.float64), age_z.astype(np.float64), sex.astype(np.float64)])

    snp_names = read_txt_list(Path(args.fhs_snp_txt))
    n_snps = len(snp_names)
    if args.snp_limit > 0:
        n_snps = min(n_snps, args.snp_limit)
        snp_names = snp_names[:n_snps]
    print(f"[mortality] FHS n={n_fhs}  events={int(event.sum())}  SNPs={n_snps:,}")

    X_snp = z["X_snp"]
    chunks = []
    start = 0
    while start < n_snps:
        end = min(n_snps, start + args.batch_size)
        G = X_snp[:, start:end].astype(np.float32)
        batch_df = batch_linear_gwas(
            event,
            G,
            cov,
            snp_names[start:end],
            maf_min=args.maf_min,
            miss_max=args.miss_max,
        )
        chunks.append(batch_df)
        if end == n_snps or (start // args.batch_size) % 20 == 0:
            print(f"  [{end:,}/{n_snps:,}] SNPs  valid={batch_df.shape[0]}")
        start = end

    df = pd.concat(chunks, ignore_index=True)
    out_csv = out_dir / "mortality_fhs_results.csv"
    df = df.sort_values(["p", "chrom", "pos"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[mortality] Wrote {out_csv}  n={len(df):,}")

    lam = genomic_lambda(df["p"].to_numpy())
    _plot_manhattan(
        df,
        out_dir / "mortality_fhs_manhattan.png",
        f"FHS mortality GWAS (event ~ SNP + age + sex, n={n_fhs})",
        args.discovery_p,
    )
    _plot_qq(df["p"].to_numpy(), out_dir / "mortality_fhs_qq.png", lam, "FHS mortality GWAS QQ")

    top10 = df.head(10)[["snp", "chrom", "pos", "beta", "p", "maf"]].to_dict(orient="records")
    summary = {
        "n_samples": int(n_fhs),
        "n_events": int(event.sum()),
        "n_snps_tested": int(len(df)),
        "lambda_gc": lam,
        "discovery_p_threshold": args.discovery_p,
        "n_genomewide": int((df["p"] <= args.discovery_p).sum()),
        "top10": top10,
    }
    write_json(out_dir / "mortality_fhs_summary.json", summary)
    print(f"[mortality] lambda_GC={lam:.3f}  n_genomewide={summary['n_genomewide']}")
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()

