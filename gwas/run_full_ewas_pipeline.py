#!/usr/bin/env python3
"""FHS full-genome EWAS pipeline (mortality + log_h discovery).

Stages
------
1. mortality  — event ~ CpG_meth + age + sex (all FHS CpGs)
2. discovery  — log_h_residual ~ CpG_meth + age + sex
3. cross        — overlap mortality vs discovery EWAS hits

Example
-------
cd D:\\SNP_datasets
$env:PYTHONPATH="D:\\SNP_datasets"
python gwas/run_full_ewas_pipeline.py --stage all \\
  --out-dir runs/aesurv_joint_epoch26_analysis/feature_importance/ewas \\
  --risk-parquet runs/aesurv_joint_epoch26_analysis/bio_relevance/lifestyle/lifestyle_risk_merged.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gwas.gwas_common import (
    annotate_cpg_df,
    batch_linear_ewas,
    bh_fdr,
    compute_logh_residual,
    genomic_lambda,
    load_cpg_annotation,
    load_fhs_logh_from_parquet,
    read_age_sex,
    read_txt_list,
    write_json,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def _plot_manhattan(df: pd.DataFrame, out_path: Path, title: str, p_line: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    d = df.sort_values(["chrom", "pos"]).reset_index(drop=True)
    # Plot at most 80k points for responsiveness on full methylome scans.
    if len(d) > 80_000:
        top = d.nsmallest(40_000, "p")
        rest_idx = d.index.difference(top.index)
        if len(rest_idx) > 40_000:
            rest_idx = np.random.default_rng(42).choice(rest_idx.to_numpy(), size=40_000, replace=False)
        d = pd.concat([top, d.loc[rest_idx]], ignore_index=True).sort_values(["chrom", "pos"])
    chroms = sorted(d["chrom"].unique())
    cum = 0
    tick_pos, tick_lab = [], []
    palette = ["#8e44ad", "#9b59b6"]
    for i, ch in enumerate(chroms):
        sub = d[d["chrom"] == ch]
        x = cum + np.arange(len(sub))
        ax.scatter(
            x, -np.log10(sub["p"].clip(lower=1e-300)),
            c=palette[i % 2], s=3, linewidths=0, alpha=0.8, rasterized=True,
        )
        tick_pos.append(cum + len(sub) / 2)
        tick_lab.append(str(int(ch)) if ch <= 22 else ("X" if ch == 23 else "?"))
        cum += len(sub)
    ax.axhline(-np.log10(p_line), color="#c0392b", lw=0.8, ls="--", label=f"p={p_line:g}")
    ax.set_xticks(tick_pos[:: max(1, len(tick_pos) // 24)])
    ax.set_xticklabels(tick_lab[:: max(1, len(tick_lab) // 24)], fontsize=7)
    ax.set_xlabel("Chromosome (CpG position)")
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
    ax.scatter(-np.log10(p_exp), -np.log10(p_obs.clip(min=1e-300)), s=4, c="#2c3e50", linewidths=0)
    mx = max(-np.log10(p_exp.min()), -np.log10(p_obs.min().clip(min=1e-300)))
    ax.plot([0, mx], [0, mx], color="#c0392b", lw=1)
    ax.set_title(f"{title}   (lambda_GC = {lam:.3f})", fontsize=11)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_ewas_scan(
    y: np.ndarray,
    X_meth,
    cpg_names: List[str],
    cov: np.ndarray,
    annot: pd.DataFrame,
    *,
    batch_size: int,
    cpg_limit: int,
    label: str,
) -> pd.DataFrame:
    n_cpgs = len(cpg_names)
    if cpg_limit > 0:
        n_cpgs = min(n_cpgs, cpg_limit)
        cpg_names = cpg_names[:n_cpgs]
    print(f"[{label}] CpGs to test: {n_cpgs:,}")
    chunks: List[pd.DataFrame] = []
    t0 = time.time()
    for start in range(0, n_cpgs, batch_size):
        end = min(n_cpgs, start + batch_size)
        G = X_meth[:, start:end].astype(np.float32)
        batch_df = batch_linear_ewas(y, G, cov, cpg_names[start:end])
        chunks.append(batch_df)
        if (start // batch_size) % 10 == 0 or end == n_cpgs:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1e-6)
            print(f"  [{end:,}/{n_cpgs:,}] CpGs  {rate:.0f} CpG/s")
    df = pd.concat(chunks, ignore_index=True)
    df = annotate_cpg_df(df, annot)
    df["q"] = bh_fdr(df["p"].to_numpy())
    return df.sort_values(["p", "chrom", "pos"]).reset_index(drop=True)


def run_mortality(args: argparse.Namespace, out_dir: Path, annot: pd.DataFrame) -> pd.DataFrame:
    out_csv = out_dir / "mortality_fhs_ewas_results.csv"
    if out_csv.exists() and not args.force:
        print(f"[mortality] Loading cached {out_csv}")
        return pd.read_csv(out_csv)

    z = np.load(args.fhs_npz, mmap_mode="r")
    n_fhs = int(z["X_meth"].shape[0])
    event = z["event"].astype(np.float32)
    age, sex = read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    age_z = (age - age.mean()) / (age.std() + 1e-6)
    cov = np.column_stack([np.ones(n_fhs), age_z.astype(np.float64), sex.astype(np.float64)])

    cpg_names = read_txt_list(Path(args.cpg_columns_txt))
    df = _run_ewas_scan(
        event, z["X_meth"], cpg_names, cov, annot,
        batch_size=args.batch_size, cpg_limit=args.cpg_limit, label="mortality",
    )
    df.to_csv(out_csv, index=False)
    lam = genomic_lambda(df["p"].to_numpy())
    _plot_manhattan(df, out_dir / "mortality_fhs_ewas_manhattan.png",
                    f"FHS mortality EWAS (event ~ CpG + age + sex, n={n_fhs})", args.ewas_p)
    _plot_qq(df["p"].to_numpy(), out_dir / "mortality_fhs_ewas_qq.png", lam, "FHS mortality EWAS QQ")
    write_json(out_dir / "mortality_fhs_ewas_summary.json", {
        "n_samples": n_fhs, "n_events": int(event.sum()), "n_cpgs_tested": len(df),
        "lambda_gc": lam, "n_p_lt_0.05": int((df["p"] < 0.05).sum()),
        "top10": df.head(10).to_dict(orient="records"),
    })
    print(f"[mortality] Wrote {out_csv}  n={len(df):,}  lambda={lam:.3f}")
    return df


def run_discovery(args: argparse.Namespace, out_dir: Path, annot: pd.DataFrame) -> pd.DataFrame:
    out_csv = out_dir / "discovery_fhs_ewas_results.csv"
    if out_csv.exists() and not args.force:
        print(f"[discovery] Loading cached {out_csv}")
        return pd.read_csv(out_csv)

    z = np.load(args.fhs_npz, mmap_mode="r")
    n_fhs = int(z["X_meth"].shape[0])
    age, sex = read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    log_h = load_fhs_logh_from_parquet(
        Path(args.fhs_meta_pq), args.fhs_id_col, Path(args.risk_parquet), n_fhs,
    )
    y, pheno_info = compute_logh_residual(log_h, age, sex=sex)
    cov = np.column_stack([np.ones(n_fhs), age.astype(np.float64), sex.astype(np.float64)])

    cpg_names = read_txt_list(Path(args.cpg_columns_txt))
    df = _run_ewas_scan(
        y, z["X_meth"], cpg_names, cov, annot,
        batch_size=args.batch_size, cpg_limit=args.cpg_limit, label="discovery",
    )
    df.to_csv(out_csv, index=False)
    lam = genomic_lambda(df["p"].to_numpy())
    _plot_manhattan(df, out_dir / "discovery_fhs_ewas_manhattan.png",
                    f"FHS discovery EWAS: log_h residual (n={n_fhs})", args.ewas_p)
    _plot_qq(df["p"].to_numpy(), out_dir / "discovery_fhs_ewas_qq.png", lam, "FHS discovery EWAS QQ")
    write_json(out_dir / "discovery_fhs_ewas_summary.json", {
        "n_samples": n_fhs, "phenotype": "age+sex adjusted log_h residual",
        "pheno_fit": pheno_info, "n_cpgs_tested": len(df), "lambda_gc": lam,
        "n_p_lt_0.05": int((df["p"] < 0.05).sum()),
        "top10": df.head(10).to_dict(orient="records"),
    })
    print(f"[discovery] Wrote {out_csv}  n={len(df):,}  lambda={lam:.3f}")
    return df


def run_cross_compare(mort: pd.DataFrame, disc: pd.DataFrame, out_dir: Path, *, p_cut: float) -> None:
    mort_hits = mort.loc[mort["p"] < p_cut, ["cpg", "beta", "p", "gene"]].rename(
        columns={"beta": "mortality_beta", "p": "mortality_p", "gene": "mortality_gene"}
    )
    disc_hits = disc.loc[disc["p"] < p_cut, ["cpg", "beta", "p", "gene"]].rename(
        columns={"beta": "discovery_beta", "p": "discovery_p", "gene": "discovery_gene"}
    )
    cross = mort_hits.merge(disc_hits, on="cpg", how="inner")
    cross["same_sign"] = np.sign(cross["mortality_beta"]) == np.sign(cross["discovery_beta"])
    cross["gene"] = cross["mortality_gene"].fillna("").replace("", np.nan).fillna(cross["discovery_gene"])
    cross = cross.drop(columns=["mortality_gene", "discovery_gene"]).sort_values("cpg")
    cross.to_csv(out_dir / "ewas_cross_compare_overlap.csv", index=False)
    overlap = cross["cpg"].tolist()
    write_json(out_dir / "ewas_cross_compare_summary.json", {
        "p_cut": p_cut,
        "n_mortality_hits": int(len(mort_hits)),
        "n_discovery_hits": int(len(disc_hits)),
        "n_exact_overlap": len(overlap),
        "n_same_sign": int(cross["same_sign"].sum()) if not cross.empty else 0,
        "overlap_cpgs": overlap,
    })
    print(f"[cross] mortality hits={len(mort_hits)}  discovery hits={len(disc_hits)}  overlap={len(overlap)}")


def main() -> None:
    p = argparse.ArgumentParser(description="FHS full-genome EWAS pipeline")
    p.add_argument("--stage", choices=["mortality", "discovery", "cross", "all"], default="all")
    p.add_argument("--fhs-npz", default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-meta-pq", default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", default="Share_ID")
    p.add_argument("--cpg-columns-txt", default="FHS_methylation_with_snp_1milfeatures_cpg_columns.txt")
    p.add_argument("--annot-csv", default="Annotation.csv")
    p.add_argument("--risk-parquet", required=False,
                   default="runs/aesurv_joint_epoch26_analysis/bio_relevance/lifestyle/lifestyle_risk_merged.parquet")
    p.add_argument("--out-dir", default="feature_importance/ewas")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--cpg-limit", type=int, default=0, help="Smoke test: first N CpGs only")
    p.add_argument("--ewas-p", type=float, default=1e-5)
    p.add_argument("--cross-p", type=float, default=0.05)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annot = load_cpg_annotation(Path(args.annot_csv))
    print(f"[ewas] Loaded annotation for {len(annot):,} CpGs")

    mort_df = disc_df = None
    if args.stage in ("mortality", "all"):
        mort_df = run_mortality(args, out_dir, annot)
    if args.stage in ("discovery", "all"):
        if not Path(args.risk_parquet).exists():
            raise SystemExit(f"--risk-parquet not found: {args.risk_parquet}")
        disc_df = run_discovery(args, out_dir, annot)
    if args.stage in ("cross", "all"):
        if mort_df is None:
            mort_df = pd.read_csv(out_dir / "mortality_fhs_ewas_results.csv")
        if disc_df is None:
            disc_df = pd.read_csv(out_dir / "discovery_fhs_ewas_results.csv")
        run_cross_compare(mort_df, disc_df, out_dir, p_cut=args.cross_p)

    print(f"\nDone. EWAS outputs in {out_dir}/")


if __name__ == "__main__":
    main()
