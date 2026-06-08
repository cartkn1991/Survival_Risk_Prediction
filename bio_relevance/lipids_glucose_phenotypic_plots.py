#!/usr/bin/env python3
"""Phenotypic plots for LDL/HDL/TC/TG/glucose vs AESurv log_h and mortality.

Reads:
  feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet

Writes:
  feature_importance/bio_relevance/lifestyle/figures/lipids_glucose_correlations.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bio_relevance.lifestyle_common import partial_spearman  # noqa: E402


TRAITS = ["ldl", "hdl", "total_cholesterol", "triglycerides", "glucose"]


def _zscore(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    m, s = v.mean(), v.std()
    if not np.isfinite(s) or s < 1e-8:
        return v * 0.0
    return (v - m) / s


def build_covariates(df: pd.DataFrame) -> np.ndarray:
    cols: List[np.ndarray] = []
    for c in ("age", "sex"):
        if c in df.columns:
            cols.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64))
    if "batch" in df.columns:
        dummies = pd.get_dummies(df["batch"].astype(str), drop_first=True)
        for c in dummies.columns:
            cols.append(dummies[c].to_numpy(dtype=np.float64))
    if not cols:
        return np.zeros((len(df), 1), dtype=np.float64)
    return np.column_stack(cols)


def correlation_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    y = pd.to_numeric(df["log_h"], errors="coerce").to_numpy(dtype=np.float64)
    cov = build_covariates(df)
    for trait in TRAITS:
        if trait not in df.columns:
            continue
        x = pd.to_numeric(df[trait], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            continue
        rho, p = stats.spearmanr(x[mask], y[mask])
        pr, pp = partial_spearman(y, x, cov)
        out[trait] = {
            "n": float(mask.sum()),
            "spearman_r": float(rho),
            "spearman_p": float(p),
            "partial_r": float(pr),
            "partial_p": float(pp),
        }
    return out


def make_plots(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    corr = correlation_summary(df)
    traits = list(corr.keys())
    if not traits:
        print("No lipid/glucose traits found in merged table.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: partial correlations bar plot
    ax = axes[0]
    vals = [corr[t]["partial_r"] for t in traits]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
    ax.barh(traits, vals, color=colors)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Partial Spearman r (adj. age, sex, batch)")
    ax.set_title("Lipids/glucose vs AESurv log_h (FHS)")

    # Panel B: scatter of standardized trait vs log_h (stacked)
    ax = axes[1]
    plot_df_rows: List[pd.DataFrame] = []
    for trait in traits:
        z = _zscore(df[trait])
        sub = pd.DataFrame(
            {
                "trait_z": z,
                "log_h": pd.to_numeric(df["log_h"], errors="coerce"),
                "trait": trait,
            }
        )
        plot_df_rows.append(sub)
    plot_df = pd.concat(plot_df_rows, ignore_index=True)
    plot_df = plot_df[np.isfinite(plot_df["trait_z"]) & np.isfinite(plot_df["log_h"])]
    sns.scatterplot(
        data=plot_df.sample(min(len(plot_df), 5000), random_state=7),
        x="trait_z",
        y="log_h",
        hue="trait",
        alpha=0.4,
        s=10,
        ax=ax,
    )
    ax.set_xlabel("Standardized trait (z)")
    ax.set_ylabel("AESurv log_h")
    ax.set_title("Per-trait scatter (subset of points)")
    ax.legend(title="Trait", fontsize=7, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--merged-parquet",
        type=str,
        default="feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default="feature_importance/bio_relevance/lifestyle/figures/lipids_glucose_correlations.pdf",
    )
    p.add_argument(
        "--fhs-only",
        action="store_true",
        help="Restrict to FHS cohort for plots.",
    )
    args = p.parse_args()

    path = Path(args.merged_parquet)
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run attach_dann_risk.py first.")
    df = pd.read_parquet(path)
    if args.fhs_only:
        df = df[df["cohort"].astype(str) == "FHS"].copy()
        print(f"FHS subset: n={len(df)}")
    make_plots(df, Path(args.out_path))


if __name__ == "__main__":
    main()

