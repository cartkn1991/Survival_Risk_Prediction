#!/usr/bin/env python3
"""Second lifestyle figure: raw vs adjusted alcohol–log_h (batch confounding).

Outputs:
  feature_importance/bio_relevance/lifestyle/figures/lifestyle_confounding_figure.pdf
  feature_importance/bio_relevance/lifestyle/figures/lifestyle_confounding_figure.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bio_relevance.lifestyle_risk_validation import build_covariate_matrix

BATCH_ORDER = ["Gen 3", "UMN", "JHU"]
BATCH_COLORS = {"Gen 3": "#2ecc71", "UMN": "#3498db", "JHU": "#e67e22"}


def _residualize(y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """OLS residuals of y on [intercept, cov]."""
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones(len(y)), cov])
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full(len(y), np.nan, dtype=np.float64)
    if mask.sum() < 10:
        return out
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    out[mask] = y[mask] - X[mask] @ coef
    return out


def _alc_col(df: pd.DataFrame) -> str:
    if "alcohol_amount_per_occasion" in df.columns:
        return "alcohol_amount_per_occasion"
    if "alcohol_drinks_per_week" in df.columns:
        return "alcohol_drinks_per_week"
    raise ValueError("No alcohol column in merged table")


def _scatter_by_batch(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    batch: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    r_label: str,
) -> None:
    for b in BATCH_ORDER:
        m = (batch.astype(str) == b) & np.isfinite(x) & np.isfinite(y)
        if m.sum() == 0:
            continue
        ax.scatter(
            x[m], y[m], c=BATCH_COLORS.get(b, "#7f8c8d"), label=b,
            alpha=0.35, s=14, edgecolors="none",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.text(
        0.03, 0.97, r_label, transform=ax.transAxes, va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def make_confounding_figure(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["cohort"].astype(str) == "FHS"].copy()
    alc = _alc_col(sub)
    sub = sub.dropna(subset=[alc, "log_h", "age", "batch"]).copy()
    if len(sub) < 50:
        raise ValueError(f"Too few FHS rows with lifestyle: {len(sub)}")

    x_raw = pd.to_numeric(sub[alc], errors="coerce").to_numpy(dtype=np.float64)
    y_raw = sub["log_h"].to_numpy(dtype=np.float64)
    age = pd.to_numeric(sub["age"], errors="coerce").to_numpy(dtype=np.float64)
    batch = sub["batch"].astype(str).to_numpy()

    cov = build_covariate_matrix(sub, include_batch=True)
    x_res = _residualize(x_raw, cov)
    y_res = _residualize(y_raw, cov)

    r_raw, p_raw = stats.spearmanr(x_raw, y_raw)
    mask = np.isfinite(x_res) & np.isfinite(y_res)
    r_adj, p_adj = stats.spearmanr(x_res[mask], y_res[mask])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    _scatter_by_batch(
        axes[0, 0], x_raw, y_raw, batch,
        xlabel="Alcohol amount per occasion",
        ylabel="Predicted log-hazard (DANN-Aux)",
        title="A  Raw association",
        r_label=f"Spearman ρ = {r_raw:.3f}\np = {p_raw:.2e}",
    )

    _scatter_by_batch(
        axes[0, 1], x_res[mask], y_res[mask], batch[mask],
        xlabel="Alcohol residual (adj. age, sex, batch)",
        ylabel="log_h residual (adj. age, sex, batch)",
        title="B  Adjusted association (partial)",
        r_label=f"Spearman ρ = {r_adj:.3f}\np = {p_adj:.3f}",
    )
    axes[0, 1].axhline(0, color="k", lw=0.6, alpha=0.5)
    axes[0, 1].axvline(0, color="k", lw=0.6, alpha=0.5)

    _scatter_by_batch(
        axes[1, 0], age, x_raw, batch,
        xlabel="Age (years)",
        ylabel="Alcohol amount per occasion",
        title="C  Why raw correlation is misleading",
        r_label="Younger cohorts (Gen 3)\ndrink more but have lower log_h",
    )

    _scatter_by_batch(
        axes[1, 1], age, y_raw, batch,
        xlabel="Age (years)",
        ylabel="Predicted log-hazard",
        title="D  Age and risk by batch",
        r_label="log_h rises with age;\nbatch ≈ age in FHS",
    )

    fig.suptitle(
        "FHS lifestyle confounding: alcohol vs DANN-Aux log_h (n={:,})".format(len(sub)),
        fontsize=12, y=1.01,
    )
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
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/lifestyle")
    p.add_argument("--fhs-only", action="store_true", default=True)
    args = p.parse_args()

    df = pd.read_parquet(args.merged_parquet)
    if args.fhs_only:
        m = (df["cohort"].astype(str) == "FHS")
        if "cigarettes_per_day" in df.columns:
            m &= df["cigarettes_per_day"].notna()
        df = df.loc[m].copy()

    out = Path(args.out_dir) / "figures" / "lifestyle_confounding_figure.pdf"
    make_confounding_figure(df, out)


if __name__ == "__main__":
    main()
