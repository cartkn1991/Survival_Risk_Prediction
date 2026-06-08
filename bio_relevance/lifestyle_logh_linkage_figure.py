#!/usr/bin/env python3
"""Lifestyle–log_h linkage figures (simple supplement + main dual-story figure).

Outputs:
  figures/lifestyle_logh_association_simple.pdf
  figures/lifestyle_logh_linkage_main.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bio_relevance.lifestyle_confounding_figure import BATCH_COLORS, BATCH_ORDER, _residualize
from bio_relevance.lifestyle_risk_validation import build_covariate_matrix

TRAITS: List[Tuple[str, str, str]] = [
    ("cigarettes_per_day", "Cigarettes per day", "Cigarettes/day"),
    ("alcohol_amount_per_occasion", "Alcohol per occasion", "Alcohol/occasion"),
    ("sleep_hours", "Sleep hours", "Sleep (h)"),
]


def _prep_fhs(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["cohort"].astype(str) == "FHS"].copy()
    need = ["log_h", "age", "sex", "batch", "cigarettes_per_day", "sleep_hours"]
    if "alcohol_amount_per_occasion" not in sub.columns and "alcohol_drinks_per_week" in sub.columns:
        sub["alcohol_amount_per_occasion"] = sub["alcohol_drinks_per_week"]
    need.append("alcohol_amount_per_occasion")
    sub = sub.dropna(subset=[c for c in need if c in sub.columns]).copy()
    if len(sub) < 50:
        raise ValueError(f"Too few FHS rows: {len(sub)}")
    return sub


def _add_loess(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(y[m], x[m], frac=0.35, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color="k", ls="--", lw=1.8, alpha=0.85, zorder=5)
    except ImportError:
        xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 80)
        coef = np.polyfit(x[m], y[m], 2)
        ax.plot(xs, np.polyval(coef, xs), color="k", ls="--", lw=1.8, alpha=0.85, zorder=5)


def _trait_correlations(sub: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    y = sub["log_h"].to_numpy(dtype=np.float64)
    cov = build_covariate_matrix(sub, include_batch=True)
    y_res = _residualize(y, cov)
    out: Dict[str, Dict[str, float]] = {}
    for col, _, _ in TRAITS:
        x = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
        r_raw, p_raw = stats.spearmanr(x, y)
        x_res = _residualize(x, cov)
        m = np.isfinite(x_res) & np.isfinite(y_res)
        r_adj, p_adj = stats.spearmanr(x_res[m], y_res[m]) if m.sum() >= 20 else (np.nan, np.nan)
        out[col] = {
            "rho_raw": float(r_raw), "p_raw": float(p_raw),
            "rho_partial": float(r_adj), "p_partial": float(p_adj),
        }
    return out


def _raw_scatter_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    col: str,
    xlabel: str,
    title: str,
    corr: Dict[str, float],
    *,
    show_ylabel: bool = True,
    show_legend: bool = False,
) -> None:
    x = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
    y = sub["log_h"].to_numpy(dtype=np.float64)
    batch = sub["batch"].astype(str).to_numpy()
    for b in BATCH_ORDER:
        m = (batch == b) & np.isfinite(x) & np.isfinite(y)
        if m.sum() == 0:
            continue
        ax.scatter(
            x[m], y[m], c=BATCH_COLORS.get(b, "#7f8c8d"), label=b,
            alpha=0.35, s=14, edgecolors="none",
        )
    _add_loess(ax, x, y)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Predicted log-hazard")
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.text(
        0.03, 0.97, f"ρ = {corr['rho_raw']:.3f}\np = {corr['p_raw']:.2e}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _adjusted_scatter_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    col: str,
    xlabel: str,
    title: str,
    corr: Dict[str, float],
    *,
    show_ylabel: bool = True,
) -> None:
    cov = build_covariate_matrix(sub, include_batch=True)
    x = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
    y = sub["log_h"].to_numpy(dtype=np.float64)
    x_res = _residualize(x, cov)
    y_res = _residualize(y, cov)
    m = np.isfinite(x_res) & np.isfinite(y_res)
    ax.scatter(x_res[m], y_res[m], c="#5d6d7e", alpha=0.3, s=12, edgecolors="none")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    _add_loess(ax, x_res[m], y_res[m])
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("log_h residual")
    ax.set_title(title)
    ax.text(
        0.03, 0.97,
        f"partial ρ = {corr['rho_partial']:.3f}\np = {corr['p_partial']:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def make_simple_figure(df: pd.DataFrame, out_path: Path) -> None:
    sub = _prep_fhs(df)
    corrs = _trait_correlations(sub)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, (col, xlabel, title)) in enumerate(zip(axes, TRAITS)):
        _raw_scatter_panel(
            ax, sub, col, xlabel, title, corrs[col],
            show_ylabel=(i == 0), show_legend=(i == 2),
        )
    fig.suptitle(
        f"Lifestyle vs DANN-Aux log-hazard (FHS, n={len(sub):,}) — unadjusted",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_path)


def make_main_figure(
    df: pd.DataFrame,
    summary: Optional[Dict[str, Any]],
    out_path: Path,
) -> None:
    sub = _prep_fhs(df)
    corrs = _trait_correlations(sub)

    fig = plt.figure(figsize=(12, 11))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 0.85], hspace=0.38, wspace=0.32)

    for j, (col, xlabel, short) in enumerate(TRAITS):
        ax_raw = fig.add_subplot(gs[0, j])
        _raw_scatter_panel(
            ax_raw, sub, col, xlabel, f"Raw: {short}", corrs[col],
            show_ylabel=(j == 0),
        )
        ax_adj = fig.add_subplot(gs[1, j])
        _adjusted_scatter_panel(
            ax_adj, sub, col, f"{xlabel} (residual)", f"Adjusted: {short}", corrs[col],
            show_ylabel=(j == 0),
        )

    # G: heatmap
    ax_hm = fig.add_subplot(gs[2, 0])
    labels = [t[2] for t in TRAITS]
    mat = np.array([
        [corrs[c]["rho_raw"] for c, _, _ in TRAITS],
        [corrs[c]["rho_partial"] for c, _, _ in TRAITS],
    ])
    vmax = max(0.5, np.nanmax(np.abs(mat)))
    im = ax_hm.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_hm.set_xticks(range(len(labels)))
    ax_hm.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax_hm.set_yticks([0, 1])
    ax_hm.set_yticklabels(["ρ raw", "ρ partial"], fontsize=9)
    ax_hm.set_title("G  Correlation summary")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_hm.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9, color="k")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    # H: Cox C-index
    ax_cox = fig.add_subplot(gs[2, 1])
    cox = (summary or {}).get("cox_models", {}).get("cohorts", {}).get("FHS", {})
    palette = {"M0": "#95a5a6", "M1": "#3498db", "M2": "#e74c3c", "M3": "#8e44ad"}
    names, vals, cols = [], [], []
    for m in ["M0", "M1", "M2", "M3"]:
        if m in cox and cox[m].get("c_index") is not None:
            names.append(m)
            vals.append(cox[m]["c_index"])
            cols.append(palette[m])
    if vals:
        ax_cox.bar(names, vals, color=cols)
        ax_cox.set_ylim(0.78, 0.86)
        ax_cox.set_ylabel("Harrell C-index")
        ax_cox.set_title("H  Mortality discrimination")
        for i, v in enumerate(vals):
            ax_cox.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    else:
        ax_cox.text(0.5, 0.5, "Cox results unavailable", ha="center", transform=ax_cox.transAxes)

    # I: residual risk callout
    ax_res = fig.add_subplot(gs[2, 2])
    ax_res.axis("off")
    resid = (summary or {}).get("residual_risk", {}).get("cox", {}).get("FHS", {})
    hr = resid.get("hr_per_sd", float("nan"))
    p = resid.get("p", float("nan"))
    ci = resid.get("c_index_resid_only", float("nan"))
    lines = [
        "I  log_h beyond lifestyle",
        "",
        "log_h_resid = log_h adjusted for:",
        "  age, sex, batch, cigarettes,",
        "  alcohol, sleep",
        "",
    ]
    if np.isfinite(hr):
        lines.append(f"Cox HR per SD: {hr:.2f}")
        lines.append(f"p = {p:.2e}" if p < 0.001 else f"p = {p:.4f}")
    if np.isfinite(ci):
        lines.append(f"C-index (resid only): {ci:.3f}")
    lines.extend([
        "",
        "Omics risk predicts mortality",
        "after measured lifestyle is removed.",
    ])
    ax_res.text(
        0.05, 0.95, "\n".join(lines), transform=ax_res.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="#fef9e7", alpha=0.95),
    )

    fig.suptitle(
        f"Lifestyle–log_h linkage (FHS, n={len(sub):,}): confounding vs direct association",
        fontsize=12, y=1.01,
    )
    _save(fig, out_path)


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("simple", "main", "both"), default="both")
    p.add_argument(
        "--merged-parquet",
        type=str,
        default="feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet",
    )
    p.add_argument("--summary-json", type=str,
                   default="feature_importance/bio_relevance/lifestyle/lifestyle_validation_summary.json")
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/lifestyle")
    args = p.parse_args()

    df = pd.read_parquet(args.merged_parquet)
    m = (df["cohort"].astype(str) == "FHS")
    if "cigarettes_per_day" in df.columns:
        m &= df["cigarettes_per_day"].notna()
    df = df.loc[m].copy()

    summary: Optional[Dict[str, Any]] = None
    sp = Path(args.summary_json)
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))

    fig_dir = Path(args.out_dir) / "figures"
    if args.mode in ("simple", "both"):
        make_simple_figure(df, fig_dir / "lifestyle_logh_association_simple.pdf")
    if args.mode in ("main", "both"):
        make_main_figure(df, summary, fig_dir / "lifestyle_logh_linkage_main.pdf")


if __name__ == "__main__":
    main()
