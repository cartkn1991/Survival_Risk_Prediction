#!/usr/bin/env python3
"""Create correlation and TD-AUROC plots for epoch-26 joint model.

Outputs:
  - logh_vs_age_train_val_whi.png
  - td_auroc_train_val_whi.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plot_aesurv_diagnostics import (
    _compute_ipcw_td_auroc_curve,
    _split_indices,
    _color_event,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def _panel_corr(ax, age, risk, ev, title: str) -> None:
    colors = _color_event(ev.astype(int))
    ax.scatter(age, risk, c=colors, s=24, alpha=0.75, linewidths=0)
    m, b = np.polyfit(age, risk, 1)
    xs = np.linspace(float(np.min(age)), float(np.max(age)), 120)
    ax.plot(xs, m * xs + b, "--", color="#2c3e50", lw=1.6, label=f"linear fit (slope={m:.3f})")
    r_p, p_p = stats.pearsonr(age, risk)
    r_s, _ = stats.spearmanr(age, risk)
    ax.set_title(
        f"{title}\nPearson r={r_p:.3f} (p={p_p:.1e})  |  Spearman {r_s:.3f}",
        fontsize=15,
    )
    ax.set_xlabel("Chronological age (years)", fontsize=14)
    ax.set_ylabel("Predicted log-hazard", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper left", fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3)


def _panel_td(ax, cur, title: str) -> None:
    if cur is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=15)
        return
    et, aucs, mean_i = cur
    ax.plot(et, aucs, "o-", color="#2c3e50", lw=1.4, ms=4, label="TD-AUROC(t)")
    ax.axhline(mean_i, color="#c0392b", ls="--", lw=1.2, label=f"integrated mean = {mean_i:.3f}")
    ax.set_xlabel("Time (evaluation grid, event-time quantiles)", fontsize=14)
    ax.set_ylabel("Time-dependent AUROC", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title, fontsize=15)
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3)


def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parents[1]
    default_analysis = repo_root / "runs" / "aesurv_joint_epoch26_analysis"
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, default=default_analysis)
    args = p.parse_args()

    root = args.analysis_dir.resolve()
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    merged = root / "bio_relevance" / "lifestyle" / "lifestyle_risk_merged.parquet"
    out_corr = root / "logh_vs_age_train_val_whi.png"
    out_auc = root / "td_auroc_train_val_whi.png"

    df = pd.read_parquet(merged)
    fhs = df[df["cohort"].astype(str) == "FHS"].copy().reset_index(drop=True)
    whi = df[df["cohort"].astype(str) == "WHI"].copy().reset_index(drop=True)

    e_fhs = fhs["event"].astype(int).to_numpy()
    tr_idx, va_idx = _split_indices(len(fhs), 0.15, 42, stratify_event=e_fhs.astype(np.int32))

    age_fhs = fhs["age"].astype(float).to_numpy()
    risk_fhs = fhs["log_h"].astype(float).to_numpy()
    time_fhs = fhs["time"].astype(float).to_numpy()
    event_fhs = e_fhs.astype(np.int32)

    age_whi = whi["age"].astype(float).to_numpy()
    risk_whi = whi["log_h"].astype(float).to_numpy()
    time_whi = whi["time"].astype(float).to_numpy()
    event_whi = whi["event"].astype(int).to_numpy().astype(np.int32)

    # Correlation plot: train / val / whi
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=False)
    _panel_corr(axes[0], age_fhs[tr_idx], risk_fhs[tr_idx], event_fhs[tr_idx], "FHS training")
    _panel_corr(axes[1], age_fhs[va_idx], risk_fhs[va_idx], event_fhs[va_idx], "FHS validation")
    _panel_corr(axes[2], age_whi, risk_whi, event_whi, "WHI test")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#c0392b", label="event=1 (death)"),
        plt.Line2D([], [], marker="o", linestyle="", color="#7f8c8d", label="event=0 (censored)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=12)
    fig.suptitle("Chronological age vs predicted log-hazard", y=1.02, fontsize=17)
    fig.tight_layout()
    fig.savefig(out_corr, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # TD-AUROC: train / val / whi
    cur_train = _compute_ipcw_td_auroc_curve(
        time_fhs[tr_idx], event_fhs[tr_idx], time_fhs[tr_idx], event_fhs[tr_idx], risk_fhs[tr_idx]
    )
    cur_val = _compute_ipcw_td_auroc_curve(
        time_fhs[tr_idx], event_fhs[tr_idx], time_fhs[va_idx], event_fhs[va_idx], risk_fhs[va_idx]
    )
    cur_whi = _compute_ipcw_td_auroc_curve(
        time_whi, event_whi, time_whi, event_whi, risk_whi
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), sharey=True)
    _panel_td(axes[0], cur_train, "FHS training\n(IPCW censoring: FHS training)")
    _panel_td(axes[1], cur_val, "FHS validation\n(IPCW censoring: FHS training)")
    _panel_td(axes[2], cur_whi, "WHI test\n(IPCW censoring: WHI cohort)")
    fig.suptitle("Time-dependent AUROC (Uno's cumulative dynamic AUC)", y=1.02, fontsize=17)
    fig.tight_layout()
    fig.savefig(out_auc, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_corr}")
    print(f"Wrote {out_auc}")


if __name__ == "__main__":
    main()

