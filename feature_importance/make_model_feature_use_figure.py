#!/usr/bin/env python3
"""One-page figure: AESurv feature-use pipeline + risk gradient volcano + biology overlap.

Outputs (default ``feature_importance/figures/``):
  model_feature_use_onepage.pdf
  model_feature_use_onepage.png

If ``significance/significance_risk.npz`` exists (from ``select_significant_features.py``),
the volcano includes a subsampled genome-wide background; otherwise only selected
features are plotted (still valid for the paper).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
def _load_summary(sig_dir: Path) -> dict:
    return json.loads((sig_dir / "summary.json").read_text(encoding="utf-8"))


def _draw_pipeline(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    boxes = [
        (0.2, 7.2, 2.6, 1.4, "Raw omics\n393k CpGs + 979k SNPs\n(N = 3,719 pooled)"),
        (3.2, 7.2, 2.4, 1.4, "Scaler + JL\n→ 2048-d"),
        (5.9, 7.2, 2.4, 1.4, "Frozen DANN\n→ 128-d latent"),
        (3.2, 4.8, 2.4, 1.4, "AESurv head\n8-d bottleneck z"),
        (5.9, 4.8, 2.4, 1.4, "Outputs\nlog-hazard, age,\ncells"),
        (0.2, 2.0, 4.0, 1.6, "Gradients ∂(Σ log-h)\n/ ∂(projected x)\nper sample"),
        (4.6, 2.0, 4.8, 1.6, "Back-project via W\n→ mean d(risk)/d(feature)\n5σ MAD + FDR filter"),
    ]
    for x, y, w, h, txt in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8)
    arrows = [
        ((2.8, 7.9), (3.2, 7.9)),
        ((5.6, 7.9), (5.9, 7.9)),
        ((7.1, 7.2), (7.1, 6.2)),
        ((7.1, 6.2), (4.4, 6.2)),
        ((4.4, 4.8), (4.4, 3.6)),
        ((2.2, 2.8), (4.6, 2.8)),
        ((7.1, 4.8), (7.1, 3.6)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.2))
    ax.text(5, 9.5, "Why the model consistently uses selected features",
            ha="center", fontsize=11, fontweight="bold")


def _volcano_panel(ax, sig_dir: Path, meta: dict, max_bg: int = 80_000) -> None:
    thr = float(meta["risk_threshold"]["threshold"])
    npz_path = sig_dir / "significance_risk.npz"
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        mean_x = z["mean_x"]
        q = z["q"]
        selected = z["selected"].astype(bool)
        n_cpg = int(z["n_cpg"])
        is_cpg = np.arange(mean_x.size) < n_cpg
        rng = np.random.default_rng(0)
        bg_idx = np.flatnonzero(~selected)
        if bg_idx.size > max_bg:
            bg_idx = rng.choice(bg_idx, size=max_bg, replace=False)
        idx = np.unique(np.concatenate([bg_idx, np.flatnonzero(selected)]))
        mean_x, q, is_cpg, selected = mean_x[idx], q[idx], is_cpg[idx], selected[idx]
        subtitle = "genome-wide subsample (gray) + selected (color)"
    else:
        df = pd.read_csv(sig_dir / "significant_risk.csv")
        mean_x = df["mean_grad"].to_numpy()
        q = df["q"].to_numpy()
        is_cpg = (df["kind"] == "cpg").to_numpy()
        selected = np.ones(len(df), dtype=bool)
        subtitle = "selected features only (run select_significant_features.py for full volcano)"

    logq = -np.log10(np.clip(q.astype(float), 1e-300, 1.0))
    ns_cpg = ~selected & is_cpg
    ns_snp = ~selected & ~is_cpg
    ax.scatter(mean_x[ns_cpg], logq[ns_cpg], s=1, c="#bdc3c7", alpha=0.2, linewidths=0, rasterized=True)
    ax.scatter(mean_x[ns_snp], logq[ns_snp], s=1, c="#dadfe1", alpha=0.2, linewidths=0, rasterized=True)
    s_cpg = selected & is_cpg
    s_snp = selected & ~is_cpg
    ax.scatter(mean_x[s_cpg], logq[s_cpg], s=6, c="#2980b9", alpha=0.75, linewidths=0, label=f"CpG selected ({int(s_cpg.sum()):,})")
    ax.scatter(mean_x[s_snp], logq[s_snp], s=6, c="#c0392b", alpha=0.75, linewidths=0, label=f"SNP selected ({int(s_snp.sum()):,})")
    ax.axvline(+thr, color="black", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(-thr, color="black", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"mean $\partial$(log-hazard)$/\partial$(feature)")
    ax.set_ylabel(r"$-\log_{10}$(BH FDR)")
    ax.set_title(f"Risk-target gradient selection  |  {subtitle}", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=True, markerscale=1.5)
    ax.grid(True, alpha=0.25)


def _counts_panel(ax, meta: dict) -> None:
    labels = ["Risk SNPs", "Risk CpGs", "Age SNPs", "Age CpGs", "Both SNPs"]
    vals = [
        meta["n_selected_risk_snp"],
        meta["n_selected_risk_cpg"],
        meta["n_selected_age_snp"],
        meta["n_selected_age_cpg"],
        meta["n_selected_both_snp"],
    ]
    colors = ["#c0392b", "#2980b9", "#e74c3c", "#3498db", "#8e44ad"]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Features passing 5σ MAD & q < 0.05")
    ax.set_title("Selection counts (N = {:,} samples)".format(meta["n_samples"]), fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=7)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)


def _biology_panel(ax, gene_summary_path: Path) -> None:
    g = json.loads(gene_summary_path.read_text(encoding="utf-8"))
    b = g["blocks"]["significant_risk"]
    labels = ["SNP genes", "CpG genes", "Shared", "SNP-only genes"]
    vals = [b["n_unique_genes_snp"], b["n_unique_genes_cpg"], b["n_shared_genes_cpg_snp"], b["n_only_snp_genes"]]
    colors = ["#c0392b", "#2980b9", "#27ae60", "#e67e22"]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Unique gene symbols")
    ax.set_title("Biological concordance (risk-selected features)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    root = Path(__file__).resolve().parent
    sig_dir = root / "significance"
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_summary(sig_dir)
    gene_path = root / "gene_overlap_from_features" / "gene_overlap_summary.json"

    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.32)
    ax_pipe = fig.add_subplot(gs[0, :])
    ax_vol = fig.add_subplot(gs[1, 0])
    gs_r = gs[1, 1].subgridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.45)
    ax_cnt = fig.add_subplot(gs_r[0, 0])
    ax_bio = fig.add_subplot(gs_r[1, 0])

    _draw_pipeline(ax_pipe)
    _volcano_panel(ax_vol, sig_dir, meta)
    _counts_panel(ax_cnt, meta)
    if gene_path.exists():
        _biology_panel(ax_bio, gene_path)
    fig.text(0.76, 0.02,
             f"Filter: |mean grad| > {meta['risk_threshold']['threshold']:.2e} "
             f"({meta['k_sigma']}×MAD, FDR<{meta['fdr']}); "
             f"{meta['tallies_risk']['n_q05']:,} pass FDR only, "
             f"{meta['n_selected_risk']:,} pass both.",
             ha="center", fontsize=7.5)

    fig.suptitle("AESurv-DANN-Aux: consistent mortality sensitivity across 1.37M omics features",
                 fontsize=12, fontweight="bold", y=0.98)
    for ext in ("pdf", "png"):
        out = out_dir / f"model_feature_use_onepage.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
