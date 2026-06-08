#!/usr/bin/env python3
"""Plot mini-GWAS SNP effects: SNPs on y-axis, log-odds (beta) + 95% CI on x-axis.

Mini-GWAS tests association of each genotype with odds of being in the **accelerated** vs
**resilient** outlier contrast (see ``minigwas_outliers.py``).  Here:

  - **x-axis**: ``beta`` (log-odds for the coded allele toward **accelerated**), with Wald
    **95% CI** from ``se``.
  - **y-axis**: SNP identifiers (top ``--top-n`` by smallest ``p``).
  - **Color**: optional merge with ``significant_risk.csv`` → AESurv risk-gradient **direction**
    (**accelerator** / **decelerator**).  If no merge, points are colored by sign of ``beta``
    (toward accelerated vs resilient in the mini-GWAS only).

Outputs a single PNG (default ``minigwas_effects_forest.png`` next to the CSVs).

Example::

  python feature_importance/minigwas/plot_minigwas_effects.py

  python feature_importance/minigwas/plot_minigwas_effects.py --top-n 50 --metric z
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    need = {"snp", "beta", "se", "p", "or_acc"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{path}: missing columns {sorted(miss)}")
    return df


def merge_model_direction(mg: pd.DataFrame, sig_path: Path | None) -> pd.DataFrame:
    if sig_path is None or not sig_path.exists():
        mg = mg.copy()
        mg["model_direction"] = ""
        return mg
    sig = pd.read_csv(sig_path, low_memory=False)
    if "feature" not in sig.columns or "direction" not in sig.columns:
        raise SystemExit(f"{sig_path}: need feature, direction")
    snp = sig[sig["kind"].astype(str).str.lower().eq("snp")][["feature", "direction"]].copy()
    snp = snp.rename(columns={"feature": "snp"})
    out = mg.merge(snp, on="snp", how="left")
    out["model_direction"] = out["direction"].fillna("").astype(str)
    return out.drop(columns=["direction"], errors="ignore")


def pick_top(df: pd.DataFrame, top_n: int, metric: str) -> pd.DataFrame:
    d = df.copy()
    if metric == "z":
        if "z" not in d.columns:
            raise SystemExit("metric=z requires column 'z' in results CSV")
        d["_rank"] = d["z"].astype(float).abs()
    else:
        d["_rank"] = -np.log10(np.clip(d["p"].astype(float), 1e-300, None))
    d = d.sort_values("_rank", ascending=False).head(int(top_n))
    d = d.sort_values("beta", ascending=True).reset_index(drop=True)
    return d


def _annot_cell(val: object) -> str:
    """Return display string for rsid/gene; empty if missing (avoid literal 'nan' from float NaN)."""
    if val is None:
        return ""
    try:
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return ""
    except TypeError:
        pass
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return ""
    return s


def label_row(r: pd.Series, max_len: int = 44) -> str:
    rs = _annot_cell(r.get("rsids", np.nan))
    g = _annot_cell(r.get("gene_ensembl", np.nan))
    base = str(r["snp"])
    if rs:
        s = f"{base}  ({rs})"
    elif g:
        s = f"{base}  [{g}]"
    else:
        s = base
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def main() -> None:
    p = argparse.ArgumentParser(description="Forest plot of mini-GWAS SNP effects.")
    p.add_argument(
        "--results",
        type=str,
        default="feature_importance/minigwas/minigwas_results.csv",
    )
    p.add_argument(
        "--sig-csv",
        type=str,
        default="feature_importance/significance/significant_risk.csv",
        help="Optional: merge SNP model direction (accelerator/decelerator).",
    )
    p.add_argument("--no-sig-merge", action="store_true", help="Do not read significant_risk.csv.")
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--metric", type=str, default="p", choices=["p", "z"], help="Rank SNPs by -log10(p) or |z|.")
    p.add_argument("--out", type=str, default="feature_importance/minigwas/minigwas_effects_forest.png")
    p.add_argument("--z", type=float, default=1.96, help="CI multiplier (default 1.96 ~ 95%%).")
    args = p.parse_args()

    root = _root()
    res_path = Path(args.results) if Path(args.results).is_absolute() else root / args.results
    sig_path = None if args.no_sig_merge else (
        Path(args.sig_csv) if Path(args.sig_csv).is_absolute() else root / args.sig_csv
    )
    out_path = Path(args.out) if Path(args.out).is_absolute() else root / args.out

    df = load_results(res_path)
    df = merge_model_direction(df, sig_path)
    sub = pick_top(df, args.top_n, args.metric)

    beta = sub["beta"].astype(float).to_numpy()
    se = sub["se"].astype(float).to_numpy()
    lo = beta - args.z * se
    hi = beta + args.z * se
    y = np.arange(len(sub))
    labels = [label_row(sub.iloc[i]) for i in range(len(sub))]

    use_sig = bool(sig_path and sig_path.exists() and not args.no_sig_merge)

    colors = []
    for i in range(len(sub)):
        md = str(sub.iloc[i]["model_direction"]).lower().strip()
        if md == "accelerator":
            colors.append("#c0392b")
        elif md == "decelerator":
            colors.append("#2980b9")
        else:
            if use_sig:
                colors.append("#95a5a6")
            else:
                colors.append("#27ae60" if beta[i] >= 0 else "#8e44ad")

    fig_h = max(6.0, 0.32 * len(sub) + 2.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), facecolor="white")
    ax.axvline(0.0, color="#333333", linewidth=1.0, zorder=1)
    ax.errorbar(
        beta,
        y,
        xerr=[beta - lo, hi - beta],
        fmt="none",
        ecolor="#555555",
        elinewidth=1.0,
        capsize=2.5,
        zorder=2,
    )
    ax.scatter(beta, y, s=52, c=colors, edgecolors="#222222", linewidths=0.45, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(
        "Mini-GWAS effect (log-odds toward accelerated group)\n"
        "positive → allele associates with higher odds of accelerated vs resilient",
        fontsize=12,
    )
    ax.set_title(
        f"Top {len(sub)} SNPs by {'|z|' if args.metric == 'z' else '-log10(p)'}  "
        f"(mini-GWAS; n={int(sub['n_used'].iloc[0])} per test where constant)",
        fontsize=13,
        pad=10,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend
    if use_sig:
        handles = [
            Patch(facecolor="#c0392b", edgecolor="#222", label="Model: accelerator (risk gradient)"),
            Patch(facecolor="#2980b9", edgecolor="#222", label="Model: decelerator"),
        ]
    else:
        handles = [
            Patch(facecolor="#27ae60", edgecolor="#222", label="Mini-GWAS β ≥ 0 (toward accelerated)"),
            Patch(facecolor="#8e44ad", edgecolor="#222", label="Mini-GWAS β < 0 (toward resilient)"),
        ]
    ax.legend(handles=handles, loc="lower right", fontsize=10, title_fontsize=10, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
