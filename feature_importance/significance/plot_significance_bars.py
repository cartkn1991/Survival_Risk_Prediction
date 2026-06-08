#!/usr/bin/env python3
"""Quadrant horizontal bar plots from gradient significance tables.

Layout matches ``shap_bar_<target>.png`` in ``shap_feature_importance.py``:
2×2 grid — CpG/SNP × accelerate (positive mean_grad, red) / decelerate
(negative mean_grad, green).

Input tables (e.g. ``significant_risk.csv``) are produced by the significance
pipeline and contain: ``feature``, ``kind`` (``cpg`` / ``snp``), ``mean_grad``,
``direction``, etc.

Example::

  python feature_importance/significance/plot_significance_bars.py \\
      --csv significant_risk.csv

Fonts are bumped +2 pt vs the stock SHAP bar plot for readability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_n_samples(sig_dir: Path) -> int | None:
    p = sig_dir / "summary.json"
    if not p.is_file():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        return int(meta["n_samples"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _quadrant_table(df: pd.DataFrame, kind: str, positive: bool, k: int) -> pd.DataFrame:
    sub = df[df["kind"].str.lower() == kind].copy()
    if sub.empty:
        return sub
    if positive:
        sub = sub[sub["mean_grad"] > 0].sort_values("mean_grad", ascending=False)
    else:
        sub = sub[sub["mean_grad"] < 0].sort_values("mean_grad", ascending=True)
    return sub.head(k)


def plot_quadrant_bars(
    df: pd.DataFrame,
    *,
    out_path: Path,
    target_label: str,
    bar_k: int,
    n_samples: int | None,
) -> None:
    # +2 pt vs shap_feature_importance.py defaults for bar chart
    base_title = 10 + 2
    base_tick = 8 + 2
    base_suptitle = 12 + 2
    base_xlabel = 9 + 2

    bqk = max(1, int(bar_k))
    top_cpg_pos = _quadrant_table(df, "cpg", True, bqk)
    top_cpg_neg = _quadrant_table(df, "cpg", False, bqk)
    top_snp_pos = _quadrant_table(df, "snp", True, bqk)
    top_snp_neg = _quadrant_table(df, "snp", False, bqk)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4))

    def _bar(ax: plt.Axes, dff: pd.DataFrame, color: str, title: str) -> None:
        if dff.empty:
            ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes,
                    fontsize=base_title)
            ax.set_title(title, fontsize=base_title)
            ax.axis("off")
            return
        y = np.arange(len(dff))
        ax.barh(y, dff["mean_grad"], color=color, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(dff["feature"], fontsize=base_tick)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=base_title)
        ax.set_xlabel(f"mean gradient on {target_label}", fontsize=base_xlabel)
        ax.grid(True, axis="x", alpha=0.3)

    _bar(axes[0, 0], top_cpg_pos, "#c0392b", f"CpGs that ACCELERATE {target_label}")
    _bar(axes[0, 1], top_cpg_neg, "#27ae60", f"CpGs that DECELERATE {target_label}")
    _bar(axes[1, 0], top_snp_pos, "#c0392b", f"SNPs that ACCELERATE {target_label}")
    _bar(axes[1, 1], top_snp_neg, "#27ae60", f"SNPs that DECELERATE {target_label}")

    n_txt = f"{n_samples} samples" if n_samples is not None else "pooled samples"
    fig.suptitle(
        f"Significant features by signed mean gradient ({n_txt}) — target = {target_label}  "
        f"(bar quadrants: top {bqk} by signed mean_grad; table from significance pipeline)",
        fontsize=base_suptitle,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sig-dir", type=str, default=str(here),
                   help="Directory containing significance CSVs and summary.json")
    p.add_argument(
        "--csv",
        type=str,
        default="significant_risk.csv",
        help="Table name under --sig-dir (e.g. significant_risk.csv, significant_age.csv)",
    )
    p.add_argument(
        "--target",
        type=str,
        default="",
        help="Short label for titles/x-axis (default: inferred from CSV name, e.g. risk)",
    )
    p.add_argument("--bar-k", type=int, default=15, help="Top-K per quadrant")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path (default: <sig-dir>/sig_bar_<target>.png)",
    )
    args = p.parse_args()

    sig_dir = Path(args.sig_dir)
    csv_path = sig_dir / args.csv
    if not csv_path.is_file():
        raise SystemExit(f"Missing table: {csv_path}")

    target = args.target.strip()
    if not target:
        stem = csv_path.stem.lower()
        if stem.startswith("significant_"):
            target = stem.replace("significant_", "")
        else:
            target = stem

    out_path = Path(args.out) if args.out.strip() else sig_dir / f"sig_bar_{target}.png"

    df = pd.read_csv(csv_path)
    need = {"feature", "kind", "mean_grad"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns {sorted(missing)}; have {list(df.columns)}")

    n_samples = _load_n_samples(sig_dir)
    plot_quadrant_bars(df, out_path=out_path, target_label=target, bar_k=args.bar_k, n_samples=n_samples)
    print("Done.")


if __name__ == "__main__":
    main()
