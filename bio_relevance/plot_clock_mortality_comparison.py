#!/usr/bin/env python3
"""Two-panel figure: mortality C-index for methylation clocks vs AESurv log_h.

Reads feature_importance/bio_relevance/clocks/clock_mortality_cindex.json and writes:
  clock_mortality_comparison.pdf / .png
  clock_mortality_comparison_summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Main predictors only (exclude *_accel)
MAIN_PREDICTORS = [
    "GrimAge",
    "Hannum",
    "Horvath",
    "PhenoAge",
    "EpiClock",
    "chronological_age",
    "AESurv_log_h",
    "DunedinPACE",
]

DISPLAY_LABELS = {
    "AESurv_log_h": "AESurv log_h",
    "chronological_age": "Chronological age",
    "EpiClock": "EpiClock",
    "GrimAge": "GrimAge",
    "Hannum": "Hannum",
    "Horvath": "Horvath",
    "PhenoAge": "PhenoAge",
    "DunedinPACE": "DunedinPACE",
}

CATEGORY = {
    "AESurv_log_h": "aesurv",
    "EpiClock": "epiclock",
    "GrimAge": "grimage",
    "chronological_age": "age",
    "Hannum": "dnampublished",
    "Horvath": "dnampublished",
    "PhenoAge": "dnampublished",
    "DunedinPACE": "dnampublished",
}

COLORS = {
    "aesurv": "#c0392b",
    "epiclock": "#1a7f7a",
    "grimage": "#6b4c9a",
    "dnampublished": "#4a6fa5",
    "age": "#95a5a6",
    "accel": "#999999",
}

LEGEND_LABELS = {
    "aesurv": "AESurv log_h (mortality hazard)",
    "epiclock": "EpiClock (custom)",
    "grimage": "GrimAge (methylation + age + sex)",
    "dnampublished": "Published DNAm clocks",
    "age": "Chronological age",
    "accel": "Age-acceleration variants (*_accel)",
}


def load_comparison_table(json_path: Path, include_accel: bool = False) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    fhs_predictors = data["FHS"]["predictors"]
    if include_accel:
        preds = sorted(fhs_predictors.keys(), key=lambda k: float(fhs_predictors[k].get("cindex_val", 0.0)))
    else:
        preds = MAIN_PREDICTORS
    rows: List[Dict] = []
    for pred in preds:
        fhs_entry = data["FHS"]["predictors"].get(pred, {})
        whi_entry = data["WHI"]["predictors"].get(pred, {})
        fhs_ci = fhs_entry.get("cindex_val")
        whi_ci = whi_entry.get("cindex_all")
        if fhs_ci is None and whi_ci is None:
            continue
        rows.append({
            "predictor": pred,
            "label": DISPLAY_LABELS.get(pred, pred),
            "category": "accel" if pred.endswith("_accel") else CATEGORY.get(pred, "dnampublished"),
            "fhs_cindex": fhs_ci,
            "whi_cindex": whi_ci,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No main predictors found in {json_path}")
    return df.sort_values("fhs_cindex", ascending=True).reset_index(drop=True)


def plot_clock_mortality_comparison(
    json_path: Path,
    out_dir: Path,
    xlim: Tuple[float, float] | None = None,
    include_accel: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_comparison_table(json_path, include_accel=include_accel)

    csv_path = out_dir / "clock_mortality_comparison_summary.csv"
    df.to_csv(csv_path, index=False)

    fhs_n_val = json.loads(json_path.read_text(encoding="utf-8"))["FHS"].get("n_val", 482)
    whi_n = json.loads(json_path.read_text(encoding="utf-8"))["WHI"].get("n_total", 510)

    if xlim is None:
        xmin = float(min(df["fhs_cindex"].min(), df["whi_cindex"].min())) - 0.03
        xmax = float(max(df["fhs_cindex"].max(), df["whi_cindex"].max())) + 0.03
        xlim = (max(0.30, xmin), min(0.95, xmax))

    n_rows = len(df)
    fig_h = max(6.2, 0.44 * n_rows + 2.8)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, fig_h), sharey=True)
    panels = [
        ("fhs_cindex", f"FHS validation (15%, n={fhs_n_val})"),
        ("whi_cindex", f"WHI external test (n={whi_n})"),
    ]

    y_pos = range(len(df))
    labels = df["label"].tolist()

    for ax, (col, title) in zip(axes, panels):
        colors = [COLORS[c] for c in df["category"]]
        vals = df[col].to_numpy(dtype=float)
        ax.barh(list(y_pos), vals, color=colors, height=0.72, edgecolor="white", linewidth=0.6)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Harrell C-index", fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        ax.axvline(0.5, color="#dddddd", ls=":", lw=0.8, zorder=0)
        for i, v in enumerate(vals):
            if pd.notna(v):
                ax.text(
                    min(v + 0.008, xlim[1] - 0.02),
                    i,
                    f"{v:.3f}",
                    va="center",
                    ha="left",
                    fontsize=11.5,
                )
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=11)

    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(labels, fontsize=11)

    legend_order = ("grimage", "dnampublished", "epiclock", "aesurv", "age")
    if "accel" in set(df["category"]):
        legend_order = legend_order + ("accel",)
    legend_handles = [
        Patch(facecolor=COLORS[k], edgecolor="white", label=LEGEND_LABELS[k])
        for k in legend_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Mortality discrimination by predictor (Harrell C-index)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.08,
        "Higher = better ranking of mortality risk. FHS: stratified val split (seed=42). "
        "See feature_importance/bio_relevance/clocks/METHODS.md.",
        ha="center",
        fontsize=11.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])

    suffix = "_all" if include_accel else ""
    pdf_path = out_dir / f"clock_mortality_comparison{suffix}.pdf"
    png_path = out_dir / f"clock_mortality_comparison{suffix}.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        type=str,
        default="feature_importance/bio_relevance/clocks/clock_mortality_cindex.json",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="feature_importance/bio_relevance/clocks",
    )
    p.add_argument("--include-accel", action="store_true", help="Include *_accel predictors in the plot/table.")
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--x-max", type=float, default=None)
    args = p.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    xlim = None
    if args.x_min is not None and args.x_max is not None:
        xlim = (args.x_min, args.x_max)
    pdf = plot_clock_mortality_comparison(
        json_path,
        out_dir,
        xlim=xlim,
        include_accel=args.include_accel,
    )
    print(f"Wrote {pdf}")
    suffix = "_all" if args.include_accel else ""
    print(f"Wrote {out_dir / f'clock_mortality_comparison{suffix}.png'}")
    print(f"Wrote {out_dir / 'clock_mortality_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
