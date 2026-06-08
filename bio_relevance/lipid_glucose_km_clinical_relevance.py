#!/usr/bin/env python3
"""KM and clinical relevance plots for lipid/glucose phenotypes (clinical cutpoints)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


TRAITS = ["ldl", "hdl", "total_cholesterol", "triglycerides", "glucose"]
GROUP_ORDER = ["low", "medium", "high"]
GROUP_COLORS = {"low": "#2980b9", "medium": "#7f8c8d", "high": "#c0392b"}

# Font sizes (+4 pt vs prior defaults)
FS_TITLE = 13
FS_AXIS = 12
FS_TICK = 12
FS_KEY = 10.5
FS_NOTE = 10
FS_SUPTITLE = 15

# Full clinical key (mg/dL) for manuscript / JSON export
CLINICAL_KEY: Dict[str, Dict[str, str]] = {
    "glucose": {
        "low": "Normal: <100 mg/dL (fasting)",
        "medium": "Prediabetic / impaired fasting: 100–125 mg/dL",
        "high": "Diabetic range: ≥126 mg/dL",
        "unit": "mg/dL",
    },
    "triglycerides": {
        "low": "Normal: <150 mg/dL",
        "medium": "Borderline high: 150–199 mg/dL",
        "high": "High: ≥200 mg/dL",
        "unit": "mg/dL",
    },
    "hdl": {
        "low": "Low HDL (unfavourable): <40 mg/dL",
        "medium": "Intermediate: 40–59 mg/dL",
        "high": "High HDL (protective): ≥60 mg/dL",
        "unit": "mg/dL",
    },
    "total_cholesterol": {
        "low": "Desirable: <200 mg/dL",
        "medium": "Borderline high: 200–239 mg/dL",
        "high": "High: ≥240 mg/dL",
        "unit": "mg/dL",
    },
    "ldl": {
        "low": "Optimal: <100 mg/dL",
        "medium": "Near optimal / borderline: 100–129 mg/dL",
        "high": "High: ≥130 mg/dL",
        "unit": "mg/dL",
    },
}


def _tertile_groups(x: pd.Series) -> Tuple[pd.Series, Dict]:
    v = pd.to_numeric(x, errors="coerce")
    q1, q2 = np.nanquantile(v.to_numpy(dtype=float), [1 / 3, 2 / 3])
    grp = pd.Series(np.nan, index=v.index, dtype="object")
    m = np.isfinite(v.to_numpy(dtype=float))
    vv = v[m]
    grp.loc[m] = np.where(vv <= q1, "low", np.where(vv >= q2, "high", "medium"))
    return grp, {
        "kind": "tertiles",
        "q1": float(q1),
        "q2": float(q2),
        "low": f"Low tertile: ≤{q1:.1f}",
        "medium": f"Middle tertile: {q1:.1f}–{q2:.1f}",
        "high": f"High tertile: ≥{q2:.1f}",
    }


def _clinical_bounds(trait: str) -> Tuple[float, float] | None:
    if trait == "glucose":
        return 100.0, 126.0
    if trait == "triglycerides":
        return 150.0, 200.0
    if trait == "hdl":
        return 40.0, 60.0
    if trait == "total_cholesterol":
        return 200.0, 240.0
    if trait == "ldl":
        return 100.0, 130.0
    return None


def _apply_cuts(trait: str, x: pd.Series) -> Tuple[pd.Series, Dict]:
    v = pd.to_numeric(x, errors="coerce")
    bounds = _clinical_bounds(trait)
    if bounds is None:
        return _tertile_groups(v)
    lo_hi, hi_lo = bounds
    arr = v.to_numpy(dtype=float)
    m = np.isfinite(arr)
    vals = arr[m]
    grp = pd.Series(np.nan, index=v.index, dtype="object")
    grp.loc[m] = np.where(vals < lo_hi, "low", np.where(vals >= hi_lo, "high", "medium"))
    key = CLINICAL_KEY.get(trait, {})
    info = {
        "kind": "clinical",
        "unit": "mg/dL",
        "low_cut_hi": lo_hi,
        "high_cut_lo": hi_lo,
        "low": key.get("low", f"<{lo_hi}"),
        "medium": key.get("medium", f"{lo_hi}–{hi_lo - (0.01 if trait != 'glucose' else 1)}"),
        "high": key.get("high", f"≥{hi_lo}"),
    }
    return grp, info


def _compact_cut_line(trait: str) -> str:
    """One-line clinical cutpoints for subplot subtitle."""
    lines = {
        "glucose": "Low <100 | Med 100–125 | High ≥126",
        "triglycerides": "Low <150 | Med 150–199 | High ≥200",
        "hdl": "Low <40 | Med 40–59 | High ≥60",
        "total_cholesterol": "Low <200 | Med 200–239 | High ≥240",
        "ldl": "Low <100 | Med 100–129 | High ≥130",
    }
    return lines.get(trait, "")


def _legend_label(trait: str, g: str, n: int) -> str:
    """Compact KM legend: color group + n only."""
    names = {"low": "Low", "medium": "Medium", "high": "High"}
    return f"{names.get(g, g)} (n={n})"


def _category_key_labels(trait: str) -> Dict[str, str]:
    """Short in-plot key labels (mg/dL) for legend."""
    keys = {
        "glucose": {"low": "Low: <100", "medium": "Med: 100–125", "high": "High: ≥126"},
        "triglycerides": {"low": "Low: <150", "medium": "Med: 150–199", "high": "High: ≥200"},
        "hdl": {"low": "Low: <40", "medium": "Med: 40–59", "high": "High: ≥60"},
        "total_cholesterol": {"low": "Low: <200", "medium": "Med: 200–239", "high": "High: ≥240"},
        "ldl": {"low": "Low: <100", "medium": "Med: 100–129", "high": "High: ≥130"},
    }
    return keys.get(trait, {g: g.title() for g in GROUP_ORDER})


def _add_inplot_key(ax: plt.Axes, trait: str) -> None:
    """Category key inside axes, top-right corner."""
    from matplotlib.patches import Patch

    labels = _category_key_labels(trait)
    handles = [
        Patch(facecolor=GROUP_COLORS[g], edgecolor="0.3", linewidth=0.5, label=labels[g])
        for g in GROUP_ORDER
        if g in labels
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=FS_KEY,
        framealpha=0.95,
        edgecolor="0.75",
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=1.0,
        handletextpad=0.5,
    )


def _safe_logrank_p(sub: pd.DataFrame) -> float:
    try:
        from lifelines.statistics import multivariate_logrank_test

        lr = multivariate_logrank_test(
            sub["time"].to_numpy(dtype=float),
            sub["group"].astype(str).to_numpy(),
            sub["event"].to_numpy(dtype=int),
        )
        return float(lr.p_value)
    except Exception:
        return float("nan")


def _plot_km(ax: plt.Axes, sub: pd.DataFrame, trait: str, cuts: Dict) -> Dict:
    from lifelines import KaplanMeierFitter

    stats: Dict = {}
    kmf = KaplanMeierFitter()
    for g in GROUP_ORDER:
        gsub = sub[sub["group"] == g]
        if len(gsub) < 20 or gsub["event"].sum() < 5:
            continue
        kmf.fit(
            gsub["time"].to_numpy(dtype=float),
            gsub["event"].to_numpy(dtype=int),
            label=None,
        )
        kmf.plot_survival_function(ax=ax, ci_show=False, color=GROUP_COLORS[g], lw=1.8)
        stats[f"n_{g}"] = int(len(gsub))
        stats[f"events_{g}"] = int(gsub["event"].sum())
    title_trait = trait.replace("_", " ").title()
    ax.set_title(f"{title_trait} — Kaplan–Meier (mg/dL)", fontsize=FS_TITLE, pad=6)
    ax.set_xlabel("Time (years)", fontsize=FS_AXIS)
    ax.set_ylabel("Survival", fontsize=FS_AXIS)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    _add_inplot_key(ax, trait)
    ax.grid(False)
    p = _safe_logrank_p(sub)
    if np.isfinite(p):
        ax.text(
            0.03, 0.03, f"log-rank p={p:.2e}",
            transform=ax.transAxes, fontsize=FS_NOTE, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="none"),
        )
    stats["logrank_p"] = p
    return stats


def _plot_logh_box(ax: plt.Axes, sub: pd.DataFrame, trait: str, cuts: Dict) -> Dict:
    x_labels = ["Low", "Medium", "High"]
    palette = [GROUP_COLORS[g] for g in GROUP_ORDER]
    sns.boxplot(
        data=sub,
        x="group",
        y="log_h",
        order=GROUP_ORDER,
        hue="group",
        hue_order=GROUP_ORDER,
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
        fliersize=1.5,
        linewidth=0.8,
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(x_labels, fontsize=FS_TICK)
    title_trait = trait.replace("_", " ").title()
    ax.set_title(f"{title_trait}: AESurv log_h (mg/dL)", fontsize=FS_TITLE, pad=6)
    ax.set_xlabel("Clinical category", fontsize=FS_AXIS)
    ax.set_ylabel("log_h", fontsize=FS_AXIS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(False)
    means = sub.groupby("group")["log_h"].mean()
    return {
        "mean_logh_low": float(means.get("low", np.nan)),
        "mean_logh_medium": float(means.get("medium", np.nan)),
        "mean_logh_high": float(means.get("high", np.nan)),
    }


def _save_trait_panel(sub: pd.DataFrame, trait: str, cuts: Dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    _plot_km(axes[0], sub, trait, cuts)
    _plot_logh_box(axes[1], sub, trait, cuts)
    title_trait = trait.replace("_", " ").title()
    fig.suptitle(f"{title_trait} — clinical strata (FHS)", fontsize=FS_TITLE + 1, y=1.02)
    fig.tight_layout()
    path = out_dir / f"clinical_{trait}_km_logh.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figures(df: pd.DataFrame, out_dir: Path) -> Dict[str, Dict]:
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig_km, axes_km = plt.subplots(2, 3, figsize=(14, 8.5))
    axes_km = axes_km.ravel()
    fig_box, axes_box = plt.subplots(2, 3, figsize=(14, 8.5))
    axes_box = axes_box.ravel()

    summary: Dict[str, Dict] = {}
    plot_idx = 0
    for trait in TRAITS:
        if trait not in df.columns:
            continue
        sub = df[["time", "event", "log_h", trait]].copy()
        sub[trait] = pd.to_numeric(sub[trait], errors="coerce")
        sub = sub.dropna(subset=["time", "event", "log_h", trait])
        if len(sub) < 100:
            continue
        sub["group"], cuts = _apply_cuts(trait, sub[trait])
        sub = sub.dropna(subset=["group"])
        if len(sub) < 100:
            continue

        km_stats = _plot_km(axes_km[plot_idx], sub, trait, cuts)
        box_stats = _plot_logh_box(axes_box[plot_idx], sub, trait, cuts)
        summary[trait] = {"n": int(len(sub)), "categories": cuts, **km_stats, **box_stats}
        _save_trait_panel(sub, trait, cuts, out_dir)
        plot_idx += 1

    for i in range(plot_idx, len(axes_km)):
        axes_km[i].axis("off")
        axes_box[i].axis("off")

    fig_km.suptitle(
        f"Kaplan–Meier by clinical categories (FHS, mg/dL) — {stamp}",
        y=1.01,
        fontsize=FS_SUPTITLE,
    )
    fig_km.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08, hspace=0.45, wspace=0.28)
    km_pdf = out_dir / "lipid_glucose_km_clinical_grid.pdf"
    fig_km.savefig(km_pdf, dpi=150, bbox_inches="tight")
    fig_km.savefig(km_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig_km)

    fig_box.suptitle(
        f"AESurv log_h by clinical category (FHS, mg/dL) — {stamp}",
        y=1.01,
        fontsize=FS_SUPTITLE,
    )
    fig_box.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.10, hspace=0.45, wspace=0.28)
    box_pdf = out_dir / "lipid_glucose_logh_clinical_grid.pdf"
    fig_box.savefig(box_pdf, dpi=150, bbox_inches="tight")
    fig_box.savefig(box_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig_box)

    print(f"Wrote {km_pdf}")
    print(f"Wrote {box_pdf}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--merged-parquet",
        type=str,
        default="feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="feature_importance/bio_relevance/lifestyle/figures",
    )
    p.add_argument("--fhs-only", action="store_true", default=True)
    args = p.parse_args()

    df = pd.read_parquet(args.merged_parquet)
    if args.fhs_only:
        df = df[df["cohort"].astype(str) == "FHS"].copy()

    out_dir = Path(args.out_dir)
    summary = make_figures(df, out_dir)

    key_out = {
        "note": "All thresholds in mg/dL unless noted. low/medium/high map to clinical categories.",
        "traits": CLINICAL_KEY,
        "per_trait_results": summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    key_path = out_dir / "lipid_glucose_clinical_key.json"
    key_path.write_text(json.dumps(key_out, indent=2), encoding="utf-8")

    sum_path = out_dir / "lipid_glucose_km_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {key_path}")
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
