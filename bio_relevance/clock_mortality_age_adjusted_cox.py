#!/usr/bin/env python3
"""Age-adjusted Cox models: incremental mortality signal beyond chronological age.

Primary analysis: Cox model with age (+ sex) plus each predictor (z-scored) to test
whether the hazard ratio remains significant after adjusting for chronological age.
Likelihood-ratio tests compare nested models (age vs age + predictor).

Also includes focused log_h vs GrimAge model suite (M0–M6).

Outputs (default feature_importance/bio_relevance/clocks/):
  clock_cox_age_adjusted.json
  clock_cox_incremental_validity.json
  clock_cox_incremental_forest_{fhs_train,fhs_val,whi}.pdf
  clock_cox_accel_forest_{fhs_train,fhs_val,whi}.pdf
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from bio_relevance.clock_methods_manifest import get_clock_methods_manifest
from train_dann_survival import harrell_c_index
from train_vae_cox_lite import load_bundle_with_cache

try:
    from lifelines import CoxPHFitter
except ImportError:
    CoxPHFitter = None  # type: ignore


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m, s = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def _lrt(ll_small: float, ll_big: float, df: int) -> Dict[str, float]:
    stat = max(2.0 * (ll_big - ll_small), 0.0)
    p = float(stats.chi2.sf(stat, max(df, 1)))
    return {"lr_stat": stat, "p_value": p, "df": int(df)}


def _fit_cox(
    df: pd.DataFrame,
    formula: str,
    duration_col: str = "time",
    event_col: str = "event",
    penalizer: float = 0.01,
    hr_scale: float = 1.0,
) -> Optional[Dict[str, Any]]:
    if CoxPHFitter is None:
        return {"error": "lifelines not installed"}
    use = df.dropna()
    if len(use) < 40 or use[event_col].sum() < 8:
        return {"n": int(len(use)), "error": "insufficient events or sample size"}
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(use, duration_col=duration_col, event_col=event_col, formula=formula)
    except Exception as exc:
        return {"n": int(len(use)), "formula": formula, "error": str(exc)}

    lp = cph.predict_partial_hazard(use).to_numpy().ravel()
    t = use[duration_col].to_numpy(dtype=np.float64)
    e = use[event_col].to_numpy(dtype=np.int32)
    ci = harrell_c_index(t, e, lp)

    coef_out: Dict[str, Any] = {}
    for term in cph.params_.index:
        hr = float(np.exp(cph.params_[term]))
        hr_scaled = float(np.exp(cph.params_[term] * hr_scale))
        se = float(cph.standard_errors_[term]) if term in cph.standard_errors_.index else np.nan
        z = float(cph.params_[term] / se) if np.isfinite(se) and se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        coef_out[term] = {
            "coef": float(cph.params_[term]),
            "hr_per_sd": hr,
            "hr_per_scale": hr_scaled,
            "hr_scale": float(hr_scale),
            "se": se,
            "p_value": p,
        }

    return {
        "n": int(len(use)),
        "n_events": int(e.sum()),
        "formula": formula,
        "c_index": ci,
        "log_likelihood": float(cph.log_likelihood_),
        "concordance_lifelines": float(cph.concordance_index_),
        "coefficients": coef_out,
        "_model": cph,
        "_data": use,
    }


def _grim_age_accel_regression(grim: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Biolearn-style GrimAge residual after linear adjustment for chronological age."""
    m = np.isfinite(grim) & np.isfinite(age)
    out = np.full(len(grim), np.nan, dtype=np.float64)
    if m.sum() < 20:
        return out
    lr = LinearRegression().fit(age[m].reshape(-1, 1), grim[m])
    out[m] = grim[m] - lr.predict(age[m].reshape(-1, 1))
    return out


def run_cox_suite(
    time: np.ndarray,
    event: np.ndarray,
    age: np.ndarray,
    log_h: np.ndarray,
    grim: np.ndarray,
    sex: Optional[np.ndarray] = None,
    hr_scale: float = 1.0,
) -> Dict[str, Any]:
    grim_accel = _grim_age_accel_regression(grim, age)
    base = pd.DataFrame({
        "time": time,
        "event": event.astype(int),
        "age_z": _z(age),
        "log_h_z": _z(log_h),
        "GrimAge_z": _z(grim),
        "GrimAge_accel_z": _z(grim_accel),
    })
    if sex is not None:
        base["sex"] = pd.to_numeric(sex, errors="coerce")

    sex_term = " + sex" if sex is not None and base["sex"].nunique(dropna=True) > 1 else ""
    models = {
        "M0_age": f"age_z{sex_term}",
        "M1_log_h": f"log_h_z{sex_term}",
        "M2_GrimAge": f"GrimAge_z{sex_term}",
        "M3_GrimAge_accel": f"GrimAge_accel_z{sex_term}",
        "M4_age_log_h": f"age_z + log_h_z{sex_term}",
        "M5_age_GrimAge_accel": f"age_z + GrimAge_accel_z{sex_term}",
        "M6_age_GrimAge": f"age_z + GrimAge_z{sex_term}",
    }

    fitted: Dict[str, Dict] = {}
    for name, formula in models.items():
        fitted[name] = _fit_cox(base, formula, hr_scale=hr_scale)

    lrt: Dict[str, Any] = {}
    pairs = [
        ("M4_age_log_h", "M0_age", "M4_vs_M0_log_h_increment"),
        ("M5_age_GrimAge_accel", "M0_age", "M5_vs_M0_GrimAge_accel_increment"),
        ("M6_age_GrimAge", "M0_age", "M6_vs_M0_GrimAge_increment"),
        ("M4_age_log_h", "M1_log_h", "M4_vs_M1_age_adjusts_log_h"),
        ("M5_age_GrimAge_accel", "M3_GrimAge_accel", "M5_vs_M3_age_adjusts_grim_accel"),
    ]
    for big_k, small_k, lrt_name in pairs:
        b, s = fitted.get(big_k), fitted.get(small_k)
        if not b or not s or b.get("error") or s.get("error"):
            continue
        if b.get("_model") is None or s.get("_model") is None:
            continue
        df = max(len(b["_model"].params_) - len(s["_model"].params_), 1)
        lrt[lrt_name] = _lrt(s["log_likelihood"], b["log_likelihood"], df)

    # Strip non-serializable
    for v in fitted.values():
        v.pop("_model", None)
        v.pop("_data", None)

    return {
        "models": fitted,
        "likelihood_ratio_tests": lrt,
        "interpretation": {
            "M4_vs_M0": "Does log_h add mortality prediction beyond chronological age?",
            "M5_vs_M0": "Does GrimAge acceleration add beyond age?",
            "M6_vs_M0": "Does raw GrimAge add beyond age (partially collinear; GrimAge includes age at scoring)?",
        },
    }


# Predictors scored in epigenetic_clock_mortality_benchmark.py (column name in parquet).
INCREMENTAL_PREDICTORS: List[Tuple[str, str]] = [
    ("log_h", "AESurv log_h"),
    ("Horvath", "Horvath"),
    ("Hannum", "Hannum"),
    ("PhenoAge", "PhenoAge"),
    ("GrimAge", "GrimAge V2"),
    ("DunedinPACE", "DunedinPACE"),
    ("EpiClock", "EpiClock"),
]

INCREMENTAL_ACCEL_PREDICTORS: List[Tuple[str, str]] = [
    ("Horvath_accel", "Horvath acceleration"),
    ("Hannum_accel", "Hannum acceleration"),
    ("PhenoAge_accel", "PhenoAge acceleration"),
    ("GrimAge_accel", "GrimAge acceleration"),
    ("EpiClock_accel", "EpiClock acceleration"),
]

# Clocks compared against AESurv log_h in head-to-head Cox (excludes log_h itself).
CLOCK_COMPETITORS: List[Tuple[str, str]] = [
    (c, lbl) for c, lbl in INCREMENTAL_PREDICTORS if c != "log_h"
]


def fhs_train_val_indices(
    event: np.ndarray,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same 85/15 stratified split as epigenetic_clock_mortality_benchmark.py."""
    strat = event.astype(np.int32) if event.sum() >= 2 else None
    idx = np.arange(len(event))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=strat, shuffle=True,
    )
    return np.asarray(tr_idx, dtype=np.int64), np.asarray(va_idx, dtype=np.int64)


def build_accel_scores(scores: pd.DataFrame, age: np.ndarray) -> pd.DataFrame:
    """Age acceleration columns; GrimAge uses regression residual on age."""
    out: Dict[str, np.ndarray] = {}
    for col in ("Horvath_accel", "Hannum_accel", "PhenoAge_accel", "EpiClock_accel"):
        if col in scores.columns:
            out[col] = scores[col].to_numpy(dtype=np.float64)
    if "GrimAge" in scores.columns:
        out["GrimAge_accel"] = _grim_age_accel_regression(
            scores["GrimAge"].to_numpy(dtype=np.float64), age,
        )
    return pd.DataFrame(out)


def run_incremental_validity(
    time: np.ndarray,
    event: np.ndarray,
    age: np.ndarray,
    scores: pd.DataFrame,
    sex: Optional[np.ndarray] = None,
    *,
    predictor_list: Optional[List[Tuple[str, str]]] = None,
    split_label: str = "",
    hr_scale: float = 1.0,
) -> Dict[str, Any]:
    """For each predictor: Cox with age (+ sex) vs age-only; report adjusted HR and LRT."""
    base_df = pd.DataFrame({
        "time": time,
        "event": event.astype(int),
        "age_z": _z(age),
    })
    if sex is not None:
        base_df["sex"] = pd.to_numeric(sex, errors="coerce")
    sex_term = " + sex" if sex is not None and base_df["sex"].nunique(dropna=True) > 1 else ""

    m0 = _fit_cox(base_df, f"age_z{sex_term}", hr_scale=hr_scale)
    out_preds: Dict[str, Any] = {}
    pred_list = predictor_list or INCREMENTAL_PREDICTORS

    for col, label in pred_list:
        if col not in scores.columns:
            continue
        zcol = f"{col}_z"
        use = base_df.copy()
        use[zcol] = _z(scores[col].to_numpy(dtype=np.float64))
        formula = f"age_z + {zcol}{sex_term}"
        m_joint = _fit_cox(use, formula, hr_scale=hr_scale)
        lrt_entry = None
        if (
            m0
            and m_joint
            and not m0.get("error")
            and not m_joint.get("error")
            and m0.get("_model") is not None
            and m_joint.get("_model") is not None
        ):
            df = max(len(m_joint["_model"].params_) - len(m0["_model"].params_), 1)
            lrt_entry = _lrt(m0["log_likelihood"], m_joint["log_likelihood"], df)

        pred_coef = (m_joint or {}).get("coefficients", {}).get(zcol, {})
        if m_joint:
            m_joint = {k: v for k, v in m_joint.items() if not k.startswith("_")}

        out_preds[col] = {
            "label": label,
            "formula_joint": formula,
            "model_age_plus_predictor": m_joint,
            "predictor_in_joint_model": pred_coef,
            "lrt_age_vs_age_plus_predictor": lrt_entry,
        }

    m0_clean = {k: v for k, v in (m0 or {}).items() if not k.startswith("_")}
    return {
        "split_label": split_label,
        "n": int(len(base_df)),
        "n_events": int(event.sum()),
        "baseline_model": "age_z" + sex_term,
        "baseline": m0_clean,
        "predictors": out_preds,
        "note": (
            "Incremental validity: predictor HR and p are from the joint model "
            "(age + predictor + sex). LRT compares that model to age-only (M0)."
        ),
    }


def _hr_ci_from_coef(c: Dict[str, Any], hr_scale: float = 1.0) -> Tuple[float, float, float]:
    """HR and 95% CI per ``hr_scale`` SD from Cox coefficient and SE."""
    coef = float(c["coef"])
    se = float(c.get("se", np.nan))
    hr = float(np.exp(coef * hr_scale))
    if np.isfinite(se) and se > 0:
        lo = float(np.exp((coef - 1.96 * se) * hr_scale))
        hi = float(np.exp((coef + 1.96 * se) * hr_scale))
    else:
        lo, hi = hr, hr
    return hr, lo, hi


def _p_stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _forest_rows(
    incr: Dict[str, Any],
    predictor_order: Optional[List[Tuple[str, str]]] = None,
    hr_scale: float = 1.0,
    invert_effect: bool = False,
) -> List[Dict[str, Any]]:
    preds = incr.get("predictors", {})
    order_list = predictor_order or INCREMENTAL_PREDICTORS
    rows: List[Dict[str, Any]] = []
    for col, label in order_list:
        if col not in preds:
            continue
        c = preds[col].get("predictor_in_joint_model") or {}
        if c.get("coef") is None:
            continue
        hr, lo, hi = _hr_ci_from_coef(c, hr_scale=hr_scale)
        if invert_effect:
            # Inverted presentation: lower acceleration (healthier) moves right.
            hr = 1.0 / max(hr, 1e-12)
            lo_i = 1.0 / max(hi, 1e-12)
            hi_i = 1.0 / max(lo, 1e-12)
            lo, hi = lo_i, hi_i
        rows.append({
            "id": col,
            "label": label,
            "hr": hr,
            "lo": lo,
            "hi": hi,
            "p": c.get("p_value"),
        })
    return rows


def _format_p(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _log_xlim(rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    return max(0.4, lo * 0.75), min(6.5, hi * 1.25)


def _row_y(i: int, n: int) -> float:
    """Row index i (0 = first predictor) maps to y with first row at top."""
    return float(n - 1 - i)


def _nice_log_ticks(xlo: float, xhi: float) -> List[float]:
    candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    return [t for t in candidates if xlo <= t <= xhi]


def _draw_traditional_forest_on_axes(
    ax_plot,
    ax_tab,
    rows: List[Dict[str, Any]],
    xlim: Tuple[float, float],
    hr_scale: float = 1.0,
) -> None:
    """Classic forest: horizontal CI line + square HR; numeric columns in a separate axes."""
    n = len(rows)
    y_bottom, y_top = -0.6, float(n) - 0.4

    for i, r in enumerate(rows):
        yi = _row_y(i, n)
        color = "#c0392b" if r["id"] == "log_h" else "#2c3e50"
        ax_plot.hlines(yi, r["lo"], r["hi"], colors=color, linewidth=2.0, zorder=2)
        ax_plot.plot(r["hr"], yi, marker="s", markersize=7, color=color, zorder=3, linestyle="None")

    ax_plot.axvline(1.0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    ax_plot.set_xscale("log")
    ax_plot.set_xlim(xlim)
    ticks = _nice_log_ticks(xlim[0], xlim[1])
    if 1.0 not in ticks and xlim[0] <= 1.0 <= xlim[1]:
        ticks = sorted(set(ticks + [1.0]))
    if ticks:
        ax_plot.set_xticks(ticks)
        from matplotlib.ticker import FuncFormatter

        ax_plot.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_plot.set_yticks([_row_y(i, n) for i in range(n)])
    ax_plot.set_yticklabels([r["label"] for r in rows], fontsize=13)
    ax_plot.set_ylim(y_bottom, y_top)
    ax_plot.set_xlabel(f"Hazard ratio (per {hr_scale:g} SD increase)", fontsize=13)
    ax_plot.tick_params(axis="y", length=0, pad=8)
    ax_plot.tick_params(axis="x", labelsize=12)
    ax_plot.grid(True, which="major", axis="x", linestyle="-", linewidth=0.6, alpha=0.45, color="#bdbdbd")
    ax_plot.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.4, alpha=0.35, color="#d0d0d0")
    ax_plot.minorticks_on()
    ax_plot.set_axisbelow(True)

    # Numeric columns — separate axes, shared y limits, no forest ink
    ax_tab.set_xlim(0, 1)
    ax_tab.set_ylim(y_bottom, y_top)
    ax_tab.axis("off")
    col_hr, col_ci, col_p = 0.06, 0.34, 0.72
    hdr_y = y_top + 0.15
    for x, lbl in ((col_hr, "HR"), (col_ci, "95% CI"), (col_p, "P value")):
        ax_tab.text(x, hdr_y, lbl, fontsize=13, fontweight="bold", va="bottom", ha="left")
    for i, r in enumerate(rows):
        yi = _row_y(i, n)
        p_str = _format_p(r["p"]) + _p_stars(r["p"])
        ax_tab.text(col_hr, yi, f"{r['hr']:.2f}", fontsize=13, va="center", ha="left")
        ax_tab.text(col_ci, yi, f"({r['lo']:.2f}, {r['hi']:.2f})", fontsize=13, va="center", ha="left")
        ax_tab.text(col_p, yi, p_str, fontsize=13, va="center", ha="left")


def _make_traditional_forest_figure(
    rows: List[Dict[str, Any]],
    panel_title: str,
    hr_scale: float = 1.0,
    xlim: Optional[Tuple[float, float]] = None,
    fig_rect: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[Any, Any, Any]:
    """Build one traditional forest block; returns (fig, ax_plot, ax_tab) or attaches to fig_rect."""
    import matplotlib.pyplot as plt

    n = len(rows)
    xlim = xlim or _log_xlim(rows)
    if fig_rect is None:
        fig_h = max(6.0, 0.70 * n + 3.0)
        fig = plt.figure(figsize=(12.5, fig_h), facecolor="white")
        bottom, h = 0.14, 0.72
        ax_plot = fig.add_axes([0.22, bottom, 0.36, h])
        ax_tab = fig.add_axes([0.60, bottom, 0.36, h])
    else:
        fig = plt.gcf()
        left, bottom, width, height = fig_rect
        ax_plot = fig.add_axes([left, bottom, width * 0.52, height])
        ax_tab = fig.add_axes([left + width * 0.54, bottom, width * 0.44, height])

    _draw_traditional_forest_on_axes(ax_plot, ax_tab, rows, xlim, hr_scale=hr_scale)
    ax_plot.set_title(panel_title, fontsize=14, fontweight="bold", loc="left", pad=12)
    return fig, ax_plot, ax_tab


def _save_forest_fig(fig, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(fig)


def _split_display_title(split_key: str, incr: Dict[str, Any]) -> str:
    n = incr.get("n", "?")
    ev = incr.get("n_events", "?")
    labels = {
        "FHS_train": "FHS training (85%)",
        "FHS_val": "FHS validation (15%)",
        "WHI": "WHI external test",
    }
    return f"{labels.get(split_key, split_key)} — n={n}, events={ev}"


def _plot_incremental_forest(
    incr: Dict[str, Any],
    split_key: str,
    out_path: Path,
    *,
    predictor_order: Optional[List[Tuple[str, str]]] = None,
    analysis_label: str = "clock scores",
    hr_scale: float = 1.0,
    invert_effect: bool = False,
) -> None:
    rows = _forest_rows(
        incr, predictor_order, hr_scale=hr_scale, invert_effect=invert_effect,
    )
    if not rows:
        return

    use_sex = split_key.startswith("FHS")
    fig, _, _ = _make_traditional_forest_figure(
        rows, _split_display_title(split_key, incr), hr_scale=hr_scale,
    )
    sex_note = " + sex" if use_sex else ""
    fig.suptitle(
        f"Forest plot: age-adjusted mortality HR ({analysis_label})",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.04,
        f"Cox model: age (z) + predictor (z){sex_note}; HR shown per {hr_scale:g} SD"
        + ("; inverted for acceleration direction" if invert_effect else ""),
        ha="center", fontsize=12,
    )
    _save_forest_fig(fig, out_path)


def plot_all_incremental_forests(incr_all: Dict[str, Any], out_dir: Path, hr_scale: float = 1.0) -> List[Path]:
    """Write separate forest PDF/PNG per split for raw clocks and age acceleration."""
    written: List[Path] = []
    sections = [
        ("raw_predictors", "clock_cox_incremental_forest", INCREMENTAL_PREDICTORS, "clock scores", False),
        ("age_acceleration", "clock_cox_accel_forest", INCREMENTAL_ACCEL_PREDICTORS, "age acceleration", True),
    ]
    for section_key, stem, pred_order, label, invert_effect in sections:
        for split_key, incr in incr_all.get(section_key, {}).items():
            slug = split_key.lower().replace(" ", "_")
            out_path = out_dir / f"{stem}_{slug}.pdf"
            _plot_incremental_forest(
                incr, split_key, out_path,
                predictor_order=pred_order,
                analysis_label=label,
                hr_scale=hr_scale,
                invert_effect=invert_effect,
            )
            written.extend([out_path, out_path.with_suffix(".png")])
    return written


def _plot_cox_comparison(results: Dict[str, Any], cohort: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    models = results["models"]
    order = ["M0_age", "M1_log_h", "M3_GrimAge_accel", "M4_age_log_h", "M5_age_GrimAge_accel", "M2_GrimAge", "M6_age_GrimAge"]
    labels = {
        "M0_age": "Age",
        "M1_log_h": "log_h",
        "M2_GrimAge": "GrimAge",
        "M3_GrimAge_accel": "GrimAge accel",
        "M4_age_log_h": "Age + log_h",
        "M5_age_GrimAge_accel": "Age + GrimAge accel",
        "M6_age_GrimAge": "Age + GrimAge",
    }
    vals, labs = [], []
    for k in order:
        if k not in models or models[k].get("c_index") is None:
            continue
        labs.append(labels.get(k, k))
        vals.append(models[k]["c_index"])

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(labs))
    colors = ["#7f8c8d"] * len(labs)
    for i, lab in enumerate(labs):
        if "log_h" in lab and "Age" not in lab:
            colors[i] = "#c0392b"
        elif lab == "Age + log_h":
            colors[i] = "#e74c3c"
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=12)
    ax.set_ylim(0.5, 0.92)
    ax.set_ylabel("Harrell C-index (Cox linear predictor)", fontsize=13)
    ax.set_title(f"Age-adjusted mortality Cox models — {cohort}", fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forests_from_json(json_path: Path, out_dir: Optional[Path] = None) -> None:
    """Regenerate forest plots from clock_cox_incremental_validity.json."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir = out_dir or json_path.parent
    if "raw_predictors" in data or "age_acceleration" in data:
        plot_all_incremental_forests(data, out_dir)
        return
    # Legacy single-split JSON
    for cohort, incr in data.get("cohorts", {}).items():
        _plot_incremental_forest(incr, cohort, out_dir / f"clock_cox_incremental_forest_{cohort.lower()}.pdf")


def _run_incremental_on_slice(
    time: np.ndarray,
    event: np.ndarray,
    age: np.ndarray,
    scores: pd.DataFrame,
    sex: Optional[np.ndarray],
    split_label: str,
    predictor_list: List[Tuple[str, str]],
    hr_scale: float = 1.0,
) -> Dict[str, Any]:
    return run_incremental_validity(
        time, event, age, scores, sex,
        predictor_list=predictor_list,
        split_label=split_label,
        hr_scale=hr_scale,
    )


def _cox_strip(model: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not model:
        return model
    return {k: v for k, v in model.items() if not str(k).startswith("_")}


def _lrt_nested(big: Dict[str, Any], small: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if big.get("error") or small.get("error"):
        return None
    if big.get("_model") is None or small.get("_model") is None:
        return None
    df = max(len(big["_model"].params_) - len(small["_model"].params_), 1)
    return _lrt(small["log_likelihood"], big["log_likelihood"], df)


def run_log_h_beyond_each_clock(
    time: np.ndarray,
    event: np.ndarray,
    age: np.ndarray,
    scores: pd.DataFrame,
    sex: Optional[np.ndarray] = None,
    *,
    split_label: str = "",
    hr_scale: float = 1.0,
) -> Dict[str, Any]:
    """For each clock: Cox age + clock + log_h; LRT vs age + clock and vs age + log_h."""
    if "log_h" not in scores.columns:
        return {"split_label": split_label, "error": "log_h missing from scores", "clocks": {}}

    base_df = pd.DataFrame({
        "time": time,
        "event": event.astype(int),
        "age_z": _z(age),
        "log_h_z": _z(scores["log_h"].to_numpy(dtype=np.float64)),
    })
    if sex is not None:
        base_df["sex"] = pd.to_numeric(sex, errors="coerce")
    sex_term = " + sex" if sex is not None and base_df["sex"].nunique(dropna=True) > 1 else ""

    clocks_out: Dict[str, Any] = {}
    for col, label in CLOCK_COMPETITORS:
        if col not in scores.columns:
            continue
        clock_z = f"{col}_z"
        use = base_df.copy()
        use[clock_z] = _z(scores[col].to_numpy(dtype=np.float64))

        m_clock = _fit_cox(use, f"age_z + {clock_z}{sex_term}", hr_scale=hr_scale)
        m_logh = _fit_cox(use, f"age_z + log_h_z{sex_term}", hr_scale=hr_scale)
        m_both = _fit_cox(use, f"age_z + {clock_z} + log_h_z{sex_term}", hr_scale=hr_scale)

        lrt_logh = _lrt_nested(m_both, m_clock) if m_both and m_clock else None
        lrt_clock = _lrt_nested(m_both, m_logh) if m_both and m_logh else None

        for m in (m_clock, m_logh, m_both):
            if m:
                m.pop("_model", None)
                m.pop("_data", None)

        joint = m_both or {}
        coef = joint.get("coefficients") or {}
        clocks_out[col] = {
            "clock_label": label,
            "formulas": {
                "age_clock": f"age_z + {clock_z}{sex_term}",
                "age_log_h": f"age_z + log_h_z{sex_term}",
                "age_clock_log_h": f"age_z + {clock_z} + log_h_z{sex_term}",
            },
            "model_age_clock": _cox_strip(m_clock),
            "model_age_log_h": _cox_strip(m_logh),
            "model_age_clock_log_h": _cox_strip(m_both),
            "log_h_adjusted_for_age_and_clock": coef.get("log_h_z"),
            "clock_adjusted_for_age_and_log_h": coef.get(clock_z),
            "lrt_log_h_beyond_age_and_clock": lrt_logh,
            "lrt_clock_beyond_age_and_log_h": lrt_clock,
            "interpretation": {
                "lrt_log_h": "Does log_h add beyond age + this clock?",
                "lrt_clock": "Does this clock add beyond age + log_h?",
            },
        }

    return {
        "split_label": split_label,
        "n": int(len(base_df)),
        "n_events": int(event.sum()),
        "clocks": clocks_out,
        "note": "All predictors z-scored within the split. Joint model: age + clock + log_h (+ sex in FHS).",
    }


def _logh_beyond_clock_forest_rows(block: Dict[str, Any], hr_scale: float = 1.0) -> List[Dict[str, Any]]:
    """One row per clock: log_h HR from joint model age + clock + log_h."""
    rows: List[Dict[str, Any]] = []
    for col, label in CLOCK_COMPETITORS:
        cblock = block.get("clocks", {}).get(col)
        if not cblock:
            continue
        coef = cblock.get("log_h_adjusted_for_age_and_clock") or {}
        if coef.get("coef") is None:
            continue
        hr, lo, hi = _hr_ci_from_coef(coef, hr_scale=hr_scale)
        lrt = (cblock.get("lrt_log_h_beyond_age_and_clock") or {}).get("p_value")
        rows.append({
            "id": col,
            "label": f"vs {label}",
            "hr": hr,
            "lo": lo,
            "hi": hi,
            "p": coef.get("p_value"),
            "lrt_p": lrt,
        })
    return rows


def _plot_logh_beyond_clocks_forest(
    block: Dict[str, Any], split_key: str, out_path: Path, hr_scale: float = 1.0,
) -> None:
    rows = _logh_beyond_clock_forest_rows(block, hr_scale=hr_scale)
    if not rows:
        return

    fig, _, _ = _make_traditional_forest_figure(
        rows,
        _split_display_title(split_key, block),
        hr_scale=hr_scale,
    )
    fig.suptitle(
        "AESurv log_h: HR adjusted for age, sex (FHS), and each clock",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.04,
        f"Joint Cox: age + clock + log_h. Rows show log_h HR per {hr_scale:g} SD vs each clock.",
        ha="center", fontsize=12,
    )
    _save_forest_fig(fig, out_path)


def plot_logh_beyond_clocks_forests(h2h: Dict[str, Any], out_dir: Path, hr_scale: float = 1.0) -> None:
    for split_key, block in h2h.get("splits", {}).items():
        slug = split_key.lower()
        _plot_logh_beyond_clocks_forest(
            block, split_key, out_dir / f"clock_cox_logh_beyond_clocks_{slug}.pdf", hr_scale=hr_scale,
        )


def run_head_to_head_pipeline(
    scores_dir: Path,
    out_dir: Path,
    fhs_npz: str,
    fhs_pq: str,
    whi_npz: str,
    whi_pq: str,
    val_frac: float,
    seed: int,
    hr_scale: float = 1.0,
) -> Dict[str, Any]:
    h2h: Dict[str, Any] = {
        "split": {"val_frac": val_frac, "seed": seed},
        "splits": {},
        "competitors": [c for c, _ in CLOCK_COMPETITORS],
    }

    scores_fhs = pd.read_parquet(scores_dir / "clock_scores_fhs.parquet")
    _, _, t_fhs, e_fhs, _, _ = load_bundle_with_cache(
        "FHS", Path(fhs_npz), Path(fhs_pq), None, None, None, Path("vae_cox_cache"), True,
    )
    try:
        from bio_relevance.epigenetic_clock_mortality_benchmark import _load_cohort_meta

        _, sex_fhs = _load_cohort_meta(Path(fhs_pq))
    except Exception:
        sex_fhs = None

    age_fhs = scores_fhs["age"].to_numpy()
    tr_idx, va_idx = fhs_train_val_indices(e_fhs, val_frac, seed)

    for split_key, idx in (("FHS_train", tr_idx), ("FHS_val", va_idx)):
        sex_s = sex_fhs[idx] if sex_fhs is not None else None
        h2h["splits"][split_key] = run_log_h_beyond_each_clock(
            t_fhs[idx], e_fhs[idx], age_fhs[idx], scores_fhs.iloc[idx], sex_s,
            split_label=split_key,
            hr_scale=hr_scale,
        )

    scores_whi = pd.read_parquet(scores_dir / "clock_scores_whi.parquet")
    _, _, t_whi, e_whi, _, _ = load_bundle_with_cache(
        "WHI", Path(whi_npz), Path(whi_pq), None, None, None, Path("vae_cox_cache"), True,
    )
    h2h["splits"]["WHI"] = run_log_h_beyond_each_clock(
        t_whi, e_whi, scores_whi["age"].to_numpy(), scores_whi, None, split_label="WHI", hr_scale=hr_scale,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clock_cox_logh_beyond_clocks.json").write_text(
        json.dumps(h2h, indent=2, default=str), encoding="utf-8",
    )
    plot_logh_beyond_clocks_forests(h2h, out_dir, hr_scale=hr_scale)
    return h2h


def _head_to_head_summary(h2h: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for split_key, block in h2h.get("splits", {}).items():
        summary[split_key] = {}
        for col, cblock in block.get("clocks", {}).items():
            lh = cblock.get("log_h_adjusted_for_age_and_clock") or {}
            lrt = cblock.get("lrt_log_h_beyond_age_and_clock") or {}
            clk = cblock.get("clock_adjusted_for_age_and_log_h") or {}
            lrt_c = cblock.get("lrt_clock_beyond_age_and_log_h") or {}
            summary[split_key][col] = {
                "log_h_hr": lh.get("hr_per_scale", lh.get("hr_per_sd")),
                "log_h_p": lh.get("p_value"),
                "log_h_lrt_p_vs_clock_only": lrt.get("p_value"),
                "clock_hr_in_joint": clk.get("hr_per_scale", clk.get("hr_per_sd")),
                "clock_p_in_joint": clk.get("p_value"),
                "clock_lrt_p_vs_logh_only": lrt_c.get("p_value"),
            }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", type=str, default="feature_importance/bio_relevance/clocks")
    p.add_argument("--fhs-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--fhs-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--whi-pq", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hr-scale", type=float, default=1.0, help="Report/plot HR per N SD (default: 1.0)")
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Only redraw forest plots from existing JSON outputs",
    )
    p.add_argument(
        "--head-to-head-only",
        action="store_true",
        help="Only run age + clock + log_h Cox for each clock (3 splits)",
    )
    args = p.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir) if args.out_dir else scores_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        json_path = out_dir / "clock_cox_incremental_validity.json"
        if json_path.exists():
            plot_forests_from_json(json_path, out_dir)
        h2h_path = out_dir / "clock_cox_logh_beyond_clocks.json"
        if h2h_path.exists():
            plot_logh_beyond_clocks_forests(
                json.loads(h2h_path.read_text(encoding="utf-8")), out_dir, hr_scale=args.hr_scale,
            )
        if not json_path.exists() and not h2h_path.exists():
            raise SystemExit("No JSON found; run full analysis first.")
        print(f"Forest plots written to {out_dir}")
        return

    if args.head_to_head_only:
        h2h = run_head_to_head_pipeline(
            scores_dir, out_dir, args.fhs_npz, args.fhs_pq, args.whi_npz, args.whi_pq,
            args.val_frac, args.seed, hr_scale=args.hr_scale,
        )
        print(json.dumps(_head_to_head_summary(h2h), indent=2))
        print(f"Wrote {out_dir / 'clock_cox_logh_beyond_clocks.json'}")
        return

    out: Dict[str, Any] = {"cohorts": {}, "methods": get_clock_methods_manifest()}
    incr_all: Dict[str, Any] = {
        "split": {"val_frac": args.val_frac, "seed": args.seed, "fhs_splits": ["FHS_train", "FHS_val", "WHI"]},
        "raw_predictors": {},
        "age_acceleration": {},
        "grim_age_accel_note": "GrimAge_accel = regression residual of GrimAge on age; other clocks = clock − age",
    }

    # --- FHS: train / val ---
    scores_fhs = pd.read_parquet(scores_dir / "clock_scores_fhs.parquet")
    _, _, t_fhs, e_fhs, _, _ = load_bundle_with_cache(
        "FHS", Path(args.fhs_npz), Path(args.fhs_pq), None, None, None, Path("vae_cox_cache"), True,
    )
    try:
        from bio_relevance.epigenetic_clock_mortality_benchmark import _load_cohort_meta

        _, sex_fhs = _load_cohort_meta(Path(args.fhs_pq))
    except Exception:
        sex_fhs = None

    age_fhs = scores_fhs["age"].to_numpy()
    tr_idx, va_idx = fhs_train_val_indices(e_fhs, args.val_frac, args.seed)
    accel_fhs = build_accel_scores(scores_fhs, age_fhs)

    for split_key, idx in (("FHS_train", tr_idx), ("FHS_val", va_idx)):
        sex_s = sex_fhs[idx] if sex_fhs is not None else None
        incr_all["raw_predictors"][split_key] = _run_incremental_on_slice(
            t_fhs[idx], e_fhs[idx], age_fhs[idx], scores_fhs.iloc[idx], sex_s,
            split_key, INCREMENTAL_PREDICTORS, hr_scale=args.hr_scale,
        )
        incr_all["age_acceleration"][split_key] = _run_incremental_on_slice(
            t_fhs[idx], e_fhs[idx], age_fhs[idx], accel_fhs.iloc[idx], sex_s,
            split_key, INCREMENTAL_ACCEL_PREDICTORS, hr_scale=args.hr_scale,
        )

    log_h = scores_fhs["log_h"].to_numpy()
    grim = scores_fhs["GrimAge"].to_numpy()
    res_fhs = run_cox_suite(t_fhs, e_fhs, age_fhs, log_h, grim, sex_fhs, hr_scale=args.hr_scale)
    out["cohorts"]["FHS_full"] = res_fhs
    _plot_cox_comparison(res_fhs, "FHS", out_dir / "clock_cox_age_adjusted_fhs.pdf")

    # --- WHI: full cohort ---
    scores_whi = pd.read_parquet(scores_dir / "clock_scores_whi.parquet")
    _, _, t_whi, e_whi, _, _ = load_bundle_with_cache(
        "WHI", Path(args.whi_npz), Path(args.whi_pq), None, None, None, Path("vae_cox_cache"), True,
    )
    sex_whi = None
    age_whi = scores_whi["age"].to_numpy()
    accel_whi = build_accel_scores(scores_whi, age_whi)

    incr_all["raw_predictors"]["WHI"] = _run_incremental_on_slice(
        t_whi, e_whi, age_whi, scores_whi, sex_whi, "WHI", INCREMENTAL_PREDICTORS, hr_scale=args.hr_scale,
    )
    incr_all["age_acceleration"]["WHI"] = _run_incremental_on_slice(
        t_whi, e_whi, age_whi, accel_whi, sex_whi, "WHI", INCREMENTAL_ACCEL_PREDICTORS, hr_scale=args.hr_scale,
    )

    log_h_w = scores_whi["log_h"].to_numpy()
    grim_w = scores_whi["GrimAge"].to_numpy()
    res_whi = run_cox_suite(t_whi, e_whi, age_whi, log_h_w, grim_w, sex_whi, hr_scale=args.hr_scale)
    out["cohorts"]["WHI"] = res_whi
    _plot_cox_comparison(res_whi, "WHI", out_dir / "clock_cox_age_adjusted_whi.pdf")

    plot_all_incremental_forests(incr_all, out_dir, hr_scale=args.hr_scale)

    h2h = run_head_to_head_pipeline(
        scores_dir, out_dir, args.fhs_npz, args.fhs_pq, args.whi_npz, args.whi_pq,
        args.val_frac, args.seed, hr_scale=args.hr_scale,
    )

    (out_dir / "clock_cox_age_adjusted.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (out_dir / "clock_cox_incremental_validity.json").write_text(
        json.dumps(incr_all, indent=2, default=str), encoding="utf-8"
    )

    summary: Dict[str, Any] = {}
    for section in ("raw_predictors", "age_acceleration"):
        summary[section] = {}
        for split_key, block in incr_all.get(section, {}).items():
            summary[section][split_key] = {}
            for pred, pred_block in block.get("predictors", {}).items():
                pc = pred_block.get("predictor_in_joint_model") or {}
                lrt = pred_block.get("lrt_age_vs_age_plus_predictor") or {}
                summary[section][split_key][pred] = {
                    "hr_per_scaled_sd": pc.get("hr_per_scale", pc.get("hr_per_sd")),
                    "hr_scale": args.hr_scale,
                    "p_value": pc.get("p_value"),
                    "lrt_p": lrt.get("p_value"),
                }
    print(json.dumps(summary, indent=2))
    print("\n--- log_h beyond each clock (joint model) ---")
    print(json.dumps(_head_to_head_summary(h2h), indent=2))
    print(f"Wrote {out_dir / 'clock_cox_age_adjusted.json'}")
    print(f"Wrote {out_dir / 'clock_cox_incremental_validity.json'}")
    print(f"Wrote {out_dir / 'clock_cox_logh_beyond_clocks.json'}")
    for path in sorted(out_dir.glob("clock_cox_*forest*.pdf")):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
