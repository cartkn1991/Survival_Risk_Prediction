#!/usr/bin/env python3
"""Lifestyle concordance and incremental prognostic value for DANN-Aux log_h.

Reads lifestyle_risk_merged.parquet; writes summary JSON + supplementary figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from bio_relevance.lifestyle_common import bh_fdr, partial_spearman, smoking_ordinal
from train_dann_survival import harrell_c_index

sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)

LIFESTYLE_TRAITS = [
    "smoking_ord",
    "cigarettes_per_day",
    "pack_years",
    "alcohol_drinks_per_week",
    "alcohol_amount_per_occasion",
    "sleep_hours",
    "sleep_category_ord",
    "physical_activity",
    "bmi",
    "unhealthy_lifestyle_score",
]

DEFAULT_LIFESTYLE_COX_TERMS = [
    "cigarettes_per_day",
    "alcohol_drinks_per_week",
    "alcohol_amount_per_occasion",
    "sleep_hours",
    "smoking_ord",
    "pack_years",
    "physical_activity",
    "bmi",
]

# Human-readable labels for plot B (deduplicated; prefer occasion alcohol over drinks/week)
TRAIT_LABELS: Dict[str, str] = {
    "cigarettes_per_day": "Cigarettes per day",
    "smoking_ord": "Smoking status (0=never, 1=former, 2=current)",
    "alcohol_amount_per_occasion": "Alcohol (drinks per occasion)",
    "alcohol_drinks_per_week": "Alcohol (drinks per week)",
    "sleep_hours": "Sleep duration (hours, continuous)",
    "sleep_category_ord": "Sleep category (<6 / 6–8 / >8 h)",
    "unhealthy_lifestyle_score": "Composite unhealthy lifestyle score*",
}

PLOT_B_TRAIT_ORDER = [
    "cigarettes_per_day",
    "smoking_ord",
    "alcohol_amount_per_occasion",
    "sleep_category_ord",
    "unhealthy_lifestyle_score",
]

UNHEALTHY_SCORE_FOOTNOTE = (
    "*Composite score = z(smoking ordinal) + z(alcohol per occasion) + z(sleep category),\n"
    "sleep category: 0 = adequate (6–8 h), 1 = short (<6 h), 2 = long (>8 h). Each term z-scored in FHS;\n"
    "higher score = more unfavourable lifestyle profile. Equal weights (exploratory index)."
)


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m, sd = x.mean(), x.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return x * 0.0
    return (x - m) / sd


def sleep_category_label(hours: pd.Series) -> pd.Series:
    """Standard 3-level sleep duration (hours): short <6, adequate 6–8, long >8."""
    h = pd.to_numeric(hours, errors="coerce")
    out = pd.Series(np.nan, index=h.index, dtype="object")
    m = np.isfinite(h.to_numpy(dtype=float))
    vals = h[m]
    out.loc[m] = np.where(
        vals < 6, "short (<6h)",
        np.where(vals <= 8, "adequate (6–8h)", "long (>8h)"),
    )
    return out


def sleep_category_ord(hours: pd.Series) -> pd.Series:
    """Ordinal for risk composite: 0=adequate (6–8h), 1=short (<6h), 2=long (>8h)."""
    h = pd.to_numeric(hours, errors="coerce")
    out = pd.Series(np.nan, index=h.index, dtype=float)
    m = np.isfinite(h.to_numpy(dtype=float))
    vals = h[m]
    out.loc[m] = np.where(vals < 6, 1.0, np.where(vals <= 8, 0.0, 2.0))
    return out


def add_derived(df: pd.DataFrame, short_sleep: float = 6.0) -> pd.DataFrame:
    """Derived lifestyle fields; sleep uses standard <6 / 6–8 / >8 h categories."""
    del short_sleep  # kept for CLI compatibility; bounds are fixed at 6 and 8 h
    out = df.copy()
    if "smoking_status" in out.columns:
        out["smoking_ord"] = smoking_ordinal(out["smoking_status"])
    else:
        out["smoking_ord"] = np.nan
    if "sleep_hours" in out.columns:
        out["sleep_category"] = sleep_category_label(out["sleep_hours"])
        out["sleep_category_ord"] = sleep_category_ord(out["sleep_hours"])
    parts = []
    if "smoking_ord" in out.columns:
        parts.append(_zscore(out["smoking_ord"]))
    if "alcohol_amount_per_occasion" in out.columns:
        parts.append(_zscore(out["alcohol_amount_per_occasion"]))
    elif "alcohol_drinks_per_week" in out.columns:
        parts.append(_zscore(out["alcohol_drinks_per_week"]))
    if "sleep_category_ord" in out.columns:
        parts.append(_zscore(out["sleep_category_ord"]))
    if parts:
        out["unhealthy_lifestyle_score"] = sum(parts)
    else:
        out["unhealthy_lifestyle_score"] = np.nan
    return out


def build_covariate_matrix(df: pd.DataFrame, include_batch: bool = True) -> np.ndarray:
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype=np.float64)
    sex = pd.to_numeric(df["sex"], errors="coerce").to_numpy(dtype=np.float64)
    cols = [age, sex]
    if include_batch and "batch" in df.columns and df["cohort"].eq("FHS").any():
        b = df["batch"].astype(str)
        dummies = pd.get_dummies(b, prefix="batch", drop_first=True)
        for c in dummies.columns:
            cols.append(dummies[c].to_numpy(dtype=np.float64))
    return np.column_stack(cols)


def concordance_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"traits": {}, "by_cohort": {}}
    y = df["log_h"].to_numpy(dtype=np.float64)
    cov = build_covariate_matrix(df, include_batch=True)

    p_list, names = [], []
    for trait in LIFESTYLE_TRAITS:
        if trait not in df.columns:
            continue
        x = pd.to_numeric(df[trait], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 20:
            continue
        rho, p = stats.spearmanr(y[mask], x[mask])
        pr, pp = partial_spearman(y, x, cov)
        out["traits"][trait] = {
            "n": int(mask.sum()),
            "spearman_r": float(rho),
            "spearman_p": float(p),
            "partial_r": float(pr),
            "partial_p": float(pp),
        }
        p_list.append(pp if np.isfinite(pp) else p)
        names.append(trait)

    if p_list:
        q = bh_fdr(p_list)
        for trait, qv in zip(names, q):
            out["traits"][trait]["partial_q_bh"] = float(qv)

    for cohort in df["cohort"].unique():
        sub = df[df["cohort"] == cohort]
        cov_c = build_covariate_matrix(sub, include_batch=(cohort == "FHS"))
        out["by_cohort"][cohort] = {}
        for trait in LIFESTYLE_TRAITS:
            if trait not in sub.columns:
                continue
            x = pd.to_numeric(sub[trait], errors="coerce").to_numpy(dtype=np.float64)
            yy = sub["log_h"].to_numpy(dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(yy)
            if mask.sum() < 15:
                continue
            rho, p = stats.spearmanr(yy[mask], x[mask])
            pr, pp = partial_spearman(yy, x, cov_c)
            out["by_cohort"][cohort][trait] = {
                "spearman_r": float(rho), "partial_r": float(pr), "partial_p": float(pp), "n": int(mask.sum()),
            }
    return out


def cox_models(df: pd.DataFrame) -> Dict[str, Any]:
    """M0–M3 per cohort; lifestyle block = available numeric lifestyle traits."""
    lifestyle_terms = []
    for c in DEFAULT_LIFESTYLE_COX_TERMS:
        if c in df.columns and df[c].notna().sum() >= 30:
            lifestyle_terms.append(c)
    # avoid duplicate alcohol columns in one model
    if "alcohol_amount_per_occasion" in lifestyle_terms and "alcohol_drinks_per_week" in lifestyle_terms:
        lifestyle_terms.remove("alcohol_drinks_per_week")
    life_formula = " + ".join(lifestyle_terms) if lifestyle_terms else None

    results: Dict[str, Any] = {"lifestyle_terms": lifestyle_terms, "cohorts": {}}
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        results["error"] = "lifelines not installed"
        return results

    for cohort in df["cohort"].unique():
        sub = add_derived(df[df["cohort"] == cohort].copy())
        sex_var = pd.to_numeric(sub["sex"], errors="coerce").nunique(dropna=True) > 1 if "sex" in sub.columns else False
        base = "age + sex" if sex_var else "age"
        if cohort == "FHS" and "batch" in sub.columns and sub["batch"].nunique() > 1:
            # encode batch as numeric codes for formula
            sub["batch_code"] = pd.Categorical(sub["batch"]).codes
            base = "age + sex + batch_code"

        models = {"M0": base}
        if life_formula:
            models["M1"] = f"{base} + {life_formula}"
        models["M2"] = f"{base} + log_h"
        if life_formula:
            models["M3"] = f"{base} + {life_formula} + log_h"

        cohort_res = {}
        fitted = {}
        for name, formula in models.items():
            use_cols = ["time", "event", "log_h", "age", "sex"]
            if "batch_code" in sub.columns:
                use_cols.append("batch_code")
            for t in lifestyle_terms:
                if t not in use_cols:
                    use_cols.append(t)
            use = sub[use_cols].dropna()
            if len(use) < 40 or use["event"].sum() < 8:
                cohort_res[name] = {"c_index": None, "n": int(len(use))}
                continue
            cph = CoxPHFitter(penalizer=0.01)
            try:
                cph.fit(use, duration_col="time", event_col="event", formula=formula)
                lp = cph.predict_partial_hazard(use).to_numpy().ravel()
                ci = harrell_c_index(use["time"].to_numpy(), use["event"].to_numpy(), lp)
                cohort_res[name] = {"c_index": ci, "n": int(len(use)), "formula": formula}
                fitted[name] = (cph, use)
            except Exception as exc:
                cohort_res[name] = {"c_index": None, "error": str(exc)}

        # LRT M2 vs M3, M1 vs M3
        lrt = {}
        if "M2" in fitted and "M3" in fitted:
            lrt["M3_vs_M2"] = _lrt(fitted["M2"], fitted["M3"])
        if "M1" in fitted and "M3" in fitted:
            lrt["M3_vs_M1"] = _lrt(fitted["M1"], fitted["M3"])
        cohort_res["lrt"] = lrt
        results["cohorts"][cohort] = cohort_res
    return results


def _lrt(small: Tuple, big: Tuple) -> Dict[str, float]:
    cph_s, use = small
    cph_b, _ = big
    try:
        lr = cph_b.log_likelihood_ratio_test()
        return {"stat": float(lr.test_statistic), "p": float(lr.p_value)}
    except Exception:
        ll_s = cph_s.log_likelihood_
        ll_b = cph_b.log_likelihood_
        k_s = len(cph_s.params_)
        k_b = len(cph_b.params_)
        stat = 2 * (ll_b - ll_s)
        df = max(k_b - k_s, 1)
        p = float(stats.chi2.sf(stat, df))
        return {"stat": float(stat), "p": p, "df": int(df)}


def residual_risk_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    sub = add_derived(df.copy())
    y = sub["log_h"].to_numpy(dtype=np.float64)
    X_parts = [np.ones(len(sub))]
    for c in ["age", "sex"]:
        X_parts.append(pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=np.float64))
    if "batch" in sub.columns:
        dum = pd.get_dummies(sub["batch"].astype(str), drop_first=True)
        for c in dum.columns:
            X_parts.append(dum[c].to_numpy(dtype=np.float64))
    for c in ["smoking_ord", "alcohol_drinks_per_week", "sleep_hours", "pack_years"]:
        if c in sub.columns:
            X_parts.append(pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=np.float64))
    X = np.column_stack(X_parts)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[mask] = y[mask] - X[mask] @ coef
    sub["log_h_resid"] = resid

    out["n_used"] = int(mask.sum())
    batch_labels = []
    if "batch" in sub.columns:
        batch_labels = list(pd.get_dummies(sub["batch"].astype(str), drop_first=True).columns)
    out["coef_labels"] = ["intercept", "age", "sex"] + batch_labels + [
        c for c in ["smoking_ord", "alcohol_drinks_per_week", "sleep_hours", "pack_years"] if c in sub.columns
    ]

    # Cox on residual + KM tertiles by cohort
    km = {}
    cox_resid = {}
    for cohort in sub["cohort"].unique():
        csub = sub[(sub["cohort"] == cohort) & np.isfinite(sub["log_h_resid"])].copy()
        if len(csub) < 40:
            continue
        t1, t2 = np.quantile(csub["log_h_resid"], [1 / 3, 2 / 3])
        csub["resid_tertile"] = np.where(
            csub["log_h_resid"] <= t1, "low",
            np.where(csub["log_h_resid"] >= t2, "high", "mid"),
        )
        km[cohort] = {
            "cut_lo": float(t1), "cut_hi": float(t2),
            "n_low": int((csub["resid_tertile"] == "low").sum()),
            "n_high": int((csub["resid_tertile"] == "high").sum()),
        }
        try:
            from lifelines import CoxPHFitter
            use = csub[["time", "event", "log_h_resid", "age", "sex"]].dropna()
            if use["event"].sum() >= 5:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(use, duration_col="time", event_col="event", formula="age + sex + log_h_resid")
                s = csub["log_h_resid"].to_numpy()
                cox_resid[cohort] = {
                    "hr_per_sd": float(np.exp(cph.params_["log_h_resid"])),
                    "p": float(cph.summary.loc["log_h_resid", "p"]),
                    "c_index_resid_only": harrell_c_index(
                        use["time"].to_numpy(), use["event"].to_numpy(), use["log_h_resid"].to_numpy()
                    ),
                }
        except Exception as exc:
            cox_resid[cohort] = {"error": str(exc)}

    out["km"] = km
    out["cox"] = cox_resid
    out["_sub_with_resid"] = sub
    return out


def never_smoker_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    if "smoking_status" not in df.columns:
        return {"n": 0, "note": "smoking_status not available"}
    ns = df[df["smoking_status"].astype(str).str.lower() == "never"].copy()
    if len(ns) < 30:
        return {"n": int(len(ns)), "note": "too few never-smokers"}
    y = ns["log_h"].to_numpy(dtype=np.float64)
    t = ns["time"].to_numpy(dtype=np.float64)
    e = ns["event"].to_numpy(dtype=np.int64)
    ci = harrell_c_index(t, e, y)
    rho_alc = float("nan")
    if "alcohol_drinks_per_week" in ns.columns:
        x = pd.to_numeric(ns["alcohol_drinks_per_week"], errors="coerce")
        m = x.notna() & np.isfinite(y)
        if m.sum() >= 15:
            rho_alc, _ = stats.spearmanr(y[m], x[m])
    try:
        from lifelines import CoxPHFitter
        use = ns[["time", "event", "log_h", "age", "sex"]].dropna()
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(use, duration_col="time", event_col="event", formula="age + sex + log_h")
        hr = float(np.exp(cph.params_["log_h"]))
        p = float(cph.summary.loc["log_h", "p"])
    except Exception:
        hr, p = float("nan"), float("nan")
    return {"n": int(len(ns)), "c_index_log_h": ci, "spearman_alcohol": rho_alc, "cox_hr_log_h": hr, "cox_p_log_h": p}


def noise_check(df: pd.DataFrame, seed: int = 7) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(df))
    y = df["log_h"].to_numpy(dtype=np.float64)
    r_noise, _ = stats.pearsonr(y, noise)
    traits = [c for c in LIFESTYLE_TRAITS if c in df.columns]
    r_life = []
    for c in traits:
        x = pd.to_numeric(df[c], errors="coerce")
        m = x.notna() & np.isfinite(y)
        if m.sum() >= 20:
            r, _ = stats.pearsonr(y[m], x[m])
            r_life.append(abs(r))
    return {
        "r_logh_vs_noise": float(r_noise),
        "max_abs_r_logh_lifestyle": float(max(r_life)) if r_life else float("nan"),
    }


def _traits_for_plot_b(concord: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, float], bool]]:
    """Return (trait_key, display_label, stats, is_composite) in display order."""
    traits = concord.get("traits", {})
    out: List[Tuple[str, str, Dict[str, float], bool]] = []
    seen_alcohol = False
    for key in PLOT_B_TRAIT_ORDER:
        if key not in traits:
            if key == "alcohol_amount_per_occasion" and "alcohol_drinks_per_week" in traits:
                key = "alcohol_drinks_per_week"
            else:
                continue
        if key.startswith("alcohol"):
            if seen_alcohol:
                continue
            seen_alcohol = True
        label = TRAIT_LABELS.get(key, key)
        is_comp = key == "unhealthy_lifestyle_score"
        out.append((key, label, traits[key], is_comp))
    return out


def draw_plot_b_panel(
    ax: plt.Axes,
    concord: Dict[str, Any],
    *,
    title_prefix: str = "B",
    show_footnote: bool = False,
) -> None:
    """Partial Spearman r: lifestyle traits vs AESurv log_h (adj. age, sex, batch)."""
    rows = _traits_for_plot_b(concord)
    if not rows:
        ax.text(0.5, 0.5, "No concordance results", ha="center", transform=ax.transAxes)
        return

    labels = [r[1] for r in rows]
    vals = [r[2].get("partial_r", np.nan) for r in rows]
    pvals = [r[2].get("partial_p", np.nan) for r in rows]
    is_comp = [r[3] for r in rows]

    colors = []
    for v, comp in zip(vals, is_comp):
        if comp:
            colors.append("#8e44ad")
        elif v > 0:
            colors.append("#c0392b")
        else:
            colors.append("#2980b9")

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors, edgecolor="0.25", linewidth=0.5, height=0.72)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Partial Spearman r with AESurv log_h\n(adjusted for age, sex, FHS batch)", fontsize=9)
    ax.set_title(
        f"{title_prefix}  Lifestyle association with model risk score (FHS, n≈3,138)",
        fontsize=10,
        loc="left",
    )
    ax.grid(False)

    # Annotate r and p at bar ends
    xlim = max(0.08, np.nanmax(np.abs(vals)) * 1.35 + 0.02)
    ax.set_xlim(-xlim, xlim)
    for i, (v, p, comp) in enumerate(zip(vals, pvals, is_comp)):
        if not np.isfinite(v):
            continue
        ha = "left" if v >= 0 else "right"
        dx = 0.008 if v >= 0 else -0.008
        p_str = f"p={p:.3g}" if np.isfinite(p) else "p=—"
        sig = "*" if np.isfinite(p) and p < 0.05 else ""
        ax.text(
            v + dx, i, f"r={v:+.3f}, {p_str}{sig}",
            va="center", ha=ha, fontsize=7.5,
            fontweight="bold" if comp else "normal",
        )

    # Separator before composite
    if any(is_comp):
        comp_idx = next(i for i, c in enumerate(is_comp) if c)
        if comp_idx > 0:
            ax.axhline(comp_idx - 0.5, color="0.6", ls="--", lw=0.8)

    if show_footnote:
        ax.figure.text(
            0.5, 0.01, UNHEALTHY_SCORE_FOOTNOTE,
            ha="center", va="bottom", fontsize=7.5, transform=ax.figure.transFigure,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="0.75"),
        )
    elif any(is_comp):
        ax.text(
            0.0, -0.22,
            "Composite*: z(smoking)+z(alcohol)+z(sleep cat.); see standalone plot for definition",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left", color="0.35",
        )


def make_plot_b_standalone(concord: Dict[str, Any], out_path: Path) -> None:
    """Standalone high-clarity version of panel B."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    draw_plot_b_panel(ax, concord, title_prefix="", show_footnote=True)
    fig.subplots_adjust(left=0.38, right=0.95, top=0.92, bottom=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_figure(df: pd.DataFrame, resid_sub: pd.DataFrame, concord: Dict, cox_res: Dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # A: log_h by smoking
    ax = axes[0, 0]
    plot_df = df[df["smoking_status"].notna()].copy()
    if len(plot_df) > 0:
        order = [x for x in ["never", "former", "current"] if x in plot_df["smoking_status"].unique()]
        sns.violinplot(data=plot_df, x="smoking_status", y="log_h", hue="cohort", order=order,
                       split=False, ax=ax, inner="box", cut=0)
        ax.set_title("A  log-hazard by smoking status")
        ax.set_xlabel("")
    else:
        ax.text(0.5, 0.5, "No smoking data", ha="center", transform=ax.transAxes)

    # B: partial correlation bars (clear labels + composite definition)
    draw_plot_b_panel(axes[0, 1], concord, title_prefix="B", show_footnote=False)

    # C: C-index M0–M3
    ax = axes[1, 0]
    cohorts = cox_res.get("cohorts", {})
    if cohorts:
        labels, vals, cols = [], [], []
        palette = {"M0": "#95a5a6", "M1": "#3498db", "M2": "#e74c3c", "M3": "#8e44ad"}
        x = 0
        for cohort, models in cohorts.items():
            for m in ["M0", "M1", "M2", "M3"]:
                if m not in models or models[m].get("c_index") is None:
                    continue
                labels.append(f"{cohort}\n{m}")
                vals.append(models[m]["c_index"])
                cols.append(palette.get(m, "#333"))
                x += 1
        ax.bar(range(len(vals)), vals, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.45, 0.95)
        ax.set_ylabel("Harrell C-index")
        ax.set_title("C  Cox models (M0–M3)")
    else:
        ax.text(0.5, 0.5, "Cox models unavailable", ha="center", transform=ax.transAxes)

    # D: KM high vs low log_h_resid (pooled)
    ax = axes[1, 1]
    try:
        from lifelines import KaplanMeierFitter
        sub = resid_sub[np.isfinite(resid_sub["log_h_resid"])].copy()
        if len(sub) >= 40:
            t1, t2 = np.quantile(sub["log_h_resid"], [1 / 3, 2 / 3])
            sub["grp"] = np.where(sub["log_h_resid"] <= t1, "low resid",
                                  np.where(sub["log_h_resid"] >= t2, "high resid", "mid"))
            kmf = KaplanMeierFitter()
            for g, color in [("low resid", "#27ae60"), ("high resid", "#c0392b")]:
                m = sub["grp"] == g
                if m.sum() < 5:
                    continue
                kmf.fit(sub.loc[m, "time"], sub.loc[m, "event"], label=g)
                kmf.plot_survival_function(ax=ax, color=color)
            ax.set_title("D  KM by lifestyle-adjusted log_h tertiles")
            ax.set_xlabel("Time (years)")
        else:
            ax.text(0.5, 0.5, "Insufficient data for KM", ha="center", transform=ax.transAxes)
    except ImportError:
        ax.text(0.5, 0.5, "lifelines required for KM", ha="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged-parquet", type=str,
                   default="feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet")
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/lifestyle")
    p.add_argument("--short-sleep-hours", type=float, default=6.0)
    p.add_argument("--fhs-only", action="store_true",
                   help="Restrict analysis to FHS rows with non-missing lifestyle.")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.merged_parquet)
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run attach_dann_risk.py first.")

    df = pd.read_parquet(path)
    if args.fhs_only:
        mask = df["cohort"].astype(str) == "FHS"
        if "cigarettes_per_day" in df.columns:
            mask &= df["cigarettes_per_day"].notna()
        df = df.loc[mask].copy()
        print(f"  FHS-only analysis subset: n={len(df)}")

    if "_has_lifestyle" in df.columns and df["_has_lifestyle"].sum() == 0:
        print("  WARNING: no lifestyle rows merged; results will be limited.")

    has_lifestyle = (
        ("cigarettes_per_day" in df.columns and df["cigarettes_per_day"].notna().any())
        or ("smoking_status" in df.columns and df["smoking_status"].notna().any())
    )
    df = add_derived(df, short_sleep=args.short_sleep_hours)

    summary: Dict[str, Any] = {
        "n_total": int(len(df)),
        "fhs_only": bool(args.fhs_only),
        "n_with_lifestyle": int(
            df["cigarettes_per_day"].notna().sum()
            if "cigarettes_per_day" in df.columns
            else int(df.get("_has_lifestyle", pd.Series([False] * len(df))).sum())
        ),
        "has_lifestyle_data": has_lifestyle,
        "concordance": concordance_analysis(df) if has_lifestyle else {},
        "cox_models": cox_models(df) if has_lifestyle else {},
        "never_smokers": never_smoker_analysis(df),
        "noise_check": noise_check(df, seed=args.seed),
    }

    resid = residual_risk_analysis(df) if has_lifestyle else {"note": "skipped without lifestyle"}
    summary["residual_risk"] = {k: v for k, v in resid.items() if k != "_sub_with_resid"}
    resid_sub = resid.get("_sub_with_resid", df) if isinstance(resid.get("_sub_with_resid"), pd.DataFrame) else df

    (out_dir / "lifestyle_validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'lifestyle_validation_summary.json'}")

    make_figure(df, resid_sub, summary["concordance"], summary["cox_models"],
                out_dir / "figures" / "lifestyle_validation_supp.pdf")
    make_plot_b_standalone(
        summary["concordance"],
        out_dir / "figures" / "lifestyle_plot_b_concordance.pdf",
    )

    if has_lifestyle and "batch" in df.columns:
        from bio_relevance.lifestyle_confounding_figure import make_confounding_figure
        make_confounding_figure(
            df, out_dir / "figures" / "lifestyle_confounding_figure.pdf",
        )
        from bio_relevance.lifestyle_logh_linkage_figure import make_main_figure, make_simple_figure
        make_simple_figure(df, out_dir / "figures" / "lifestyle_logh_association_simple.pdf")
        make_main_figure(df, summary, out_dir / "figures" / "lifestyle_logh_linkage_main.pdf")


if __name__ == "__main__":
    main()
