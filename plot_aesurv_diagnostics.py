#!/usr/bin/env python3
"""Diagnostic plots for the AESURV-DANN-Aux model.

Plots produced (under ``--out-dir``):

  age_vs_risk.png             True age vs predicted log-hazard (FHS train, FHS val, WHI)
  age_calibration.png         Predicted age vs true age, per cohort + R^2 / MAE
  risk_distribution.png       Risk score by event (violin): FHS-train, FHS-val, WHI
  cell_calibration.png        Predicted vs true cell proportions (FHS-train, FHS-val, WHI)
  km_by_risk_quartile.png     WHI Kaplan-Meier by predicted risk (3 tertiles ~33% each)
  km_by_risk_quartile_fhs.png FHS validation KM by tertiles (legacy filename)
  km_by_risk_tertiles_fhs_train.png  FHS training KM by tertiles
  km_by_risk_tertiles_fhs_val.png    FHS validation KM by tertiles
  km_fhs_train_sex_tertiles.png FHS training KM by sex × within-sex risk tertile
  km_fhs_val_sex_tertiles.png   FHS validation KM (same layout)
  td_auroc_fhs_val_whi_test.png Time-dependent AUROC (FHS val IPCW=FHS train; WHI IPCW=WHI)
  td_auroc_fhs_train_val.png    Time-dependent AUROC: FHS train vs FHS val (IPCW=FHS train)

Inputs:
  --risk-npz       (aesurv_aux_risk.npz) per-sample predictions of the chosen
                   model (default: best single seed)
  --ensemble-npz   optional ensemble_risk.npz (5-seed rank-averaged); if given,
                   risk plots use the ensemble where applicable
  --fhs-cell-pq / --whi-cell-pq    true cell-composition tables
  --fhs-meta-pq  / --whi-meta-pq   parquet metadata for true age
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
# +4 pt vs stock matplotlib defaults (includes prior +2 bump + new +2)
for _k in (
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "figure.titlesize",
):
    _v = plt.rcParams.get(_k)
    if isinstance(_v, (int, float)):
        plt.rcParams[_k] = _v + 4.0
_lt = plt.rcParams.get("legend.title_fontsize")
if isinstance(_lt, (int, float)):
    plt.rcParams["legend.title_fontsize"] = _lt + 4.0
_THRIFT_LIMIT = 2_147_483_647


def _split_indices(n, val_frac, seed, stratify_event):
    """Replicates train_dann_survival._split_indices exactly."""
    idx = np.arange(n)
    strat = stratify_event if (stratify_event is not None and stratify_event.sum() >= 2) else None
    tr, va = train_test_split(idx, test_size=val_frac, random_state=seed,
                              stratify=strat, shuffle=True)
    return tr, va


def _open_pq(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(str(path),
                              thrift_string_size_limit=_THRIFT_LIMIT,
                              thrift_container_size_limit=_THRIFT_LIMIT)
    except TypeError:
        return pq.ParquetFile(str(path))


def _read_age(parquet_path: Path, id_col: str) -> pd.DataFrame:
    pf = _open_pq(parquet_path)
    table = pf.read(columns=[id_col, "age"])
    df = table.to_pandas()
    df[id_col] = df[id_col].astype(str)
    return df


def _pick_sex_column(names: list[str]) -> str | None:
    """Pick a sex/gender column; avoid duplicate lowercase names where a junk column (e.g. all-zero
    ``SEX``) would overwrite a good ``sex`` column when building a naive lower-case dict.
    """
    names_set = set(names)
    for cand in ("sex", "Sex", "SEX", "gender", "Gender", "GENDER"):
        if cand in names_set:
            return cand
    for name in names:
        if name.strip().lower() in ("sex", "gender"):
            return name
    return None


def _read_age_sex(parquet_path: Path, id_col: str) -> pd.DataFrame:
    """Read ID, age, and sex/gender if a matching column exists (same row order as parquet)."""
    pf = _open_pq(parquet_path)
    sch = getattr(pf, "schema_arrow", None) or pf.schema
    names = list(sch.names)
    sex_src = _pick_sex_column(names)
    cols = [id_col, "age"]
    if sex_src:
        cols.append(sex_src)
    table = pf.read(columns=cols)
    df = table.to_pandas()
    df[id_col] = df[id_col].astype(str)
    if sex_src:
        df = df.rename(columns={sex_src: "sex"})
    else:
        df["sex"] = np.nan
    return df


def _sex_to_mf(val: object) -> str:
    """Map a raw sex/gender value to 'M', 'F', or 'U' (unknown)."""
    if val is None or pd.isna(val):
        return "U"
    if isinstance(val, (bool, np.bool_)):
        return "U"
    try:
        fi = int(float(val))
        if fi == 1:
            return "M"
        if fi == 2:
            return "F"
    except (TypeError, ValueError):
        pass
    t = str(val).strip().upper()
    if t in ("M", "MALE", "TRUE"):
        return "M"
    if t in ("F", "FEMALE"):
        return "F"
    if t in ("0", "", "NA", "NAN", "UNK", "UNKNOWN", "U"):
        return "U"
    return "U"


def _sex_labels_series(series: pd.Series) -> np.ndarray:
    return np.array([_sex_to_mf(v) for v in series], dtype=object)


def _color_event(event: np.ndarray) -> list:
    return ["#c0392b" if e else "#7f8c8d" for e in event]


# --------------------------------------------------------------------------- #
# Plot 1: True age vs predicted risk                                          #
# --------------------------------------------------------------------------- #
def plot_age_vs_risk(
    age_fhs_tr: np.ndarray, risk_fhs_tr: np.ndarray, e_fhs_tr: np.ndarray,
    age_fhs_val: np.ndarray, risk_fhs_val: np.ndarray, e_fhs_val: np.ndarray,
    age_whi: np.ndarray, risk_whi: np.ndarray, e_whi: np.ndarray,
    out_path: Path, *, ensemble_risk_whi: Optional[np.ndarray] = None,
) -> None:
    n_cols = 3 + (1 if ensemble_risk_whi is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.7), sharey=False)
    if n_cols == 1:
        axes = np.array([axes])

    def _panel(ax, age, risk, ev, title):
        if len(age) < 3:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=15)
            ax.set_title(title, fontsize=15)
            ax.set_xlabel("Chronological age (years)")
            ax.set_ylabel("Predicted log-hazard")
            ax.grid(True, alpha=0.3)
            return
        colors = _color_event(ev)
        ax.scatter(age, risk, c=colors, s=24, alpha=0.78, linewidths=0)
        m, b = np.polyfit(age, risk, 1)
        xs = np.linspace(float(np.min(age)), float(np.max(age)), 100)
        ax.plot(xs, m * xs + b, "--", color="#2c3e50", lw=1.5,
                label=f"linear fit (slope={m:.3f})")
        r_p, p_p = stats.pearsonr(age, risk)
        r_s, p_s = stats.spearmanr(age, risk)
        ax.set_title(f"{title}\nPearson r={r_p:.3f} (p={p_p:.1e})  |  Spearman {r_s:.3f}",
                     fontsize=15)
        ax.set_xlabel("Chronological age (years)")
        ax.set_ylabel("Predicted log-hazard")
        ax.legend(loc="upper left", fontsize=13, frameon=True)
        ax.grid(True, alpha=0.3)

    _panel(axes[0], age_fhs_tr, risk_fhs_tr, e_fhs_tr, "FHS training")
    _panel(axes[1], age_fhs_val, risk_fhs_val, e_fhs_val, "FHS validation")
    _panel(axes[2], age_whi, risk_whi, e_whi, "WHI (cross-cohort test)")
    if ensemble_risk_whi is not None:
        _panel(axes[3], age_whi, ensemble_risk_whi, e_whi,
               "WHI ensemble (5-seed)")

    handles = [plt.Line2D([], [], marker='o', linestyle='', color='#c0392b', label='event=1 (death)'),
               plt.Line2D([], [], marker='o', linestyle='', color='#7f8c8d', label='event=0 (censored)')]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=14)
    fig.suptitle("True chronological age vs predicted survival risk score", y=1.02, fontsize=17)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Plot 2: Predicted vs true age (calibration of the aux head)                 #
# --------------------------------------------------------------------------- #
def plot_age_calibration(
    age_fhs_tr: np.ndarray, agep_fhs_tr: np.ndarray,
    age_fhs_va: np.ndarray, agep_fhs_va: np.ndarray,
    age_whi: np.ndarray, agep_whi: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))

    def _panel(ax, y_true, y_pred, title, color):
        ax.scatter(y_true, y_pred, c=color, s=22, alpha=0.7, linewidths=0)
        lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
        pad = (hi - lo) * 0.05
        line = np.array([lo - pad, hi + pad])
        ax.plot(line, line, "k-", alpha=0.6, lw=1.0, label="ideal y=x")
        r2 = 1.0 - np.sum((y_pred - y_true) ** 2) / max(1e-9, np.sum((y_true - y_true.mean()) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        bias = float(np.mean(y_pred - y_true))
        ax.set_title(f"{title}\nR^2={r2:.3f}  MAE={mae:.2f}y  RMSE={rmse:.2f}y  bias={bias:+.2f}y",
                     fontsize=15)
        ax.set_xlabel("True age (years)")
        ax.set_ylabel("Predicted age (years)")
        ax.set_xlim(line); ax.set_ylim(line)
        ax.legend(loc="upper left", fontsize=13, frameon=True)
        ax.grid(True, alpha=0.3)

    _panel(axes[0], age_fhs_tr, agep_fhs_tr, "FHS train", "#3498db")
    _panel(axes[1], age_fhs_va, agep_fhs_va, "FHS val", "#9b59b6")
    _panel(axes[2], age_whi, agep_whi, "WHI (test)", "#e67e22")
    fig.suptitle("Aux head: predicted age vs true chronological age", y=1.02, fontsize=17)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Plot 3: Risk distribution by event                                          #
# --------------------------------------------------------------------------- #
def plot_risk_distribution(
    risk_fhs_tr: np.ndarray, e_fhs_tr: np.ndarray,
    risk_fhs_va: np.ndarray, e_fhs_va: np.ndarray,
    risk_whi: np.ndarray, e_whi: np.ndarray,
    out_path: Path,
) -> None:
    parts = []
    if len(risk_fhs_tr) >= 2:
        parts.append(pd.DataFrame({"risk": risk_fhs_tr, "event": e_fhs_tr.astype(int),
                                   "cohort": "FHS-train"}))
    if len(risk_fhs_va) >= 2:
        parts.append(pd.DataFrame({"risk": risk_fhs_va, "event": e_fhs_va.astype(int),
                                   "cohort": "FHS-val"}))
    if len(risk_whi) >= 2:
        parts.append(pd.DataFrame({"risk": risk_whi, "event": e_whi.astype(int),
                                   "cohort": "WHI"}))
    if not parts:
        print(f"  skip {out_path.name}: insufficient rows for risk violin")
        return
    df = pd.concat(parts, ignore_index=True)
    df["event_label"] = df["event"].map({0: "censored", 1: "death"})

    fig_w = 3.2 * df["cohort"].nunique() + 3.0
    fig, ax = plt.subplots(figsize=(max(9.0, fig_w), 5.2))
    sns.violinplot(data=df, x="cohort", y="risk", hue="event_label",
                   split=True, inner="quartile", palette={"censored": "#7f8c8d", "death": "#c0392b"},
                   ax=ax, density_norm="width", gap=0.07)
    ax.set_xlabel("")
    ax.set_ylabel("Predicted log-hazard")

    title_bits = ["Risk score by event status (Mann-Whitney U, events > censored)"]
    if len(risk_fhs_tr) >= 2 and (e_fhs_tr == 0).any() and (e_fhs_tr == 1).any():
        p_tr = stats.mannwhitneyu(risk_fhs_tr[e_fhs_tr == 1], risk_fhs_tr[e_fhs_tr == 0],
                                  alternative="greater").pvalue
        title_bits.append(f"FHS-train p={p_tr:.1e}")
    if len(risk_fhs_va) >= 2 and (e_fhs_va == 0).any() and (e_fhs_va == 1).any():
        p_va = stats.mannwhitneyu(risk_fhs_va[e_fhs_va == 1], risk_fhs_va[e_fhs_va == 0],
                                  alternative="greater").pvalue
        title_bits.append(f"FHS-val p={p_va:.1e}")
    if len(risk_whi) >= 2 and (e_whi == 0).any() and (e_whi == 1).any():
        p_whi = stats.mannwhitneyu(risk_whi[e_whi == 1], risk_whi[e_whi == 0],
                                   alternative="greater").pvalue
        title_bits.append(f"WHI p={p_whi:.1e}")
    ax.set_title("\n".join(title_bits), fontsize=15)
    ax.legend(title="status", loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Plot 4: Cell composition calibration                                        #
# --------------------------------------------------------------------------- #
def plot_cell_calibration(
    cell_true_fhs_tr: Optional[np.ndarray], cell_pred_fhs_tr: Optional[np.ndarray],
    cell_true_fhs_va: np.ndarray, cell_pred_fhs_va: np.ndarray,
    cell_true_whi: np.ndarray, cell_pred_whi: np.ndarray,
    cell_cols: list,
    out_path: Path,
) -> None:
    n_cells = len(cell_cols)
    n_cols = 3
    n_rows = (n_cells + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.3 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    use_tr = (
        cell_true_fhs_tr is not None and cell_pred_fhs_tr is not None
        and len(cell_true_fhs_tr) >= 3 and cell_true_fhs_tr.shape[1] == n_cells
    )

    for i, name in enumerate(cell_cols):
        ax = axes[i]
        if use_tr:
            ax.scatter(cell_true_fhs_tr[:, i], cell_pred_fhs_tr[:, i], c="#3498db",
                       s=14, alpha=0.42, linewidths=0, label="FHS-train")
        ax.scatter(cell_true_fhs_va[:, i], cell_pred_fhs_va[:, i], c="#9b59b6",
                   s=14, alpha=0.45, linewidths=0, label="FHS-val")
        ax.scatter(cell_true_whi[:, i], cell_pred_whi[:, i], c="#e67e22",
                   s=14, alpha=0.55, linewidths=0, label="WHI")
        chunks_t = [cell_true_fhs_va[:, i], cell_true_whi[:, i]]
        chunks_p = [cell_pred_fhs_va[:, i], cell_pred_whi[:, i]]
        if use_tr:
            chunks_t.insert(0, cell_true_fhs_tr[:, i])
            chunks_p.insert(0, cell_pred_fhs_tr[:, i])
        all_true = np.concatenate(chunks_t)
        all_pred = np.concatenate(chunks_p)
        lo = min(all_true.min(), all_pred.min())
        hi = max(all_true.max(), all_pred.max())
        line = np.array([lo, hi])
        ax.plot(line, line, "k-", alpha=0.6, lw=0.9, label="y=x")
        r_va = stats.pearsonr(cell_true_fhs_va[:, i], cell_pred_fhs_va[:, i])[0]
        r_whi = stats.pearsonr(cell_true_whi[:, i], cell_pred_whi[:, i])[0]
        if use_tr:
            r_tr = stats.pearsonr(cell_true_fhs_tr[:, i], cell_pred_fhs_tr[:, i])[0]
            ax.set_title(f"{name}\nr_tr={r_tr:.2f}  r_val={r_va:.2f}  r_WHI={r_whi:.2f}", fontsize=14)
        else:
            ax.set_title(f"{name}\nr_val={r_va:.2f}  r_WHI={r_whi:.2f}", fontsize=14)
        ax.set_xlabel("True proportion")
        ax.set_ylabel("Predicted proportion")
        ax.legend(fontsize=12, loc="upper left", frameon=True)
        ax.grid(True, alpha=0.3)
    for j in range(n_cells, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Predicted vs true Houseman cell-type proportions", y=1.02, fontsize=17)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Plot 5: Kaplan-Meier by predicted-risk tertiles (low / medium / high)       #
# --------------------------------------------------------------------------- #
def _km(t: np.ndarray, e: np.ndarray) -> tuple:
    """Tiny KM estimator. Returns (times, surv)."""
    order = np.argsort(t)
    t = t[order]; e = e[order]
    times = []; surv = []
    s = 1.0
    n_at_risk = len(t)
    i = 0
    while i < len(t):
        cur = t[i]
        j = i
        d = 0
        while j < len(t) and t[j] == cur:
            if e[j] == 1:
                d += 1
            j += 1
        s = s * (1.0 - d / max(1, n_at_risk))
        times.append(cur); surv.append(s)
        n_at_risk -= (j - i)
        i = j
    return np.asarray(times), np.asarray(surv)


def _logrank_test(t1, e1, t2, e2) -> float:
    """Two-sample log-rank test, returns chi-square p-value (approx)."""
    t_all = np.concatenate([t1, t2])
    e_all = np.concatenate([e1, e2])
    g = np.concatenate([np.zeros(len(t1), int), np.ones(len(t2), int)])
    order = np.argsort(t_all)
    t_all = t_all[order]; e_all = e_all[order]; g = g[order]
    unique_t = np.unique(t_all[e_all == 1])
    o1 = 0.0; e1_exp = 0.0; v = 0.0
    for tt in unique_t:
        at_risk = t_all >= tt
        n = int(at_risk.sum())
        n1 = int((at_risk & (g == 0)).sum())
        d = int(((t_all == tt) & (e_all == 1)).sum())
        d1 = int(((t_all == tt) & (e_all == 1) & (g == 0)).sum())
        if n <= 1: continue
        o1 += d1
        e1_exp += d * n1 / n
        v += d * (n1 / n) * (1 - n1 / n) * (n - d) / max(1, n - 1)
    if v <= 0: return 1.0
    chi2 = (o1 - e1_exp) ** 2 / v
    return float(stats.chi2.sf(chi2, df=1))


def _risk_tertile_groups(risk: np.ndarray) -> np.ndarray:
    """Assign 0=low, 1=medium, 2=high by sorted risk (~n/3 samples per group)."""
    n = len(risk)
    if n < 3:
        return np.zeros(n, dtype=int)
    order = np.argsort(risk, kind="mergesort")
    k0 = n // 3
    k1 = (2 * n) // 3
    groups = np.empty(n, dtype=int)
    groups[order[:k0]] = 0
    groups[order[k0:k1]] = 1
    groups[order[k1:]] = 2
    return groups


def plot_km_by_risk_tertiles(
    risk: np.ndarray,
    time_ar: np.ndarray,
    event: np.ndarray,
    out_path: Path,
    *,
    cohort_title: str,
) -> None:
    groups = _risk_tertile_groups(risk)
    labels = ["Low risk", "Medium risk", "High risk"]
    palette = ["#27ae60", "#3498db", "#c0392b"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for g in range(3):
        m = groups == g
        if m.sum() < 2:
            continue
        tt, ss = _km(time_ar[m], event[m])
        tt = np.concatenate([[0.0], tt]); ss = np.concatenate([[1.0], ss])
        ax.step(tt, ss, where="post", lw=1.7, color=palette[g],
                label=f"{labels[g]} (n={int(m.sum())}, events={int(event[m].sum())})")

    p = _logrank_test(time_ar[groups == 0], event[groups == 0],
                      time_ar[groups == 2], event[groups == 2])

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (units of follow-up time as in input)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{cohort_title} Kaplan-Meier by predicted risk (tertiles, ~33% each)\n"
                 f"log-rank low vs high: p={p:.1e}", fontsize=15)
    ax.legend(loc="lower left", fontsize=13, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_km_fhs_sex_tertiles(
    risk: np.ndarray,
    time_ar: np.ndarray,
    event: np.ndarray,
    sex: np.ndarray,
    out_path: Path,
    *,
    split_title: str,
) -> None:
    """KM curves: two colors (male / female); linestyles solid, dashed, dotted = low/med/high
    risk tertiles **within each sex** (~33% of that sex per tertile).
    """
    sex_colors = {"M": "#1f77b4", "F": "#d62728"}
    sex_names = {"M": "Male", "F": "Female"}
    tert_ls = {0: "-", 1: "--", 2: ":"}
    tert_name = {0: "low risk", 1: "medium risk", 2: "high risk"}

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    n_plotted = 0
    for sk in ("M", "F"):
        msex = sex == sk
        n_s = int(msex.sum())
        if n_s < 6:
            continue
        r_s = risk[msex].astype(np.float64)
        t_s = time_ar[msex]
        e_s = event[msex]
        groups = _risk_tertile_groups(r_s)
        for g in range(3):
            mloc = groups == g
            if int(mloc.sum()) < 2:
                continue
            tt, ss = _km(t_s[mloc], e_s[mloc])
            tt = np.concatenate([[0.0], tt])
            ss = np.concatenate([[1.0], ss])
            ax.step(
                tt, ss, where="post", lw=2.0, color=sex_colors[sk], linestyle=tert_ls[g],
                label=f"{sex_names[sk]} {tert_name[g]} (n={int(mloc.sum())}, "
                f"events={int(e_s[mloc].sum())})",
            )
            n_plotted += 1

    if n_plotted == 0:
        plt.close(fig)
        print(f"  skip {out_path.name}: need FHS {split_title} sex (M/F) and enough samples per tertile")
        return

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (units of follow-up time as in input)")
    ax.set_ylabel("Survival probability")
    ax.set_title(
        f"{split_title}: Kaplan-Meier by sex and within-sex risk tertile\n"
        "(blue = male, red = female; solid / dashed / dotted = low / medium / high risk)",
        fontsize=15,
    )
    ax.legend(loc="lower left", fontsize=13, frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _compute_ipcw_td_auroc_curve(
    ipcw_times: np.ndarray,
    ipcw_events: np.ndarray,
    test_times: np.ndarray,
    test_events: np.ndarray,
    test_risks: np.ndarray,
    *,
    n_times: int = 48,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Return (evaluation_times, aucs, integrated_mean) for Uno's cumulative_dynamic_auc.

    ``ipcw_times`` / ``ipcw_events`` define the cohort used to estimate IPCW censoring
    (``survival_train``). Use FHS **training** rows when ``test_*`` is FHS validation; use
    the **same** test cohort for both when scoring an external set whose follow-up lies
    outside the training support (e.g. WHI vs FHS-only IPCW).
    """
    try:
        from sksurv.metrics import cumulative_dynamic_auc
    except ImportError:
        return None

    t_ip = np.asarray(ipcw_times, dtype=np.float64).ravel()
    e_ip = np.asarray(ipcw_events, dtype=np.int32).ravel()
    t_te = np.asarray(test_times, dtype=np.float64).ravel()
    e_te = np.asarray(test_events, dtype=np.int32).ravel()
    risk_te = np.asarray(test_risks, dtype=np.float64).ravel()

    y_ip = np.empty(len(t_ip), dtype=[("event", np.bool_), ("time", np.float64)])
    y_ip["event"] = e_ip.astype(np.bool_)
    y_ip["time"] = t_ip
    y_te = np.empty(len(t_te), dtype=[("event", np.bool_), ("time", np.float64)])
    y_te["event"] = e_te.astype(np.bool_)
    y_te["time"] = t_te

    ev_t = t_te[e_te == 1]
    if len(ev_t) < 10:
        return None
    qs = np.linspace(0.03, 0.97, min(n_times, max(10, len(ev_t))))
    eval_times = np.unique(np.clip(np.quantile(ev_t, qs), 1e-8, None))
    tmax_ip = float(np.max(t_ip))
    eval_times = eval_times[eval_times < tmax_ip - 1e-9]
    if len(eval_times) < 4:
        lo = float(np.min(ev_t))
        hi = min(float(np.max(ev_t)), tmax_ip * 0.999)
        if hi <= lo + 1e-9:
            return None
        eval_times = np.linspace(lo, hi, min(max(n_times, 8), 30))
        eval_times = np.unique(eval_times[eval_times < tmax_ip - 1e-9])
    if len(eval_times) < 4:
        return None
    try:
        aucs, mean_auc = cumulative_dynamic_auc(y_ip, y_te, risk_te, eval_times)
    except Exception:
        return None
    return eval_times, np.asarray(aucs, dtype=np.float64).ravel(), float(mean_auc)


def plot_td_auroc_fhs_whi_panels(
    risk_data: Dict[str, np.ndarray],
    out_path: Path,
    *,
    risk_whi: np.ndarray,
) -> None:
    """Time-dependent AUROC: FHS validation (IPCW = FHS train) and WHI (IPCW = WHI)."""
    req = (
        "time_fhs_train", "event_fhs_train", "risk_fhs_train",
        "time_fhs_val", "event_fhs_val", "risk_fhs_val",
        "time_whi_test", "event_whi_test",
    )
    if any(k not in risk_data for k in req):
        print(f"  skip {out_path.name}: risk npz missing time/event columns for TD-AUROC")
        return

    t_tr = risk_data["time_fhs_train"].astype(np.float64)
    e_tr = risk_data["event_fhs_train"].astype(np.int32)
    t_whi = risk_data["time_whi_test"].astype(np.float64)
    e_whi = risk_data["event_whi_test"].astype(np.int32)
    r_whi = np.asarray(risk_whi, dtype=np.float64)

    cur_fhs = _compute_ipcw_td_auroc_curve(
        t_tr,
        e_tr,
        risk_data["time_fhs_val"].astype(np.float64),
        risk_data["event_fhs_val"].astype(np.int32),
        risk_data["risk_fhs_val"].astype(np.float64),
    )
    # WHI follow-up extends beyond FHS train max → FHS-train-only IPCW has zero
    # censoring survival. Use WHI as its own IPCW cohort (valid Uno estimator on WHI).
    cur_whi = _compute_ipcw_td_auroc_curve(
        t_whi,
        e_whi,
        t_whi,
        e_whi,
        r_whi,
    )
    if cur_fhs is None and cur_whi is None:
        print(f"  skip {out_path.name}: scikit-survival missing or insufficient events")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)

    def _one(ax, cur, title):
        if cur is None:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes, fontsize=15)
            ax.set_title(title, fontsize=15)
            return
        et, aucs, mean_i = cur
        ax.plot(et, aucs, "o-", color="#2c3e50", lw=1.4, ms=4, label="TD-AUROC(t)")
        ax.axhline(mean_i, color="#c0392b", ls="--", lw=1.2,
                   label=f"integrated mean = {mean_i:.3f}")
        ax.set_xlabel("Time (evaluation grid, event-time quantiles)", fontsize=14)
        ax.set_ylabel("Time-dependent AUROC", fontsize=14)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title, fontsize=15)
        ax.legend(loc="lower right", fontsize=12, frameon=True)
        ax.grid(True, alpha=0.3)

    _one(axes[0], cur_fhs, "FHS validation\n(IPCW censoring: FHS training)")
    _one(axes[1], cur_whi, "WHI test\n(IPCW censoring: WHI cohort)")
    fig.suptitle(
        "Time-dependent AUROC (Uno's cumulative dynamic AUC; higher risk = higher score)\n"
        "FHS val: IPCW from FHS train. WHI: IPCW from WHI itself (FHS-train IPCW fails when WHI times exceed ~15y).",
        fontsize=16, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_td_auroc_fhs_train_val_panels(risk_data: Dict[str, np.ndarray], out_path: Path) -> None:
    """Time-dependent AUROC on FHS training (IPCW = train) vs FHS validation (IPCW = train)."""
    req = (
        "time_fhs_train", "event_fhs_train", "risk_fhs_train",
        "time_fhs_val", "event_fhs_val", "risk_fhs_val",
    )
    if any(k not in risk_data for k in req):
        print(f"  skip {out_path.name}: risk npz missing FHS train/val columns for TD-AUROC")
        return

    t_tr = risk_data["time_fhs_train"].astype(np.float64)
    e_tr = risk_data["event_fhs_train"].astype(np.int32)
    r_tr = risk_data["risk_fhs_train"].astype(np.float64)
    t_va = risk_data["time_fhs_val"].astype(np.float64)
    e_va = risk_data["event_fhs_val"].astype(np.int32)
    r_va = risk_data["risk_fhs_val"].astype(np.float64)

    cur_train = _compute_ipcw_td_auroc_curve(t_tr, e_tr, t_tr, e_tr, r_tr)
    cur_val = _compute_ipcw_td_auroc_curve(t_tr, e_tr, t_va, e_va, r_va)
    if cur_train is None and cur_val is None:
        print(f"  skip {out_path.name}: scikit-survival missing or insufficient events")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)

    def _one(ax, cur, title):
        if cur is None:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes, fontsize=15)
            ax.set_title(title, fontsize=15)
            return
        et, aucs, mean_i = cur
        ax.plot(et, aucs, "o-", color="#2c3e50", lw=1.4, ms=4, label="TD-AUROC(t)")
        ax.axhline(mean_i, color="#c0392b", ls="--", lw=1.2,
                   label=f"integrated mean = {mean_i:.3f}")
        ax.set_xlabel("Time (evaluation grid, event-time quantiles)", fontsize=14)
        ax.set_ylabel("Time-dependent AUROC", fontsize=14)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title, fontsize=15)
        ax.legend(loc="lower right", fontsize=12, frameon=True)
        ax.grid(True, alpha=0.3)

    _one(axes[0], cur_train, "FHS training\n(IPCW censoring: FHS training)")
    _one(axes[1], cur_val, "FHS validation\n(IPCW censoring: FHS training)")
    fig.suptitle(
        "Time-dependent AUROC on FHS: training vs validation\n"
        "(Uno's cumulative dynamic AUC; IPCW for censoring always estimated from FHS training.)",
        fontsize=16, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--risk-npz", type=str,
                   default="runs/aesurv_aux_grid/age12.00_cell1.00/aesurv_aux_risk.npz")
    p.add_argument("--ensemble-npz", type=str,
                   default="runs/aesurv_aux_seeds/age12.00_cell1.00_ENSEMBLE/ensemble_risk.npz")
    p.add_argument("--fhs-meta-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-meta-pq", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--fhs-cell-pq", type=str, default="FHS_cell_composition.parquet")
    p.add_argument("--whi-cell-pq", type=str, default="WHI_cell_composition.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--whi-id-col", type=str, default="sample_ID")
    p.add_argument("--cell-cols", type=str, default="B,NK,CD4T,CD8T,Mono,Neutro")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--out-dir", type=str, default="plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading per-sample predictions from {args.risk_npz} ...")
    risk_data: Dict[str, np.ndarray] = dict(np.load(args.risk_npz, allow_pickle=False))
    for k, v in risk_data.items():
        print(f"  {k:30s} shape={v.shape}")

    print(f"\n[2/5] Loading true age (+ sex for FHS) from parquets ...")
    df_age_fhs = _read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    df_age_whi = _read_age(Path(args.whi_meta_pq), args.whi_id_col)
    age_fhs_full = df_age_fhs["age"].to_numpy(dtype=np.float32)
    sex_fhs_full = _sex_labels_series(df_age_fhs["sex"])
    age_whi_full = df_age_whi["age"].to_numpy(dtype=np.float32)
    print(f"  FHS rows={len(age_fhs_full)} (NaN age = {np.isnan(age_fhs_full).sum()})")
    print(f"  WHI rows={len(age_whi_full)} (NaN age = {np.isnan(age_whi_full).sum()})")
    if np.isnan(age_fhs_full).any():
        age_fhs_full = np.where(np.isnan(age_fhs_full),
                                np.nanmean(age_fhs_full), age_fhs_full)
    if np.isnan(age_whi_full).any():
        age_whi_full = np.where(np.isnan(age_whi_full),
                                np.nanmean(age_whi_full), age_whi_full)

    # Recover the FHS train/val split (seed=42, val_frac=0.15, event-stratified)
    # so we can align true age with predicted risk/age for the val set.
    e_fhs_full = np.concatenate([risk_data["event_fhs_train"], risk_data["event_fhs_val"]])
    # That concatenation is in train, then val order, not original order. So we
    # need to recover the original-order event array. We have len = n_fhs total.
    n_fhs = len(age_fhs_full)
    print(f"\n[3/5] Reproducing FHS train/val split (seed={args.seed}, val_frac={args.val_frac}) ...")
    # The training script's _split_indices takes the FULL FHS event array. We
    # don't have it directly here, but we can reconstruct: the union of train
    # and val events (with multiplicity) matches what was used. We only need
    # the original event array to call _split_indices, which we get from the
    # FHS parquet. (Both have 't' and 'e' columns under various names; we
    # cached them in vae_cox_cache.)
    event_arr = None
    cache_candidates = sorted(Path("vae_cox_cache/bundles").glob("FHS_FHS_*cpgall_snpall*.npz"))
    cache_candidates += sorted(Path("vae_cox_cache").glob("FHS_FHS_*cpgall_snpall*.npz"))
    for cache_npz in cache_candidates:
        try:
            with np.load(cache_npz, allow_pickle=False, mmap_mode="r") as zc:
                if "event" in zc.files:
                    cand = np.asarray(zc["event"], dtype=np.int32).reshape(-1)
                    if len(cand) == n_fhs:
                        event_arr = cand
                        print(f"  Using cached FHS event array from {cache_npz}")
                        break
        except Exception as exc:
            print(f"  (skip {cache_npz}: {exc})")
            continue

    if event_arr is None:
        # Heuristic: if we can't recover the original split, just plot full FHS
        # (train+val combined) using the union of predicted risks.
        print("  Could not recover original event order; FHS age/sex/cell plots will be skipped, "
              "but survival and risk from the npz are still used where possible.")
        age_fhs_tr = age_fhs_full[:0]
        age_fhs_val = age_fhs_full[:0]
        risk_fhs_train = risk_data["risk_fhs_train"].astype(np.float32)
        risk_fhs_val = risk_data["risk_fhs_val"].astype(np.float32)
        e_fhs_tr = risk_data["event_fhs_train"].astype(int)
        e_fhs_val = risk_data["event_fhs_val"].astype(int)
        t_fhs_tr = risk_data["time_fhs_train"].astype(np.float32)
        t_fhs_val = risk_data["time_fhs_val"].astype(np.float32)
        agep_fhs_val = np.empty(0); agep_fhs_tr = np.empty(0)
        cell_pred_fhs_va = np.empty((0, len(args.cell_cols.split(","))), dtype=np.float32)
        cell_pred_fhs_tr = np.empty_like(cell_pred_fhs_va)
        cell_true_fhs_tr = np.empty_like(cell_pred_fhs_va)
        cell_true_fhs_val = np.empty_like(cell_pred_fhs_va)
        sex_fhs_val = np.empty(0, dtype=object)
        sex_fhs_tr = np.empty(0, dtype=object)
    else:
        tr_idx, va_idx = _split_indices(n_fhs, args.val_frac, args.seed,
                                        stratify_event=event_arr)
        # sanity: events recovered must match
        n_ev_va_now = int(event_arr[va_idx].sum())
        n_ev_va_npz = int(risk_data["event_fhs_val"].sum())
        if n_ev_va_now != n_ev_va_npz:
            print(f"  WARNING: val event count drift ({n_ev_va_now} vs {n_ev_va_npz}); "
                  f"split mapping may be slightly off.")
        age_fhs_val = age_fhs_full[va_idx]
        age_fhs_tr = age_fhs_full[tr_idx]
        risk_fhs_val = risk_data["risk_fhs_val"]
        risk_fhs_train = risk_data["risk_fhs_train"]
        e_fhs_val = risk_data["event_fhs_val"].astype(int)
        e_fhs_tr = risk_data["event_fhs_train"].astype(int)
        t_fhs_val = risk_data["time_fhs_val"].astype(np.float32)
        t_fhs_tr = risk_data["time_fhs_train"].astype(np.float32)
        agep_fhs_val = risk_data["age_pred_fhs_val"]
        agep_fhs_tr = risk_data["age_pred_fhs_train"]
        cell_pred_fhs_va = risk_data["cell_pred_fhs_val"]
        cell_pred_fhs_tr = risk_data["cell_pred_fhs_train"]
        # We also need true cells for FHS-val
        cell_cols = [c.strip() for c in args.cell_cols.split(",") if c.strip()]
        df_cell_fhs = pd.read_parquet(args.fhs_cell_pq)
        df_cell_fhs[args.fhs_id_col] = df_cell_fhs[args.fhs_id_col].astype(str)
        df_join = df_age_fhs.merge(df_cell_fhs, on=args.fhs_id_col, how="left")
        cell_true_full = df_join[cell_cols].to_numpy(dtype=np.float32)
        cell_true_fhs_val = cell_true_full[va_idx]
        cell_true_fhs_tr = cell_true_full[tr_idx]
        sex_fhs_val = sex_fhs_full[va_idx]
        sex_fhs_tr = sex_fhs_full[tr_idx]

    age_whi = age_whi_full
    risk_whi = risk_data["risk_whi_test"]
    e_whi = risk_data["event_whi_test"].astype(int)
    t_whi = risk_data["time_whi_test"]
    agep_whi = risk_data["age_pred_whi"]
    cell_pred_whi = risk_data["cell_pred_whi"]

    cell_cols = [c.strip() for c in args.cell_cols.split(",") if c.strip()]
    df_cell_whi = pd.read_parquet(args.whi_cell_pq)
    df_cell_whi[args.whi_id_col] = df_cell_whi[args.whi_id_col].astype(str)
    df_join_w = df_age_whi.merge(df_cell_whi, on=args.whi_id_col, how="left")
    cell_true_whi = df_join_w[cell_cols].to_numpy(dtype=np.float32)

    ens_whi = None
    if args.ensemble_npz and Path(args.ensemble_npz).exists():
        ens = np.load(args.ensemble_npz, allow_pickle=False)
        if "risk_whi_test" in ens.files:
            ens_whi = ens["risk_whi_test"]
            print(f"  Using ensemble WHI risk (n={len(ens_whi)}) for the WHI ensemble panel.")

    print("\n[4/5] Rendering plots ...")
    plot_age_vs_risk(
        age_fhs_tr, risk_fhs_train, e_fhs_tr,
        age_fhs_val, risk_fhs_val, e_fhs_val,
        age_whi, risk_whi, e_whi,
        out_dir / "age_vs_risk.png",
        ensemble_risk_whi=ens_whi,
    )
    plot_age_calibration(age_fhs_tr, agep_fhs_tr,
                          age_fhs_val, agep_fhs_val,
                          age_whi, agep_whi,
                          out_dir / "age_calibration.png")
    plot_risk_distribution(
        risk_fhs_train, e_fhs_tr,
        risk_fhs_val, e_fhs_val,
        risk_whi, e_whi,
        out_dir / "risk_distribution.png",
    )
    if (
        len(cell_pred_fhs_va) >= 3
        and len(cell_true_whi) >= 3
        and cell_pred_fhs_va.shape[1] == len(cell_cols)
    ):
        plot_cell_calibration(
            cell_true_fhs_tr if len(cell_true_fhs_tr) else None,
            cell_pred_fhs_tr if len(cell_pred_fhs_tr) else None,
            cell_true_fhs_val, cell_pred_fhs_va,
            cell_true_whi, cell_pred_whi,
            cell_cols, out_dir / "cell_calibration.png",
        )
    else:
        print("  skip cell_calibration.png: need aligned FHS-val and WHI cell predictions")
    plot_km_by_risk_tertiles(
        risk_whi if ens_whi is None else ens_whi,
        t_whi, e_whi,
        out_dir / "km_by_risk_quartile.png",
        cohort_title="WHI (cross-cohort test)",
    )
    if len(t_fhs_tr) >= 8 and len(risk_fhs_train) == len(t_fhs_tr):
        plot_km_by_risk_tertiles(
            risk_fhs_train, t_fhs_tr, e_fhs_tr,
            out_dir / "km_by_risk_tertiles_fhs_train.png",
            cohort_title="FHS (training)",
        )
    else:
        print(f"  skip km_by_risk_tertiles_fhs_train.png: need aligned FHS train time/risk (got n={len(t_fhs_tr)})")

    if len(t_fhs_val) >= 8 and len(risk_fhs_val) == len(t_fhs_val):
        plot_km_by_risk_tertiles(
            risk_fhs_val, t_fhs_val, e_fhs_val,
            out_dir / "km_by_risk_tertiles_fhs_val.png",
            cohort_title="FHS (validation)",
        )
        plot_km_by_risk_tertiles(
            risk_fhs_val, t_fhs_val, e_fhs_val,
            out_dir / "km_by_risk_quartile_fhs.png",
            cohort_title="FHS (validation)",
        )
    else:
        print(f"  skip km_by_risk_quartile_fhs.png: need aligned FHS val time/risk (got n={len(t_fhs_val)})")

    if (
        len(risk_fhs_val) == len(t_fhs_val) == len(e_fhs_val) == len(sex_fhs_val)
        and len(risk_fhs_val) >= 6
        and ((sex_fhs_val == "M") | (sex_fhs_val == "F")).any()
    ):
        plot_km_fhs_sex_tertiles(
            risk_fhs_val, t_fhs_val, e_fhs_val, sex_fhs_val,
            out_dir / "km_fhs_val_sex_tertiles.png",
            split_title="FHS validation",
        )
    else:
        print(f"  skip km_fhs_val_sex_tertiles.png: aligned val rows with M/F sex required "
              f"(n_risk={len(risk_fhs_val)}, n_sex={len(sex_fhs_val)})")

    if (
        len(risk_fhs_train) == len(t_fhs_tr) == len(e_fhs_tr) == len(sex_fhs_tr)
        and len(risk_fhs_train) >= 6
        and ((sex_fhs_tr == "M") | (sex_fhs_tr == "F")).any()
    ):
        plot_km_fhs_sex_tertiles(
            risk_fhs_train, t_fhs_tr, e_fhs_tr, sex_fhs_tr,
            out_dir / "km_fhs_train_sex_tertiles.png",
            split_title="FHS training",
        )
    else:
        print(f"  skip km_fhs_train_sex_tertiles.png: aligned train rows with M/F sex required "
              f"(n_risk={len(risk_fhs_train)}, n_sex={len(sex_fhs_tr)})")

    plot_td_auroc_fhs_train_val_panels(risk_data, out_dir / "td_auroc_fhs_train_val.png")

    plot_td_auroc_fhs_whi_panels(
        risk_data,
        out_dir / "td_auroc_fhs_val_whi_test.png",
        risk_whi=risk_whi if ens_whi is None else ens_whi,
    )

    print(f"\n[5/5] Done. Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
