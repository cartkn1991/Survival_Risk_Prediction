"""Shared helpers for lifestyle harmonization and merge."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CANONICAL_LIFESTYLE_COLS = [
    "cohort",
    "subject_id",
    "merge_key",
    "smoking_status",
    "cigarettes_per_day",
    "pack_years",
    "alcohol_drinks_per_week",
    "sleep_hours",
    "sleep_quality",
    "physical_activity",
    "bmi",
    "exam_year",
]

SMOKING_LEVELS = ("never", "former", "current")


def normalize_id(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    if re.match(r"^\d+\.0$", s):
        s = s[:-2]
    return s


def load_variable_map(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pick_column(df: pd.DataFrame, name: Optional[str], alternates: List[str]) -> Optional[str]:
    if name and name in df.columns:
        return name
    lower = {c.lower(): c for c in df.columns}
    for alt in alternates:
        if alt in df.columns:
            return alt
        if alt.lower() in lower:
            return lower[alt.lower()]
    return None


def recode_smoking(series: pd.Series, value_map: Dict[str, str]) -> pd.Series:
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        key = str(v).strip()
        if key in value_map:
            out.append(value_map[key])
            continue
        try:
            fk = str(int(float(key)))
            if fk in value_map:
                out.append(value_map[fk])
                continue
        except (ValueError, TypeError):
            pass
        kl = key.lower()
        if kl in ("never", "former", "current"):
            out.append(kl)
        elif "never" in kl:
            out.append("never")
        elif "former" in kl or "ex-" in kl or "quit" in kl:
            out.append("former")
        elif "current" in kl or "yes" == kl:
            out.append("current")
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="object")


def winsorize_series(s: pd.Series, q_lo: float = 0.01, q_hi: float = 0.99) -> Tuple[pd.Series, Dict[str, float]]:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return x, {}
    lo, hi = float(valid.quantile(q_lo)), float(valid.quantile(q_hi))
    return x.clip(lo, hi), {"lo": lo, "hi": hi, "q_lo": q_lo, "q_hi": q_hi}


def smoking_ordinal(s: pd.Series) -> pd.Series:
    m = {"never": 0.0, "former": 1.0, "current": 2.0}
    return s.map(m)


def partial_spearman(
    y: np.ndarray,
    x: np.ndarray,
    cov: np.ndarray,
) -> Tuple[float, float]:
    """Spearman correlation between residuals of rank(y) and rank(x) on covariates."""
    from scipy import stats

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(cov), axis=1)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    y, x, cov = y[mask], x[mask], cov[mask]
    ry = stats.rankdata(y)
    rx = stats.rankdata(x)
    X = np.column_stack([np.ones(len(ry)), cov])
    by, *_ = np.linalg.lstsq(X, ry, rcond=None)
    bx, *_ = np.linalg.lstsq(X, rx, rcond=None)
    res_y = ry - X @ by
    res_x = rx - X @ bx
    r, p = stats.pearsonr(res_y, res_x)
    return float(r), float(p)


def bh_fdr(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n)
    out[order] = q
    return out.tolist()
