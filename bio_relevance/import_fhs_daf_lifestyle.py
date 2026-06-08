#!/usr/bin/env python3
"""Import FHS DAF Cox lifestyle file and harmonize to canonical columns.

Source: FHS_all_daf/All_data_cox.csv (shareid + 3 lifestyle columns).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from bio_relevance.lifestyle_common import normalize_id, winsorize_series

DAF_COL_ALC = "Amount of alcohol consumption at a time"
DAF_COL_CIG = "No of cig/day"
DAF_COL_SLP = "Sleep hours"


def _find_first_matching(raw: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return first column whose name contains any keyword (case-insensitive)."""
    lower = {c.lower(): c for c in raw.columns}
    for key in keywords:
        k = key.lower()
        for cname_lower, cname in lower.items():
            if k in cname_lower:
                return cname
    return None


def import_daf(
    daf_csv: Path,
    out_dir: Path,
    winsor_q: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    raw = pd.read_csv(daf_csv, low_memory=False)
    id_col = "shareid" if "shareid" in raw.columns else "Share_ID"
    for c in (DAF_COL_ALC, DAF_COL_CIG, DAF_COL_SLP):
        if c not in raw.columns:
            raise ValueError(f"Missing column {c!r}; got {list(raw.columns)}")

    out = pd.DataFrame({
        "cohort": "FHS",
        "subject_id": raw[id_col].map(normalize_id),
        "merge_key": raw[id_col].map(normalize_id),
        "cigarettes_per_day": pd.to_numeric(raw[DAF_COL_CIG], errors="coerce"),
        "alcohol_amount_per_occasion": pd.to_numeric(raw[DAF_COL_ALC], errors="coerce"),
        "sleep_hours": pd.to_numeric(raw[DAF_COL_SLP], errors="coerce"),
    })
    # Back-compat alias used by validation script
    out["alcohol_drinks_per_week"] = out["alcohol_amount_per_occasion"]

    cigs = out["cigarettes_per_day"].fillna(0)
    out["smoking_status"] = np.where(cigs <= 0, "never", "current")

    meta: dict = {
        "source": str(daf_csv),
        "n_rows_raw": int(len(raw)),
        "n_rows": int(len(out)),
        "id_column": id_col,
        "column_map": {
            "cigarettes_per_day": DAF_COL_CIG,
            "alcohol_amount_per_occasion": DAF_COL_ALC,
            "sleep_hours": DAF_COL_SLP,
        },
        "note_alcohol": "alcohol_amount_per_occasion is drinks/amount per occasion (not per week).",
        "smoking_status_rule": "cigarettes_per_day <= 0 -> never; else current (no former/current distinction in source).",
        "winsor": {},
    }
    for col in ("cigarettes_per_day", "alcohol_amount_per_occasion", "sleep_hours"):
        out[col], wb = winsorize_series(out[col], winsor_q[0], winsor_q[1])
        if wb:
            meta["winsor"][col] = wb

    # Optional cardio-metabolic traits (LDL, HDL, total cholesterol, triglycerides, glucose)
    lipid_specs = {
        "ldl": ["ldl", "low density lipoprotein"],
        "hdl": ["hdl", "high density lipoprotein"],
        "total_cholesterol": ["total cholesterol", "cholesterol total", "cholesterol"],
        "triglycerides": ["triglyceride", "triglycerides", "tg"],
        "glucose": ["glucose", "fasting glucose", "serum glucose"],
    }
    for canon_name, keys in lipid_specs.items():
        src = _find_first_matching(raw, keys)
        if not src:
            continue
        vals = pd.to_numeric(raw[src], errors="coerce")
        out[canon_name], wb = winsorize_series(vals, winsor_q[0], winsor_q[1])
        meta["column_map"][canon_name] = src
        if wb:
            meta["winsor"][canon_name] = wb

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "lifestyle_harmonized.parquet", index=False)
    meta["missingness"] = {c: float(out[c].isna().mean()) for c in out.columns if c != "cohort"}
    (out_dir / "lifestyle_harmonization.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'lifestyle_harmonized.parquet'}  n={len(out)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--daf-csv",
        type=str,
        default=r"F:\FHS_phenotypic data\FHS_DAF_COX\FHS_all_daf\All_data_cox.csv",
    )
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/lifestyle")
    args = p.parse_args()
    import_daf(Path(args.daf_csv), Path(args.out_dir))


if __name__ == "__main__":
    main()
