#!/usr/bin/env python3
"""Harmonize dbGaP lifestyle CSV exports to canonical columns.

Outputs:
  feature_importance/bio_relevance/lifestyle/lifestyle_harmonized.parquet
  feature_importance/bio_relevance/lifestyle/lifestyle_harmonization.json

Use --demo to build synthetic lifestyle for pipeline testing (not for publication).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bio_relevance.lifestyle_common import (
    CANONICAL_LIFESTYLE_COLS,
    load_variable_map,
    normalize_id,
    pick_column,
    recode_smoking,
    winsorize_series,
)

_THRIFT_LIMIT = 2_147_483_647


def _open_pq(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(str(path), thrift_string_size_limit=_THRIFT_LIMIT,
                              thrift_container_size_limit=_THRIFT_LIMIT)
    except TypeError:
        return pq.ParquetFile(str(path))


def _read_parquet_ids(path: Path, id_col: str) -> pd.DataFrame:
    pf = _open_pq(path)
    cols = [c for c in [id_col, "IID", "Share_ID", "sample_ID", "SUBJID", "dbGaP_Subject_ID"]
            if c in pf.schema_arrow.names]
    df = pf.read(columns=cols).to_pandas()
    df["cohort"] = "FHS" if "Share_ID" in cols or id_col == "Share_ID" else "WHI"
    if "FHS" not in df["cohort"].iloc[0]:
        df["cohort"] = "WHI" if "sample_ID" in cols else "FHS"
    if path.name.upper().startswith("FHS"):
        df["cohort"] = "FHS"
    else:
        df["cohort"] = "WHI"
    df["subject_id"] = df[id_col].map(normalize_id)
    return df


def _harmonize_cohort_raw(
    raw: pd.DataFrame,
    cfg: Dict[str, Any],
    harm: Dict[str, Any],
) -> pd.DataFrame:
    cohort = cfg["cohort"]
    id_col = pick_column(raw, cfg.get("id_column"), cfg.get("alternate_id_columns", []))
    if id_col is None:
        raise ValueError(f"[{cohort}] ID column not found in {list(raw.columns)}")

    colmap = cfg.get("columns", {})
    out = pd.DataFrame()
    out["cohort"] = cohort
    sid_src = colmap.get("subject_id", id_col)
    sid_c = pick_column(raw, sid_src, [id_col, "subject_id"])
    out["subject_id"] = raw[sid_c].map(normalize_id)
    out["merge_key"] = out["subject_id"]

    meta: Dict[str, Any] = {"cohort": cohort, "id_column_used": id_col, "winsor": {}}

    sm_col = pick_column(raw, colmap.get("smoking_status"), ["smoking_status", "SMOKE", "SMOKING"])
    if sm_col:
        out["smoking_status"] = recode_smoking(
            raw[sm_col], {str(k): v for k, v in cfg.get("smoking_value_map", {}).items()}
        )
    else:
        out["smoking_status"] = np.nan

    numeric_fields = [
        ("cigarettes_per_day", ["cigarettes_per_day", "CIGPD", "CIGDAY"]),
        ("pack_years", ["pack_years", "PACKYRS"]),
        ("alcohol_drinks_per_week", ["alcohol_drinks_per_week", "DRINKSWK", "ALCDRWK"]),
        ("sleep_hours", ["sleep_hours", "SLEEPHR", "SLEEPHRS"]),
        ("sleep_quality", ["sleep_quality", "SLEEPQL", "SLEEPQUAL"]),
        ("physical_activity", ["physical_activity", "ACTIVITY", "METWEEK"]),
        ("bmi", ["bmi", "BMI"]),
        ("exam_year", ["exam_year", "EXAMYR", "VISITYR"]),
    ]
    wq = harm.get("winsor_quantiles", [0.01, 0.99])
    for canon, alts in numeric_fields:
        src = colmap.get(canon)
        c = pick_column(raw, src, alts)
        if c is None:
            out[canon] = np.nan
            continue
        out[canon] = pd.to_numeric(raw[c], errors="coerce")
        if canon in ("cigarettes_per_day", "pack_years", "alcohol_drinks_per_week",
                     "sleep_hours", "physical_activity", "bmi"):
            out[canon], wb = winsorize_series(out[canon], wq[0], wq[1])
            if wb:
                meta["winsor"][canon] = wb

    cap = harm.get("alcohol_drinks_per_week_cap")
    if cap is not None and "alcohol_drinks_per_week" in out.columns:
        out["alcohol_drinks_per_week"] = out["alcohol_drinks_per_week"].clip(upper=float(cap))

    # WHI: optional map dbGaP_Subject_ID -> SAMPLE_ID for merge to sample_ID
    map_csv = cfg.get("whi_id_map_csv")
    if cohort == "WHI" and map_csv and Path(map_csv).exists():
        mfrom = cfg.get("whi_id_map_from", "dbGaP_Subject_ID")
        mto = cfg.get("whi_id_map_to", "SAMPLE_ID")
        mp = pd.read_csv(map_csv, low_memory=False)
        mfrom_c = pick_column(mp, mfrom, [mfrom])
        mto_c = pick_column(mp, mto, [mto])
        if mfrom_c and mto_c:
            mp = mp[[mfrom_c, mto_c]].drop_duplicates()
            mp[mfrom_c] = mp[mfrom_c].map(normalize_id)
            mp[mto_c] = mp[mto_c].map(normalize_id)
            out = out.merge(
                mp.rename(columns={mfrom_c: "subject_id", mto_c: "sample_id_mapped"}),
                on="subject_id",
                how="left",
            )
            out["merge_key"] = out["sample_id_mapped"].where(
                out["sample_id_mapped"].notna() & (out["sample_id_mapped"] != ""),
                out["subject_id"],
            )
            meta["whi_sample_id_map"] = {"from": mfrom_c, "to": mto_c, "n_mapped": int(out["sample_id_mapped"].notna().sum())}

    meta["n_rows"] = int(len(out))
    meta["missingness"] = {c: float(out[c].isna().mean()) for c in out.columns if c not in ("cohort",)}
    return out, meta


def build_demo_lifestyle(
    fhs_pq: Path,
    whi_pq: Path,
    seed: int,
) -> pd.DataFrame:
    """Synthetic lifestyle correlated with age for pipeline smoke tests only."""
    rng = np.random.default_rng(seed)
    parts = []
    for path, id_col, cohort in [
        (fhs_pq, "Share_ID", "FHS"),
        (whi_pq, "sample_ID", "WHI"),
    ]:
        pf = _open_pq(path)
        cols = [id_col, "age", "sex"]
        df = pf.read(columns=[c for c in cols if c in pf.schema_arrow.names]).to_pandas()
        n = len(df)
        age = df["age"].to_numpy(dtype=np.float64)
        age_z = (age - np.nanmean(age)) / (np.nanstd(age) + 1e-6)
        smoke_p = 1 / (1 + np.exp(-(0.8 * age_z + rng.normal(0, 0.5, n))))
        status = np.where(smoke_p < 0.45, "never", np.where(smoke_p < 0.75, "former", "current"))
        parts.append(pd.DataFrame({
            "cohort": cohort,
            "subject_id": df[id_col].map(normalize_id),
            "merge_key": df[id_col].map(normalize_id),
            "smoking_status": status,
            "cigarettes_per_day": np.where(status == "current", np.clip(rng.poisson(8, n), 0, 40), 0),
            "pack_years": np.clip(age * smoke_p * 0.4 + rng.normal(0, 2, n), 0, 80),
            "alcohol_drinks_per_week": np.clip(3 + 2 * age_z + rng.normal(0, 4, n), 0, 50),
            "sleep_hours": np.clip(7.2 - 0.15 * age_z + rng.normal(0, 0.8, n), 4, 10),
            "sleep_quality": rng.integers(1, 6, n),
            "physical_activity": np.clip(10 - age_z * 2 + rng.normal(0, 3, n), 0, 30),
            "bmi": np.clip(26 + 0.3 * age_z + rng.normal(0, 3, n), 18, 45),
            "exam_year": np.nan,
            "_demo": True,
        }))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Harmonize dbGaP lifestyle exports.")
    p.add_argument("--map", type=str, default="bio_relevance/lifestyle_variable_map.json",
                   help="Variable map JSON (copy from .example.json).")
    p.add_argument("--map-example", action="store_true",
                   help="Use lifestyle_variable_map.example.json.")
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/lifestyle")
    p.add_argument("--demo", action="store_true",
                   help="Write synthetic lifestyle (pipeline test only).")
    p.add_argument("--fhs-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        df = build_demo_lifestyle(Path(args.fhs_parquet), Path(args.whi_parquet), args.seed)
        harm_doc = {
            "mode": "demo_synthetic",
            "warning": "Not for publication. Replace with dbGaP exports.",
            "n_rows": int(len(df)),
            "seed": args.seed,
        }
        df.drop(columns=["_demo"], errors="ignore").to_parquet(out_dir / "lifestyle_harmonized.parquet", index=False)
        (out_dir / "lifestyle_harmonization.json").write_text(json.dumps(harm_doc, indent=2), encoding="utf-8")
        print(f"Wrote demo {out_dir / 'lifestyle_harmonized.parquet'}  n={len(df)}")
        return

    map_path = Path("bio_relevance/lifestyle_variable_map.example.json" if args.map_example else args.map)
    if not map_path.exists():
        raise SystemExit(
            f"Map not found: {map_path}. Copy lifestyle_variable_map.example.json or use --demo."
        )
    vmap = load_variable_map(map_path)
    harm = vmap.get("harmonization", {})
    frames: List[pd.DataFrame] = []
    meta_all: Dict[str, Any] = {"map": str(map_path), "cohorts": {}}

    for key in ("fhs", "whi"):
        if key not in vmap:
            continue
        cfg = vmap[key]
        raw_path = Path(cfg["raw_csv"])
        if not raw_path.exists():
            print(f"  SKIP {key}: missing {raw_path}")
            continue
        raw = pd.read_csv(raw_path, low_memory=False)
        out, meta = _harmonize_cohort_raw(raw, cfg, harm)
        frames.append(out)
        meta_all["cohorts"][key] = meta
        print(f"  harmonized {key}: n={len(out)} from {raw_path}")

    if not frames:
        raise SystemExit("No raw lifestyle CSVs found. Export from dbGaP or use --demo.")

    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in CANONICAL_LIFESTYLE_COLS if c in df.columns]]
    df.to_parquet(out_dir / "lifestyle_harmonized.parquet", index=False)
    meta_all["n_rows"] = int(len(df))
    meta_all["missingness"] = {c: float(df[c].isna().mean()) for c in df.columns if c != "cohort"}
    (out_dir / "lifestyle_harmonization.json").write_text(json.dumps(meta_all, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'lifestyle_harmonized.parquet'}  n={len(df)}")


if __name__ == "__main__":
    main()
