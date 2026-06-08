#!/usr/bin/env python3
"""Build per-sample table: DANN-Aux log_h + survival + IDs + lifestyle merge.

Outputs:
  feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet
  feature_importance/bio_relevance/lifestyle/merge_qc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bio_relevance.lifestyle_common import normalize_id


_THRIFT_LIMIT = 2_147_483_647


def _open_pq(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(str(path), thrift_string_size_limit=_THRIFT_LIMIT,
                              thrift_container_size_limit=_THRIFT_LIMIT)
    except TypeError:
        return pq.ParquetFile(str(path))


def _meta_columns(pf: pq.ParquetFile) -> List[str]:
    names = set(pf.schema_arrow.names)
    want = [
        "Share_ID", "Patient_ID", "IID", "sample_ID", "SUBJID", "dbGaP_Subject_ID",
        "SAMPLE_ID", "age", "sex", "time", "event", "batch",
    ]
    return [c for c in want if c in names]


def load_cohort_meta(parquet_path: Path, cohort: str) -> pd.DataFrame:
    pf = _open_pq(parquet_path)
    cols = _meta_columns(pf)
    df = pf.read(columns=cols).to_pandas()
    df["cohort"] = cohort
    if cohort == "FHS":
        df["primary_id"] = df["Share_ID"].map(normalize_id) if "Share_ID" in df.columns else df["IID"].map(normalize_id)
    else:
        df["primary_id"] = df["sample_ID"].map(normalize_id) if "sample_ID" in df.columns else ""
    for c in ["Share_ID", "IID", "sample_ID", "SUBJID", "Patient_ID", "SAMPLE_ID", "dbGaP_Subject_ID"]:
        if c in df.columns:
            df[f"id_{c}"] = df[c].map(normalize_id)
    return df


def attach_log_h(
    df: pd.DataFrame,
    log_h: np.ndarray,
    age_pred: np.ndarray,
    n_fhs: int,
) -> pd.DataFrame:
    if len(log_h) != len(df):
        raise ValueError(f"log_h length {len(log_h)} != meta rows {len(df)}")
    out = df.copy()
    out["log_h"] = log_h.astype(np.float64)
    out["age_pred"] = age_pred.astype(np.float64)
    out["is_fhs"] = (np.arange(len(df)) < n_fhs).astype(np.int8)
    return out


def merge_lifestyle_v2(base: pd.DataFrame, lifestyle: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
    """Merge lifestyle per cohort using best ID key."""
    qc: Dict = {"attempts": [], "final_merge_rate": {}}
    lifestyle_cols = [
        c for c in lifestyle.columns
        if c not in ("cohort", "subject_id", "merge_key")
    ]
    out = base.copy()
    for c in lifestyle_cols:
        out[c] = pd.Series([np.nan] * len(out), index=out.index, dtype=object if c == "smoking_status" else float)
    out["_has_lifestyle"] = False

    for cohort in ("FHS", "WHI"):
        b_mask = out["cohort"] == cohort
        b = out.loc[b_mask].copy()
        l = lifestyle[lifestyle["cohort"] == cohort].copy()
        if b.empty or l.empty:
            continue
        l_idx = l.set_index("merge_key")
        best_rate, best_key, best_series = 0.0, None, {}

        if cohort == "FHS":
            key_map = [
                ("primary_id", "merge_key"),
                ("id_Share_ID", "merge_key"),
                ("id_IID", "merge_key"),
            ]
        else:
            key_map = [
                ("primary_id", "merge_key"),
                ("id_sample_ID", "merge_key"),
                ("id_SUBJID", "merge_key"),
                ("id_dbGaP_Subject_ID", "merge_key"),
                ("id_SAMPLE_ID", "merge_key"),
            ]

        for left_col, _ in key_map:
            if left_col not in b.columns:
                continue
            keys = b[left_col].astype(str)
            n_hit = int(keys.isin(l_idx.index).sum())
            rate = float(n_hit) / max(len(b), 1)
            qc["attempts"].append({"cohort": cohort, "left_col": left_col, "merge_rate": rate})
            if rate > best_rate:
                best_rate = rate
                best_key = left_col
                for lc in lifestyle_cols:
                    if lc in l_idx.columns:
                        best_series[lc] = keys.map(l_idx[lc])
                    else:
                        best_series[lc] = np.nan

        qc["final_merge_rate"][cohort] = {"key": best_key, "rate": best_rate, "n": int(b_mask.sum())}
        for lc in lifestyle_cols:
            if lc in best_series:
                out.loc[b_mask, lc] = best_series[lc].to_numpy()
        if "smoking_status" in best_series:
            out.loc[b_mask, "_has_lifestyle"] = out.loc[b_mask, "smoking_status"].notna().to_numpy()

    qc["overall_lifestyle_rate"] = float(out["_has_lifestyle"].mean())
    return out, qc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/lifestyle")
    p.add_argument("--lifestyle-parquet", type=str,
                   default="feature_importance/bio_relevance/lifestyle/lifestyle_harmonized.parquet")
    p.add_argument("--fhs-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--bundle-dir", type=str, default="models/aesurv_final")
    p.add_argument("--proj-fhs-npz", type=str, default="vae_cox_cache/jl_proj/proj_FHS_b24e34c31a.npz")
    p.add_argument("--proj-whi-npz", type=str, default="vae_cox_cache/jl_proj/proj_WHI_b159d5bf88.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--joint-analysis-dir",
        type=str,
        default=None,
        help="Use joint epoch checkpoint from this folder (e.g. runs/aesurv_joint_epoch26_analysis)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fhs = load_cohort_meta(Path(args.fhs_parquet), "FHS")
    whi = load_cohort_meta(Path(args.whi_parquet), "WHI")
    base = pd.concat([fhs, whi], ignore_index=True)
    n_fhs = len(fhs)

    import torch
    from bio_relevance.model_scores import pooled_predictions

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    joint_dir = Path(args.joint_analysis_dir) if args.joint_analysis_dir else None
    log_h, age_pred, nf, nw = pooled_predictions(
        Path(args.bundle_dir),
        Path(args.proj_fhs_npz),
        Path(args.proj_whi_npz),
        device,
        joint_analysis_dir=joint_dir,
    )
    assert nf == n_fhs and nw == len(whi)
    merged_base = attach_log_h(base, log_h, age_pred, n_fhs)

    ls_path = Path(args.lifestyle_parquet)
    if ls_path.exists():
        lifestyle = pd.read_parquet(ls_path)
        merged, qc = merge_lifestyle_v2(merged_base, lifestyle)
    else:
        print(f"  WARNING: no {ls_path}; writing risk table without lifestyle.")
        merged = merged_base
        qc = {"note": "lifestyle parquet missing"}

    merged.to_parquet(out_dir / "lifestyle_risk_merged.parquet", index=False)
    qc["n_total"] = int(len(merged))
    qc["n_fhs"] = n_fhs
    qc["n_whi"] = int(nw)
    (out_dir / "merge_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'lifestyle_risk_merged.parquet'}  n={len(merged)}")
    print(f"  merge_qc: {json.dumps(qc['final_merge_rate'], indent=2) if 'final_merge_rate' in qc else qc}")


if __name__ == "__main__":
    main()
