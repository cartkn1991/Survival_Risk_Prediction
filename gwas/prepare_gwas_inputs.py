#!/usr/bin/env python3
"""Prepare phenotype and covariate files for GWAS and survival GWAS.

Inputs
------
- feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet
    Built by bio_relevance/attach_dann_risk.py

Outputs (under gwas/)
---------------------
- gwas/fhs_phenotypes.tsv
- gwas/fhs_covariates.tsv
- gwas/whi_phenotypes.tsv
- gwas/whi_covariates.tsv

These files are suitable for:
- Linear GWAS on AESurv log_h and cardio-metabolic traits
- Cox survival GWAS on mortality (using external tools such as R/survival or SAIGE-survival)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _pick_id_cols(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (FID, IID) for PLINK-style GWAS."""
    # Prefer IID-style column when available; fall back to Share_ID / sample_ID.
    if "id_IID" in df.columns:
        iid = df["id_IID"].astype(str)
    elif "IID" in df.columns:
        iid = df["IID"].astype(str)
    elif "id_Share_ID" in df.columns:
        iid = df["id_Share_ID"].astype(str)
    elif "Share_ID" in df.columns:
        iid = df["Share_ID"].astype(str)
    elif "id_sample_ID" in df.columns:
        iid = df["id_sample_ID"].astype(str)
    elif "sample_ID" in df.columns:
        iid = df["sample_ID"].astype(str)
    else:
        raise SystemExit("Could not find an ID column suitable for IID.")
    fid = iid  # family ID not distinguished here
    return fid, iid


def _write_tsv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}  n={len(df)}")


def build_gwas_inputs(merged_parquet: Path, out_dir: Path) -> None:
    df = pd.read_parquet(merged_parquet)

    # Common columns across cohorts
    base_cols = ["cohort", "time", "event", "age", "sex", "log_h"]
    trait_cols = ["ldl", "hdl", "total_cholesterol", "triglycerides", "glucose"]

    for cohort in ("FHS", "WHI"):
        sub = df[df["cohort"].astype(str) == cohort].copy()
        if sub.empty:
            print(f"  No rows for cohort {cohort}; skipping.")
            continue

        fid, iid = _pick_id_cols(sub)

        pheno_cols = []
        for c in base_cols + trait_cols:
            if c in sub.columns:
                pheno_cols.append(c)

        pheno = pd.DataFrame({"FID": fid, "IID": iid})
        for c in pheno_cols:
            pheno[c] = sub[c]

        cov_cols = ["age", "sex"]
        if "batch" in sub.columns and cohort == "FHS":
            cov_cols.append("batch")
        cov = pd.DataFrame({"FID": fid, "IID": iid})
        for c in cov_cols:
            if c in sub.columns:
                cov[c] = sub[c]

        tag = cohort.lower()
        _write_tsv(pheno, out_dir / f"{tag}_phenotypes.tsv")
        _write_tsv(cov, out_dir / f"{tag}_covariates.tsv")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--merged-parquet",
        type=str,
        default="feature_importance/bio_relevance/lifestyle/lifestyle_risk_merged.parquet",
        help="Output of bio_relevance/attach_dann_risk.py",
    )
    p.add_argument("--out-dir", type=str, default="gwas")
    args = p.parse_args()

    build_gwas_inputs(Path(args.merged_parquet), Path(args.out_dir))


if __name__ == "__main__":
    main()

