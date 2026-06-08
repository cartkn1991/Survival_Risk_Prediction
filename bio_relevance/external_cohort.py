#!/usr/bin/env python3
"""Cross-cohort importance agreement (FHS vs WHI) as a proxy for external reproducibility.

Reads ``feature_importance/risk_attrib.npz`` (mean-gradient back-projection per cohort)
and optionally ``feature_importance/shap/shap_cohort_compare_risk.csv``.

Writes ``feature_importance/bio_relevance/external_cohort/``:

  gradient_fhs_vs_whi.json   Spearman/Pearson on |grad| for CpG and SNP blocks
  shap_cohort_summary.json   copies key stats from SHAP cohort CSV if present
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/external_cohort")
    p.add_argument("--attrib-npz", type=str, default="feature_importance/risk_attrib.npz")
    p.add_argument("--shap-cohort-csv", type=str, default="feature_importance/shap/shap_cohort_compare_risk.csv")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summ = {}
    ap = Path(args.attrib_npz)
    if ap.exists():
        z = np.load(ap, allow_pickle=False)
        gf = z["grad_raw_fhs"].astype(np.float64)
        gw = z["grad_raw_whi"].astype(np.float64)
        n_cpg = int(z["n_cpg"])
        def block_stats(a, b, name):
            aa, bb = np.abs(a), np.abs(b)
            m = np.isfinite(aa) & np.isfinite(bb)
            rp, _ = pearsonr(aa[m], bb[m])
            rs, _ = spearmanr(aa[m], bb[m])
            return {f"{name}_pearson_abs": float(rp), f"{name}_spearman_abs": float(rs)}
        summ["gradient_risk"] = {
            **block_stats(gf[:n_cpg], gw[:n_cpg], "cpg"),
            **block_stats(gf[n_cpg:], gw[n_cpg:], "snp"),
        }
    else:
        summ["gradient_risk"] = {"error": f"missing {ap}"}

    sc = Path(args.shap_cohort_csv)
    if sc.exists():
        df = pd.read_csv(sc, low_memory=False)
        summ["shap_risk_cohort"] = {
            "n_features": int(len(df)),
            "pearson_mean_abs_fhs_vs_whi": float(pearsonr(
                df["mean_abs_fhs"], df["mean_abs_whi"])[0]),
            "spearman_mean_abs_fhs_vs_whi": float(spearmanr(
                df["mean_abs_fhs"], df["mean_abs_whi"])[0]),
        }
    else:
        summ["shap_risk_cohort"] = {"error": f"missing {sc}"}

    (out / "gradient_fhs_vs_whi.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"  wrote {out / 'gradient_fhs_vs_whi.json'}")


if __name__ == "__main__":
    main()
