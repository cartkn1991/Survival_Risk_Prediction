#!/usr/bin/env python3
"""Generate EWAS plots + cross-compare from existing result CSVs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from gwas.run_full_ewas_pipeline import _plot_manhattan, _plot_qq, run_cross_compare
from gwas.gwas_common import genomic_lambda, write_json

out_dir = Path(__file__).resolve().parent
mort = pd.read_csv(out_dir / "mortality_fhs_ewas_results.csv", usecols=["cpg", "beta", "p", "chrom", "pos", "gene"])
disc = pd.read_csv(out_dir / "discovery_fhs_ewas_results.csv", usecols=["cpg", "beta", "p", "chrom", "pos", "gene"])

def _qq_sample(pvals, max_n: int = 50_000):
    p = pvals[np.isfinite(pvals)]
    if len(p) <= max_n:
        return p
    return np.random.default_rng(42).choice(p, size=max_n, replace=False)

_plot_qq(_qq_sample(mort["p"].to_numpy()), out_dir / "mortality_fhs_ewas_qq.png",
         genomic_lambda(mort["p"].to_numpy()), "FHS mortality EWAS QQ")
_plot_qq(_qq_sample(disc["p"].to_numpy()), out_dir / "discovery_fhs_ewas_qq.png",
         genomic_lambda(disc["p"].to_numpy()), "FHS discovery EWAS QQ")
# Lightweight Manhattan: top 5000 hits only
for label, df, png in (
    ("mortality", mort, "mortality_fhs_ewas_manhattan.png"),
    ("discovery", disc, "discovery_fhs_ewas_manhattan.png"),
):
    _plot_manhattan(df.nsmallest(5000, "p"), out_dir / png, f"FHS {label} EWAS (top 5000 CpGs)", 1e-5)
run_cross_compare(mort, disc, out_dir, p_cut=0.05)
print("finalized EWAS plots and cross-compare")
