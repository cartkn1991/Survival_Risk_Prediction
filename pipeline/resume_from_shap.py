#!/usr/bin/env python3
"""Resume downstream after significance (SHAP + bio-relevance + GWAS prep)."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "runs" / "aesurv_joint_epoch26_analysis"
PY = sys.executable

p = argparse.ArgumentParser()
p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
p.add_argument("--device", default="cuda")
args = p.parse_args()
ANALYSIS = args.analysis_dir.resolve()
if not ANALYSIS.is_absolute():
    ANALYSIS = (ROOT / ANALYSIS).resolve()
JOINT = f"--joint-analysis-dir {ANALYSIS}"
fi_root = ANALYSIS / "feature_importance"
bio = ANALYSIS / "bio_relevance"
sig_out = fi_root / "significance"
shap_out = fi_root / "shap"
lifestyle_out = bio / "lifestyle"
clocks_out = bio / "clocks"

steps = [
    f"{PY} shap_feature_importance.py {JOINT} --candidate-csv {sig_out}/significant_either.csv --out-dir {shap_out} --device {args.device}",
    f"{PY} feature_importance/intersect_significant_shap.py --sig-csv {sig_out}/significant_either.csv --shap-dir {shap_out} --out-dir {shap_out}",
    f"{PY} annotate_shap_features.py --shap-dir {shap_out}",
    f"{PY} -m bio_relevance.pathway_enrichment --out-dir {bio}/pathway --shap-annot-cpg {shap_out}/annot/risk_top_cpg_annotated.csv --shap-annot-snp {shap_out}/annot/risk_top_snp_annotated.csv",
    f"{PY} -m bio_relevance.kegg_gsea --out-dir {bio}/kegg_gsea --shap-cpg {shap_out}/shap_all_cpg_risk.csv --shap-snp {shap_out}/shap_all_snp_risk.csv",
    f"{PY} bio_relevance/cell_chromatin.py {JOINT} --out-dir {bio}/cell_chromatin --shap-top-cpg {shap_out}/shap_top_cpg_risk.csv --device cuda",
    f"{PY} -m bio_relevance.negative_controls --out-dir {bio}/negative_controls",
    f"{PY} bio_relevance/epigenetic_clock_mortality_benchmark.py {JOINT} --out-dir {clocks_out} --device {args.device}",
    f"{PY} bio_relevance/clock_mortality_age_adjusted_cox.py --scores-dir {clocks_out} --out-dir {clocks_out}",
    f"{PY} bio_relevance/plot_clock_mortality_comparison.py --json {clocks_out}/clock_mortality_cindex.json --out-dir {clocks_out}",
    f"{PY} bio_relevance/lifestyle_risk_validation.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out} --fhs-only",
    f"{PY} bio_relevance/lifestyle_logh_linkage_figure.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out} --mode both",
    f"{PY} bio_relevance/lifestyle_confounding_figure.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out}",
    f"{PY} bio_relevance/lipids_glucose_phenotypic_plots.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-pdf {lifestyle_out}/figures/lipids_glucose_correlations.pdf",
    f"{PY} bio_relevance/lipid_glucose_km_clinical_relevance.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out}/figures",
    f"{PY} gwas/prepare_gwas_inputs.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {fi_root}/gwas",
]

for cmd in steps:
    print(f"\n>>> {cmd}\n", flush=True)
    if subprocess.call(cmd, shell=True, cwd=str(ROOT)) != 0:
        sys.exit(1)
print("Done.")
