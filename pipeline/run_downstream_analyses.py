#!/usr/bin/env python3
"""Re-run bio-relevance + feature-importance pipelines for joint epoch-26 checkpoint."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "runs" / "aesurv_joint_epoch26_analysis"
PY = sys.executable


def _resolve_analysis(path: Path | None) -> Path:
    analysis = (path or DEFAULT_ANALYSIS).resolve()
    if not analysis.is_absolute():
        analysis = (ROOT / analysis).resolve()
    return analysis


def run(cmd: str) -> int:
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.call(cmd, shell=True, cwd=str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-rfe", action="store_true")
    p.add_argument("--skip-gwas", action="store_true")
    p.add_argument("--skip-feature-importance", action="store_true")
    args = p.parse_args()

    ANALYSIS = _resolve_analysis(args.analysis_dir)
    JOINT = f"--joint-analysis-dir {ANALYSIS}"

    fi_root = ANALYSIS / "feature_importance"
    bio = ANALYSIS / "bio_relevance"
    lifestyle_out = bio / "lifestyle"
    clocks_out = bio / "clocks"
    fi_root.mkdir(parents=True, exist_ok=True)
    lifestyle_out.mkdir(parents=True, exist_ok=True)
    clocks_out.mkdir(parents=True, exist_ok=True)

    align_src = ROOT / "models" / "aesurv_final"
    align_dst = fi_root / "alignment"
    if align_src.is_dir() and not (align_dst / "feature_alignment.json").exists():
        align_dst.mkdir(parents=True, exist_ok=True)
        for name in (
            "feature_alignment.json", "feature_alignment_cpg.npy", "feature_alignment_snp.npy",
        ):
            src = align_src / name
            if src.exists():
                shutil.copy2(src, align_dst / name)

    ls_src = ROOT / "feature_importance" / "bio_relevance" / "lifestyle" / "lifestyle_harmonized.parquet"
    ls_dst = lifestyle_out / "lifestyle_harmonized.parquet"
    if ls_src.exists() and not ls_dst.exists():
        shutil.copy2(ls_src, ls_dst)
        for extra in ("lifestyle_harmonization.json",):
            e = ls_src.parent / extra
            if e.exists():
                shutil.copy2(e, lifestyle_out / extra)

    steps = [
        f'{PY} bio_relevance/attach_dann_risk.py {JOINT} --out-dir {lifestyle_out}',
        f'{PY} bio_relevance/epigenetic_clock_mortality_benchmark.py {JOINT} --out-dir {clocks_out} --device {args.device}',
        f'{PY} bio_relevance/clock_mortality_age_adjusted_cox.py --scores-dir {clocks_out} --out-dir {clocks_out}',
        f'{PY} bio_relevance/plot_clock_mortality_comparison.py --json {clocks_out}/clock_mortality_cindex.json --out-dir {clocks_out}',
        f'{PY} bio_relevance/lifestyle_risk_validation.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out} --fhs-only',
        f'{PY} bio_relevance/lifestyle_logh_linkage_figure.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out} --mode both',
        f'{PY} bio_relevance/lifestyle_confounding_figure.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out}',
        f'{PY} bio_relevance/lipids_glucose_phenotypic_plots.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-path {lifestyle_out}/figures/lipids_glucose_correlations.pdf --fhs-only',
        f'{PY} bio_relevance/lipid_glucose_km_clinical_relevance.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {lifestyle_out}/figures',
    ]

    if not args.skip_feature_importance:
        sig_out = fi_root / "significance"
        shap_out = fi_root / "shap"
        steps = [
            f'{PY} joint_feature_importance_aux.py --joint-analysis-dir {ANALYSIS} --out-dir {fi_root} --alignment-dir {align_dst if (align_dst / "feature_alignment.json").exists() else align_src}',
            f'{PY} select_significant_features.py --joint-analysis-dir {ANALYSIS} --out-dir {sig_out} --W-npy {fi_root}/dann_W.npy --alignment-dir {align_dst if (align_dst / "feature_alignment.json").exists() else align_src} --device {args.device}',
        ] + steps
        if not args.skip_shap:
            steps.insert(
                2,
                f'{PY} shap_feature_importance.py --joint-analysis-dir {ANALYSIS} --candidate-csv {sig_out}/significant_either.csv --out-dir {shap_out} --device {args.device}',
            )
            steps.extend([
                f'{PY} feature_importance/intersect_significant_shap.py --sig-csv {sig_out}/significant_either.csv --shap-dir {shap_out} --out-dir {shap_out}',
                f'{PY} annotate_shap_features.py --shap-dir {shap_out}',
                f'{PY} -m bio_relevance.pathway_enrichment --out-dir {bio}/pathway --shap-annot-cpg {shap_out}/annot/risk_top_cpg_annotated.csv --shap-annot-snp {shap_out}/annot/risk_top_snp_annotated.csv',
                f'{PY} -m bio_relevance.kegg_gsea --out-dir {bio}/kegg_gsea --shap-cpg {shap_out}/shap_all_cpg_risk.csv --shap-snp {shap_out}/shap_all_snp_risk.csv',
                f'{PY} bio_relevance/cell_chromatin.py {JOINT} --out-dir {bio}/cell_chromatin --shap-top-cpg {shap_out}/shap_top_cpg_risk.csv --device {args.device}',
                f'{PY} -m bio_relevance.negative_controls --out-dir {bio}/negative_controls --bundle-dir models/aesurv_final',
            ])
        if not args.skip_rfe:
            print("NOTE: RFE/ablation still use frozen-bundle masking; skipped for joint epoch26 (use significance/SHAP outputs).")

    if not args.skip_gwas:
        gwas_out = fi_root / "gwas"
        gwas_out.mkdir(parents=True, exist_ok=True)
        steps.extend([
            f'{PY} gwas/prepare_gwas_inputs.py --merged-parquet {lifestyle_out}/lifestyle_risk_merged.parquet --out-dir {gwas_out}',
        ])

    for cmd in steps:
        code = run(cmd)
        if code != 0:
            print(f"FAILED (exit {code}): {cmd}")
            return code

    (ANALYSIS / "downstream_run_complete.txt").write_text(
        "All downstream steps finished OK.\n", encoding="utf-8",
    )
    print("\nAll downstream analyses completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
