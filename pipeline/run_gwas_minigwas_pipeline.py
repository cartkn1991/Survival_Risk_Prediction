#!/usr/bin/env python3
"""Run miniGWAS + tiered GWAS comparison + WHI Cox SNP replication."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = REPO_ROOT / "runs" / "aesurv_joint_epoch26_analysis"
PY = sys.executable


def run(cmd: str) -> int:
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.call(cmd, shell=True, cwd=str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    p.add_argument("--gwas-dir", type=Path, default=None, help="Full GWAS results (default: repo feature_importance/gwas)")
    args = p.parse_args()

    analysis = args.analysis_dir.resolve()
    fi = analysis / "feature_importance"
    mini_out = fi / "minigwas"
    repl_out = mini_out / "whi_replication"
    gwas_dir = (args.gwas_dir or REPO_ROOT / "feature_importance" / "gwas").resolve()
    risk_pq = analysis / "bio_relevance" / "lifestyle" / "lifestyle_risk_merged.parquet"
    sig_csv = fi / "significance" / "significant_risk.csv"

    mini_out.mkdir(parents=True, exist_ok=True)
    repl_out.mkdir(parents=True, exist_ok=True)

    compare = fi / "minigwas" / "compare_gwas_minigwas_markers.py"
    if not compare.exists():
        compare = REPO_ROOT / "final" / "scripts" / "minigwas" / "compare_gwas_minigwas_markers.py"

    steps = [
        (
            f'{PY} minigwas_outliers.py '
            f'--sig-csv {sig_csv} '
            f'--sig-age-csv {fi / "significance" / "significant_age.csv"} '
            f'--risk-parquet {risk_pq} '
            f'--out-dir {mini_out}'
        ),
        (
            f'{PY} {compare} '
            f'--mortality-csv {gwas_dir / "mortality_fhs_results.csv"} '
            f'--logh-csv {gwas_dir / "discovery_fhs_results.csv"} '
            f'--out-dir {mini_out}'
        ),
        (
            f'{PY} gwas/run_fhs_snp_cox.py --cohort FHS '
            f'--snp-csv {mini_out / "discovered_markers_moderate.csv"} '
            f'--out-dir {repl_out}'
        ),
        (
            f'{PY} gwas/run_fhs_snp_cox.py --cohort WHI '
            f'--snp-csv {mini_out / "discovered_markers_moderate.csv"} '
            f'--out-dir {repl_out} '
            f'--merge-fhs-csv {repl_out / "fhs_snp_cox_results.csv"}'
        ),
        f'{PY} {repl_out / "make_manuscript_table.py"}',
    ]

    for cmd in steps:
        code = run(cmd)
        if code != 0:
            print(f"FAILED: {cmd}")
            return code

    (analysis / "gwas_minigwas_complete.txt").write_text("miniGWAS + WHI replication finished OK.\n", encoding="utf-8")
    print("\nGWAS / miniGWAS pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
