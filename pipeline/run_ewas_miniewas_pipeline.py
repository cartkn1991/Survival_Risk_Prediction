#!/usr/bin/env python3
"""Run full EWAS + mini-EWAS for epoch-26 joint model analysis."""
from __future__ import annotations

import argparse
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
    args = p.parse_args()

    ANALYSIS = _resolve_analysis(args.analysis_dir)
    FI = ANALYSIS / "feature_importance"
    RISK_PQ = ANALYSIS / "bio_relevance" / "lifestyle" / "lifestyle_risk_merged.parquet"
    SIG_CSV = FI / "significance" / "significant_risk.csv"
    EWAS_OUT = FI / "ewas"
    MINI_OUT = FI / "miniewas"
    EWAS_OUT.mkdir(parents=True, exist_ok=True)
    MINI_OUT.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            f'{PY} gwas/run_full_ewas_pipeline.py --stage all '
            f'--out-dir {EWAS_OUT} '
            f'--risk-parquet {RISK_PQ}'
        ),
        (
            f'{PY} miniewas_outliers.py '
            f'--sig-csv {SIG_CSV} '
            f'--sig-age-csv {FI / "significance" / "significant_age.csv"} '
            f'--risk-parquet {RISK_PQ} '
            f'--out-dir {MINI_OUT}'
        ),
        f'{PY} {FI / "ewas" / "finalize_ewas_outputs.py"}',
        f'{PY} {FI / "compare_ewas_miniewas_markers.py"}',
        f'{PY} {FI / "miniewas" / "whi_replication" / "run_cpg_cox_replication.py"}',
    ]

    for cmd in steps:
        code = run(cmd)
        if code != 0:
            print(f"FAILED: {cmd}")
            return code

    (ANALYSIS / "ewas_miniewas_complete.txt").write_text("EWAS + mini-EWAS pipeline finished OK.\n", encoding="utf-8")
    print("\nEWAS + mini-EWAS pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
