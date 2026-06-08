#!/usr/bin/env python3
"""Run FHS + WHI Cox replication for moderate-tier mini-EWAS CpG markers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPL = Path(__file__).resolve().parent
MINI = REPL.parent
PY = sys.executable
MARKERS = MINI / "discovered_cpg_markers_moderate.csv"


def run(cmd: str) -> int:
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.call(cmd, shell=True, cwd=str(ROOT))


def main() -> int:
    REPL.mkdir(parents=True, exist_ok=True)
    rel_repl = REPL.relative_to(ROOT)
    rel_markers = MARKERS.relative_to(ROOT)

    steps = [
        (
            f'{PY} gwas/run_fhs_cpg_cox.py --cohort FHS '
            f'--cpg-csv {rel_markers} --out-dir {rel_repl}'
        ),
        (
            f'{PY} gwas/run_fhs_cpg_cox.py --cohort WHI '
            f'--cpg-csv {rel_markers} --out-dir {rel_repl} '
            f'--merge-fhs-csv {rel_repl / "fhs_cpg_cox_results.csv"}'
        ),
        f'{PY} {rel_repl / "make_manuscript_table.py"}',
    ]
    for cmd in steps:
        code = run(cmd)
        if code != 0:
            print(f"FAILED: {cmd}")
            return code
    print("\nCpG Cox WHI replication completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
