#!/usr/bin/env python3
"""AESURV full downstream analysis pipeline."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "runs" / "aesurv_joint_epoch26_analysis"
PY = sys.executable


def run(cmd: str) -> int:
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.call(cmd, shell=True, cwd=str(REPO_ROOT))


def _ensure_analysis_layout(analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mappings = [
        (SCRIPTS_DIR / "compare_ewas_miniewas_markers.py", analysis_dir / "feature_importance" / "compare_ewas_miniewas_markers.py"),
        (SCRIPTS_DIR / "ewas" / "finalize_ewas_outputs.py", analysis_dir / "feature_importance" / "ewas" / "finalize_ewas_outputs.py"),
        (SCRIPTS_DIR / "minigwas" / "compare_gwas_minigwas_markers.py", analysis_dir / "feature_importance" / "minigwas" / "compare_gwas_minigwas_markers.py"),
        (SCRIPTS_DIR / "minigwas" / "whi_replication" / "make_manuscript_table.py", analysis_dir / "feature_importance" / "minigwas" / "whi_replication" / "make_manuscript_table.py"),
        (SCRIPTS_DIR / "miniewas" / "whi_replication" / "make_manuscript_table.py", analysis_dir / "feature_importance" / "miniewas" / "whi_replication" / "make_manuscript_table.py"),
        (SCRIPTS_DIR / "miniewas" / "whi_replication" / "run_cpg_cox_replication.py", analysis_dir / "feature_importance" / "miniewas" / "whi_replication" / "run_cpg_cox_replication.py"),
        (SCRIPTS_DIR / "miniewas" / "whi_replication" / "README.md", analysis_dir / "feature_importance" / "miniewas" / "whi_replication" / "README.md"),
    ]
    for src, dst in mappings:
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for name, src_name in (("model_manifest.json", "model_manifest.example.json"), ("load_model.py", "load_model.py")):
        dst = analysis_dir / name
        if not dst.exists():
            src = CONFIG_DIR / src_name
            if src.exists():
                shutil.copy2(src, dst)


def main() -> int:
    p = argparse.ArgumentParser(description="AESURV downstream analysis pipeline")
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument(
        "--stage",
        nargs="+",
        choices=["all", "downstream", "gwas", "ewas", "figures"],
        default=["all"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-rfe", action="store_true")
    p.add_argument("--skip-gwas-inputs", action="store_true")
    args = p.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    if not analysis_dir.is_absolute():
        analysis_dir = (REPO_ROOT / analysis_dir).resolve()
    _ensure_analysis_layout(analysis_dir)

    stages = set(args.stage)
    if "all" in stages:
        stages = {"downstream", "gwas", "ewas", "figures"}

    rel = analysis_dir.relative_to(REPO_ROOT) if analysis_dir.is_relative_to(REPO_ROOT) else analysis_dir
    common = f'--analysis-dir "{rel}" --device {args.device}'

    steps: list[str] = []
    if "downstream" in stages:
        flags = ""
        if args.skip_gwas_inputs:
            flags += " --skip-gwas"
        if args.skip_shap:
            flags += " --skip-shap"
        if args.skip_rfe:
            flags += " --skip-rfe"
        steps.append(f'{PY} pipeline/run_downstream_analyses.py {common}{flags}')
    if "gwas" in stages:
        steps.append(f"{PY} pipeline/run_gwas_minigwas_pipeline.py {common}")
    if "ewas" in stages:
        steps.append(f"{PY} pipeline/run_ewas_miniewas_pipeline.py {common}")
    if "figures" in stages:
        steps.append(f"{PY} pipeline/make_logh_age_tdauroc_plots.py {common}")

    for cmd in steps:
        if run(cmd) != 0:
            print(f"FAILED: {cmd}")
            return 1

    (analysis_dir / "pipeline_complete.txt").write_text(
        f"AESURV analysis pipeline finished OK.\nstages={sorted(stages)}\n",
        encoding="utf-8",
    )
    print(f"\nPipeline complete. Outputs: {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
