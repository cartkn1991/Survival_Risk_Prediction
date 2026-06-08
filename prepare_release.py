#!/usr/bin/env python3
"""Sync core library code from the local SNP_datasets workspace into this repo.

Run from repo root after cloning:

    python prepare_release.py

This copies training + analysis implementation modules that the orchestrators
(`run_pipeline.py`, `run_train.py`) invoke. Large data, checkpoints, and run
outputs are never copied.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
# Parent workspace (adjust if your checkout layout differs)
SRC = REPO.parent if (REPO.parent / "train_aesurv_joint_dann_contrastive.py").exists() else REPO

DIRS = [
    "bio_relevance",
    "gwas",
    "experiments/aesurv_contrastive",
    "feature_importance",
    "external_phenotypes",
    "environment",
]

FILES = [
    "aesurv_domain.py",
    "mini_vae_dann_pipeline.py",
    "train_aesurv_joint_dann_contrastive.py",
    "train_aesurv_dann_latent_aux_contrastive.py",
    "train_aesurv_dann_latent_aux.py",
    "train_dann_survival.py",
    "train_vae_cox_lite.py",
    "joint_feature_importance_aux.py",
    "select_significant_features.py",
    "shap_feature_importance.py",
    "annotate_shap_features.py",
    "minigwas_outliers.py",
    "miniewas_outliers.py",
    "plot_aesurv_diagnostics.py",
    "joint_feature_importance_aux.py",
]

SKIP_DIR_PATTERNS = (
    "__pycache__",
    ".git",
    "runs",
    "analysis_out",
    "logs",
    "vae_cox_cache",
)

SKIP_FILE_PATTERNS = (
    ".pt",
    ".npz",
    ".parquet",
    ".pq",
    ".npy",
    ".csv",
    ".tsv",
    ".pdf",
    ".png",
    ".jpg",
    ".log",
)


def _should_skip(path: Path) -> bool:
    name = path.name
    if any(p in path.parts for p in SKIP_DIR_PATTERNS):
        return True
    return any(name.endswith(ext) for ext in SKIP_FILE_PATTERNS)


def copy_tree(rel: str) -> None:
    src = SRC / rel
    dst = REPO / rel
    if not src.exists():
        print(f"  skip missing: {rel}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _ignore(directory: str, names: list[str]) -> list[str]:
        ignored = []
        for n in names:
            p = Path(directory) / n
            if _should_skip(p):
                ignored.append(n)
        return ignored

    shutil.copytree(src, dst, ignore=_ignore)
    print(f"  copied dir: {rel}")


def copy_file(rel: str) -> None:
    src = SRC / rel
    dst = REPO / rel
    if not src.exists():
        print(f"  skip missing: {rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied file: {rel}")


def main() -> None:
    print(f"Source workspace: {SRC}")
    print(f"Target repo:    {REPO}\n")
    for d in DIRS:
        copy_tree(d)
    for f in FILES:
        copy_file(f)
    print("\nDone. Review with `git status` before commit.")


if __name__ == "__main__":
    main()
