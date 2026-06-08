#!/usr/bin/env python3
"""AESURV training pipeline entry point.

Stages:
  1. dann     — pretrain domain-adversarial encoder (mini_vae_dann_pipeline.py)
  2. head     — train frozen-DANN AESURV contrastive head
  3. joint    — end-to-end joint DANN + AESURV training (recommended final model)
  4. all      — dann → head → joint (long run; requires data + GPU)

Example::

    export PYTHONPATH=.
    python run_train.py --stage joint --device cuda

See training/README.md for details.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PY = sys.executable
TRAIN_CFG = REPO_ROOT / "training" / "joint_default_config.json"


def run(cmd: str) -> int:
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.call(cmd, shell=True, cwd=str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="AESURV training pipeline")
    p.add_argument(
        "--stage",
        choices=["dann", "head", "joint", "all"],
        default="joint",
        help="Training stage (default: joint)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "aesurv_joint_dann_contrastive")
    p.add_argument("--config", type=Path, default=TRAIN_CFG)
    p.add_argument("--dann-out", type=Path, default=REPO_ROOT / "runs" / "mini_vae_dann_mmd_rich")
    p.add_argument("--head-out", type=Path, default=REPO_ROOT / "runs" / "aesurv_contrastive_best")
    args = p.parse_args()

    steps: list[str] = []

    if args.stage in ("dann", "all"):
        steps.append(
            f"{PY} mini_vae_dann_pipeline.py "
            f"--out-dir {args.dann_out} --device {args.device}"
        )

    if args.stage in ("head", "all"):
        steps.append(
            f"{PY} train_aesurv_dann_latent_aux_contrastive.py "
            f"--dann-encoder-ckpt {args.dann_out}/mini_dann_model.pt "
            f"--dann-preprocess-npz {args.dann_out}/mini_dann_preprocess.npz "
            f"--aux-age-weight 12 --aux-cell-weight 0.5 "
            f"--aux-contrast-weight 1.0 --contrast-tau 0.07 "
            f"--z-dim 8 --seed 42 --balance-events "
            f"--device {args.device} --epochs 200 --patience 25 "
            f"--out-dir {args.head_out}"
        )

    if args.stage in ("joint", "all"):
        steps.append(
            f"{PY} training/run_joint_train.py "
            f"--config {args.config} --out-dir {args.out_dir} --device {args.device}"
        )

    for cmd in steps:
        if run(cmd) != 0:
            print(f"FAILED: {cmd}")
            return 1

    print(f"\nTraining stage '{args.stage}' complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
