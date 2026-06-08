#!/usr/bin/env python3
"""Launch joint DANN + AESurv contrastive training from JSON config."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train_aesurv_joint_dann_contrastive.py"
DEFAULT_CFG = Path(__file__).parent / "joint_default_config.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    p.add_argument("--out-dir", type=str, default="runs/aesurv_joint_dann_contrastive_a3")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cmd = [
        sys.executable, str(TRAIN),
        "--dann-encoder-ckpt", cfg["dann_encoder_ckpt"],
        "--dann-preprocess-npz", cfg["dann_preprocess_npz"],
        "--out-dir", args.out_dir,
        "--device", args.device,
        "--aux-age-weight", str(cfg.get("aux_age_weight", 12)),
        "--aux-cell-weight", str(cfg.get("aux_cell_weight", 0.5)),
        "--aux-contrast-weight", str(cfg.get("aux_contrast_weight", 1.0)),
        "--contrast-tau", str(cfg.get("contrast_tau", 0.07)),
        "--contrast-warmup-epochs", str(cfg.get("contrast_warmup_epochs", 10)),
        "--contrast-proj-dim", str(cfg.get("contrast_proj_dim", 64)),
        "--w-recon", str(cfg.get("w_recon", 0.5)),
        "--w-dom", str(cfg.get("w_dom", 0.6)),
        "--w-mmd", str(cfg.get("w_mmd", 5.0)),
        "--w-mmd-batch", str(cfg.get("w_mmd_batch", 2.0)),
        "--w-batch-adv", str(cfg.get("w_batch_adv", 1.0)),
        "--domain-loss-scale", str(cfg.get("domain_loss_scale", 0.6)),
        "--lr-dann", str(cfg.get("lr_dann", 1e-4)),
        "--lr-head", str(cfg.get("lr_head", 3e-4)),
        "--epochs", str(cfg.get("epochs", 80)),
        "--patience", str(cfg.get("patience", 25)),
    ]
    if cfg.get("init_head"):
        cmd.extend(["--init-head", cfg["init_head"]])
    if cfg.get("balance_events", True):
        cmd.append("--balance-events")

    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
