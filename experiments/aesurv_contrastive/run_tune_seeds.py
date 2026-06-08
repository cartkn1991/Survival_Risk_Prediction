#!/usr/bin/env python3
"""Train best contrastive config across multiple seeds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aesurv_contrastive.ensemble_whi_risk import resolve_torch_device
DANN_CKPT = ROOT / "runs/mini_vae_dann_mmd_rich/mini_dann_model.pt"
DANN_NPZ = ROOT / "runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz"
BASELINE_INIT = ROOT / "experiments/aesurv_risk_recon/checkpoints_baseline/aesurv_aux_model.pt"
TRAIN = ROOT / "train_aesurv_dann_latent_aux_contrastive.py"
DEFAULT_SEEDS = (0, 1, 2, 42, 123)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="runs/aesurv_contrastive_tune/best_config.json")
    p.add_argument("--out-root", type=str, default="runs/aesurv_contrastive_ensemble")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    device = str(resolve_torch_device(args.device))
    if device != args.device.strip().lower() and not args.device.strip().lower().startswith("cpu"):
        print(f"Note: --device {args.device!r} -> {device!r} (CUDA unavailable)")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    w = cfg.get("w", cfg.get("aux_contrast_weight", 1.0))
    tau = cfg.get("tau", cfg.get("contrast_tau", 0.07))
    warmup = cfg.get("warmup", cfg.get("contrast_warmup_epochs", 10))
    proj = cfg.get("proj", cfg.get("contrast_proj_dim", 64))
    age_w = cfg.get("aux_age_weight", 12)
    cell_w = cfg.get("aux_cell_weight", 1)
    init = cfg.get("init_from") or str(BASELINE_INIT)

    out_root = Path(args.out_root)
    py = sys.executable
    ckpts = []

    for seed in args.seeds:
        out_dir = out_root / f"seed_{seed}"
        cmd = [
            py, str(TRAIN),
            "--dann-encoder-ckpt", str(DANN_CKPT),
            "--dann-preprocess-npz", str(DANN_NPZ),
            "--init-from", init,
            "--aux-age-weight", str(age_w),
            "--aux-cell-weight", str(cell_w),
            "--aux-contrast-weight", str(w),
            "--contrast-tau", str(tau),
            "--contrast-warmup-epochs", str(warmup),
            "--contrast-proj-dim", str(proj),
            "--z-dim", "8", "--seed", str(seed), "--balance-events",
            "--device", device, "--epochs", str(args.epochs),
            "--out-dir", str(out_dir),
        ]
        print("seed", seed)
        if args.dry_run:
            print(" ", " ".join(cmd))
            ckpts.append(str(out_dir / "aesurv_aux_contrastive_model.pt"))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "train.log").open("w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        ckpts.append(str(out_dir / "aesurv_aux_contrastive_model.pt"))

    if not args.dry_run and ckpts:
        ens = [
            py, str(Path(__file__).parent / "ensemble_whi_risk.py"),
            "--checkpoints", *ckpts,
            "--out-json", str(out_root / "ensemble_summary.json"),
            "--device", device,
        ]
        subprocess.run(ens, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
