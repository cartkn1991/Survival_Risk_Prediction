#!/usr/bin/env python3
"""Load joint_model_epoch26.pt for downstream analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aesurv_contrastive.eval_joint_checkpoint import build_model_from_joint_ckpt
from train_aesurv_joint_dann_contrastive import _resolve_device
from train_dann_survival import _setup_logger


def load_epoch26_model(device: str = "cuda", analysis_dir: Path | None = None):
    base = (analysis_dir or ROOT / "runs" / "aesurv_joint_epoch26_analysis").resolve()
    manifest_path = base / "model_manifest.json"
    if not manifest_path.exists():
        manifest_path = HERE / "model_manifest.example.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log = _setup_logger(None)
    dev = _resolve_device(device, log)
    ckpt = base / manifest["weights_file"]
    cfg_from = ROOT / "runs" / "aesurv_joint_dann_contrastive_best" / "joint_dann_aesurv_model.pt"
    model = build_model_from_joint_ckpt(
        ckpt,
        ROOT / manifest["dann_encoder_ckpt"],
        ROOT / manifest["dann_preprocess_npz"],
        dev,
        config_from=cfg_from,
    )
    return model, manifest, dev


if __name__ == "__main__":
    model, manifest, dev = load_epoch26_model()
    print(f"Loaded epoch {manifest['epoch']} on {dev}")
