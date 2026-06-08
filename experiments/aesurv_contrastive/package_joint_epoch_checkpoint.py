#!/usr/bin/env python3
"""Copy a joint epoch checkpoint + metadata into a self-contained analysis folder."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_vae_cox_lite import _resolve_data_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-ckpt", type=str, required=True)
    p.add_argument("--config-from", type=str, required=True, help="joint_dann_aesurv_model.pt with architecture config")
    p.add_argument("--parent-run-dir", type=str, required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--dann-encoder-ckpt", type=str, default="runs/mini_vae_dann_mmd_rich/mini_dann_model.pt")
    p.add_argument("--dann-preprocess-npz", type=str, default="runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz")
    args = p.parse_args()

    out = _resolve_data_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = _resolve_data_path(args.source_ckpt)
    cfg_src = _resolve_data_path(args.config_from)
    parent = _resolve_data_path(args.parent_run_dir)
    epoch = int(args.epoch)

    dest_ckpt = out / f"joint_model_epoch{epoch}.pt"
    shutil.copy2(src, dest_ckpt)

    parent_cfg = torch.load(cfg_src, map_location="cpu", weights_only=False)
    whi_raw = torch.load(src, map_location="cpu", weights_only=False)

    manifest = {
        "model_role": "exploratory_whi_top1",
        "selection_note": (
            "Checkpoint from joint training WHI top-k (epoch 26). "
            "Not the FHS-val primary model; chosen for downstream WHI-focused analysis."
        ),
        "epoch": epoch,
        "source_run": str(parent),
        "source_checkpoint": str(src),
        "weights_file": str(dest_ckpt.name),
        "dann_encoder_ckpt": str(_resolve_data_path(args.dann_encoder_ckpt)),
        "dann_preprocess_npz": str(_resolve_data_path(args.dann_preprocess_npz)),
        "architecture_config": parent_cfg.get("config", {}),
        "whi_cindex_at_save": whi_raw.get("whi_cindex"),
        "exploratory_flag": whi_raw.get("exploratory", True),
    }
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics_line = None
    metrics_path = parent / "joint_dann_aesurv_metrics.jsonl"
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if int(row.get("epoch", -1)) == epoch:
                metrics_line = row
                break
    if metrics_line:
        (out / f"training_log_epoch{epoch}.json").write_text(json.dumps(metrics_line, indent=2), encoding="utf-8")

    run_meta = parent / "joint_dann_aesurv_run_meta.json"
    if run_meta.is_file():
        shutil.copy2(run_meta, out / "parent_run_meta.json")

    load_cmd = f"""conda activate snp_torch
cd {ROOT}
$env:PYTHONPATH="{ROOT}"

python experiments/aesurv_contrastive/eval_joint_checkpoint.py ^
  --checkpoint {out / dest_ckpt.name} ^
  --config-from {cfg_src} ^
  --eval-fhs-train ^
  --device cuda ^
  --out-json {out / "cindex_eval.json"}
"""
    (out / "LOAD_MODEL.md").write_text(
        "# Joint model epoch 26 (analysis bundle)\n\n"
        "## C-index at epoch 26 (from training log)\n\n"
        f"- FHS validation: `{metrics_line['val_cindex']:.4f}`\n" if metrics_line else ""
        f"- WHI test (logged during train): `{metrics_line['test_whi_cindex']:.4f}`\n\n" if metrics_line else ""
        "## Files\n\n"
        f"- `{dest_ckpt.name}` — model weights (`state_dict`)\n"
        "- `model_manifest.json` — paths, config, provenance\n"
        "- `cindex_eval.json` — full-split re-eval (train/val/WHI)\n"
        f"- `training_log_epoch{epoch}.json` — metrics.jsonl row for epoch {epoch}\n\n"
        "## Reload & evaluate\n\n"
        "```powershell\n" + load_cmd + "```\n",
        encoding="utf-8",
    )
    print(f"Packaged -> {out}")


if __name__ == "__main__":
    main()
