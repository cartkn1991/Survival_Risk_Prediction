#!/usr/bin/env python3
"""Compare frozen A3 vs joint DANN training vs joint with higher MMD/domain weights."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train_aesurv_joint_dann_contrastive.py"
FROZEN_META = ROOT / "runs/aesurv_contrastive_best/aesurv_aux_contrastive_run_meta.json"
DEFAULT_OUT = ROOT / "runs/aesurv_joint_ablation"


def _read_meta(path: Path) -> dict:
    if not path.is_file():
        return {"error": f"missing {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_joint(
    out_dir: Path,
    *,
    w_mmd: float,
    w_dom: float,
    epochs: int,
    device: str,
    init_head: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TRAIN),
        "--dann-encoder-ckpt", "runs/mini_vae_dann_mmd_rich/mini_dann_model.pt",
        "--dann-preprocess-npz", "runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz",
        "--init-head", init_head,
        "--aux-age-weight", "12", "--aux-cell-weight", "0.5",
        "--w-recon", "0.5", "--w-dom", str(w_dom), "--w-mmd", str(w_mmd),
        "--w-mmd-batch", "2.0", "--w-batch-adv", "1.0",
        "--balance-events",
        "--epochs", str(epochs), "--patience", "20", "--min-epochs", "10",
        "--out-dir", str(out_dir), "--device", device,
    ]
    print("RUN", out_dir.name, "w_mmd=", w_mmd, "w_dom=", w_dom)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    meta_path = out_dir / "joint_dann_aesurv_run_meta.json"
    m = _read_meta(meta_path)
    return {
        "out_dir": str(out_dir),
        "fhs_val_cindex": m.get("best_val_cindex"),
        "whi_cindex": m.get("final_whi_cindex"),
        "max_whi_cindex": m.get("max_whi_cindex_during_train"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", type=str, default=str(DEFAULT_OUT / "comparison.json"))
    p.add_argument("--epochs", type=int, default=30, help="Epochs per joint run (use 80 for full)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-train", action="store_true", help="Only aggregate existing run dirs")
    p.add_argument("--init-head", type=str, default="runs/aesurv_contrastive_best/aesurv_aux_contrastive_model.pt")
    args = p.parse_args()

    out_root = DEFAULT_OUT
    frozen = _read_meta(FROZEN_META)
    results = {
        "frozen_a3": {
            "out_dir": "runs/aesurv_contrastive_best",
            "fhs_val_cindex": frozen.get("best_val_cindex"),
            "whi_cindex": frozen.get("final_whi_cindex"),
            "max_whi_cindex": frozen.get("max_whi_cindex_during_train"),
        },
    }

    if not args.skip_train:
        results["joint_default"] = _run_joint(
            out_root / "joint_default",
            w_mmd=5.0, w_dom=0.6, epochs=args.epochs, device=args.device, init_head=args.init_head,
        )
        results["joint_high_mmd"] = _run_joint(
            out_root / "joint_high_mmd",
            w_mmd=10.0, w_dom=1.2, epochs=args.epochs, device=args.device, init_head=args.init_head,
        )
    else:
        for name, sub in [("joint_default", "joint_default"), ("joint_high_mmd", "joint_high_mmd")]:
            m = _read_meta(out_root / sub / "joint_dann_aesurv_run_meta.json")
            results[name] = {
                "out_dir": str(out_root / sub),
                "fhs_val_cindex": m.get("best_val_cindex"),
                "whi_cindex": m.get("final_whi_cindex"),
                "max_whi_cindex": m.get("max_whi_cindex_during_train"),
            }
        smoke = _read_meta(out_root / "joint_default_smoke" / "joint_dann_aesurv_run_meta.json")
        if "best_val_cindex" in smoke or "final_whi_cindex" in smoke:
            results["joint_smoke"] = {
                "out_dir": str(out_root / "joint_default_smoke"),
                "fhs_val_cindex": smoke.get("best_val_cindex"),
                "whi_cindex": smoke.get("final_whi_cindex"),
                "max_whi_cindex": smoke.get("max_whi_cindex_during_train"),
            }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim = {}
    for k, v in results.items():
        slim[k] = {
            "fhs_val_cindex": v.get("fhs_val_cindex") or v.get("best_val_cindex"),
            "whi_cindex": v.get("whi_cindex") or v.get("final_whi_cindex"),
            "max_whi_cindex": v.get("max_whi_cindex") or v.get("max_whi_cindex_during_train"),
            "out_dir": v.get("out_dir"),
        }
    out_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
