#!/usr/bin/env python3
"""Hyperparameter grid for contrastive AESURV (phase contrast + phase aux)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DANN_CKPT = ROOT / "runs/mini_vae_dann_mmd_rich/mini_dann_model.pt"
DANN_NPZ = ROOT / "runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz"
BASELINE_INIT = ROOT / "experiments/aesurv_risk_recon/checkpoints_baseline/aesurv_aux_model.pt"
TRAIN = ROOT / "train_aesurv_dann_latent_aux_contrastive.py"

CONTRAST_SPECS = [
    {"id": "T0_ref", "w": 1.0, "tau": 0.07, "warmup": 10, "proj": 64},
    {"id": "T1", "w": 0.25, "tau": 0.07, "warmup": 10, "proj": 64},
    {"id": "T2", "w": 0.5, "tau": 0.07, "warmup": 10, "proj": 64},
    {"id": "T3", "w": 2.0, "tau": 0.07, "warmup": 10, "proj": 64},
    {"id": "T4", "w": 1.0, "tau": 0.05, "warmup": 10, "proj": 64},
    {"id": "T5", "w": 1.0, "tau": 0.10, "warmup": 10, "proj": 64},
    {"id": "T6", "w": 1.0, "tau": 0.07, "warmup": 5, "proj": 64},
    {"id": "T7", "w": 1.0, "tau": 0.07, "warmup": 20, "proj": 64},
    {"id": "T8", "w": 1.0, "tau": 0.07, "warmup": 10, "proj": 32},
    {"id": "T9", "w": 1.0, "tau": 0.07, "warmup": 10, "proj": 128},
]


def _run_train(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        return subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True).returncode


def _load_meta(out_dir: Path) -> dict:
    p = out_dir / "aesurv_aux_contrastive_run_meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _summarize_runs(out_root: Path, runs: list[dict]) -> dict:
    rows = []
    for r in runs:
        meta = _load_meta(Path(r["out_dir"]))
        rows.append({**r, **meta})
    best_val = max(rows, key=lambda x: x.get("best_val_cindex", -1), default=None)
    best_whi_at_val = max(rows, key=lambda x: x.get("final_whi_cindex", -1), default=None)
    best_max_whi = max(rows, key=lambda x: x.get("max_whi_cindex_during_train", -1), default=None)
    summary = {
        "runs": rows,
        "best_fhs_val": best_val,
        "best_final_whi": best_whi_at_val,
        "best_max_whi_train": best_max_whi,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_val:
        (out_root / "best_config.json").write_text(json.dumps(best_val, indent=2), encoding="utf-8")
    return summary


def run_contrast_phase(out_root: Path, device: str, epochs: int, dry_run: bool) -> list[dict]:
    py = sys.executable
    runs = []
    for sp in CONTRAST_SPECS:
        out_dir = out_root / sp["id"]
        cmd = [
            py, str(TRAIN),
            "--dann-encoder-ckpt", str(DANN_CKPT),
            "--dann-preprocess-npz", str(DANN_NPZ),
            "--init-from-baseline", str(BASELINE_INIT),
            "--aux-age-weight", "12", "--aux-cell-weight", "1",
            "--aux-contrast-weight", str(sp["w"]),
            "--contrast-tau", str(sp["tau"]),
            "--contrast-warmup-epochs", str(sp["warmup"]),
            "--contrast-proj-dim", str(sp["proj"]),
            "--z-dim", "8", "--seed", "42", "--balance-events",
            "--device", device, "--epochs", str(epochs),
            "--out-dir", str(out_dir),
        ]
        print("RUN", sp["id"])
        if dry_run:
            print(" ", " ".join(cmd))
            runs.append({"id": sp["id"], "out_dir": str(out_dir), "phase": "contrast", **sp})
            continue
        rc = _run_train(cmd, out_dir / "train.log")
        row = {"id": sp["id"], "out_dir": str(out_dir), "returncode": rc, "phase": "contrast", **sp}
        row.update(_load_meta(out_dir))
        runs.append(row)
    return runs


def run_aux_phase(
    out_root: Path,
    best: dict,
    device: str,
    epochs: int,
    dry_run: bool,
) -> list[dict]:
    py = sys.executable
    w = best.get("w", best.get("aux_contrast_weight", 1.0))
    tau = best.get("tau", best.get("contrast_tau", 0.07))
    warmup = best.get("warmup", best.get("contrast_warmup_epochs", 10))
    proj = best.get("proj", best.get("contrast_proj_dim", 64))
    best_pt = Path(best.get("out_dir", "")) / "aesurv_aux_contrastive_model.pt"

    aux_specs = [
        {"id": "A0", "age": 12, "cell": 1, "init": str(BASELINE_INIT), "ep": epochs},
        {"id": "A1", "age": 10, "cell": 1, "init": str(BASELINE_INIT), "ep": epochs},
        {"id": "A2", "age": 14, "cell": 1, "init": str(BASELINE_INIT), "ep": epochs},
        {"id": "A3", "age": 12, "cell": 0.5, "init": str(BASELINE_INIT), "ep": epochs},
        {"id": "A4", "age": 12, "cell": 2, "init": str(BASELINE_INIT), "ep": epochs},
        {"id": "A5", "age": 12, "cell": 1, "init": str(best_pt), "ep": min(100, epochs), "patience": 20},
    ]
    runs = []
    for sp in aux_specs:
        out_dir = out_root / "aux" / sp["id"]
        cmd = [
            py, str(TRAIN),
            "--dann-encoder-ckpt", str(DANN_CKPT),
            "--dann-preprocess-npz", str(DANN_NPZ),
            "--init-from", sp["init"],
            "--aux-age-weight", str(sp["age"]),
            "--aux-cell-weight", str(sp["cell"]),
            "--aux-contrast-weight", str(w),
            "--contrast-tau", str(tau),
            "--contrast-warmup-epochs", str(warmup),
            "--contrast-proj-dim", str(proj),
            "--z-dim", "8", "--seed", "42", "--balance-events",
            "--device", device,
            "--epochs", str(sp["ep"]),
            "--patience", str(sp.get("patience", 25)),
            "--out-dir", str(out_dir),
        ]
        print("RUN aux", sp["id"])
        if dry_run:
            print(" ", " ".join(cmd))
            runs.append({"id": sp["id"], "out_dir": str(out_dir), "phase": "aux", **sp})
            continue
        rc = _run_train(cmd, out_dir / "train.log")
        row = {"id": sp["id"], "out_dir": str(out_dir), "returncode": rc, "phase": "aux", **sp}
        row.update(_load_meta(out_dir))
        runs.append(row)
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=str, default="runs/aesurv_contrastive_tune")
    p.add_argument("--phase", type=str, choices=("contrast", "aux", "all"), default="all")
    p.add_argument("--best-config", type=str, default=None,
                   help="best_config.json from contrast phase (required for aux-only).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out_root)
    all_runs: list[dict] = []

    contrast_runs: list[dict] = []
    if args.phase in ("contrast", "all"):
        contrast_runs = run_contrast_phase(out_root, args.device, args.epochs, args.dry_run)
        all_runs.extend(contrast_runs)
        if not args.dry_run and contrast_runs:
            _summarize_runs(out_root, contrast_runs)
            print("Wrote contrast summary ->", out_root / "best_config.json")

    if args.phase in ("aux", "all"):
        cfg_path = Path(args.best_config) if args.best_config else out_root / "best_config.json"
        if not cfg_path.exists() and not args.dry_run:
            raise SystemExit(
                f"Missing {cfg_path}. Run contrast phase first, or:\n"
                f"  python experiments/aesurv_contrastive/run_tune_grid.py --phase contrast ..."
            )
        best = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else CONTRAST_SPECS[2]
        aux_runs = run_aux_phase(out_root, best, args.device, args.epochs, args.dry_run)
        all_runs.extend(aux_runs)

    if not args.dry_run and all_runs:
        summary = _summarize_runs(out_root, all_runs)
        print(json.dumps({
            "best_fhs_val_id": summary.get("best_fhs_val", {}).get("id"),
            "best_fhs_val": summary.get("best_fhs_val", {}).get("best_val_cindex"),
            "best_final_whi": summary.get("best_final_whi", {}).get("final_whi_cindex"),
        }, indent=2))
        print("Wrote", out_root / "tune_summary.json")


if __name__ == "__main__":
    main()
