#!/usr/bin/env python3
"""Evaluate FHS val / WHI C-index for a joint_dann_aesurv checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aesurv_contrastive.aesurv_head_aux_contrastive import AESurvHeadAuxContrastive
from experiments.aesurv_contrastive.joint_dann_aesurv_model import JointDannAesurvContrastive
from train_aesurv_joint_dann_contrastive import _eval_cindex_raw, _resolve_device
from train_dann_survival import _setup_logger, _split_indices
from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    _resolve_data_path,
    load_bundle_with_cache,
)


def _load_joint_config(ckpt_path: Path, config_from: Path | None) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw.get("config")
    if cfg:
        return dict(cfg)
    if config_from is None:
        sibling = ckpt_path.parent / "joint_dann_aesurv_model.pt"
        if sibling.is_file():
            config_from = sibling
    if config_from is None:
        raise SystemExit(f"No config in {ckpt_path}; pass --config-from joint_dann_aesurv_model.pt")
    parent = torch.load(config_from, map_location="cpu", weights_only=False)
    return dict(parent["config"])


def build_model_from_joint_ckpt(
    ckpt_path: Path,
    dann_ckpt: Path,
    dann_npz: Path,
    device: torch.device,
    config_from: Path | None = None,
) -> JointDannAesurvContrastive:
    cfg = _load_joint_config(ckpt_path, config_from)
    enc_h = tuple(int(x) for x in str(cfg.get("enc_hidden", "64,32")).split(",") if x.strip())
    dec_h = tuple(int(x) for x in str(cfg.get("dec_hidden", "32,64")).split(",") if x.strip())
    cell_n = int(cfg.get("n_cells", 6))
    head = AESurvHeadAuxContrastive(
        in_dim=int(cfg["latent_dim"]),
        enc_hidden=enc_h,
        dec_hidden=dec_h,
        z_dim=int(cfg.get("z_dim", 8)),
        cohort_hidden=int(cfg.get("cohort_hidden", 8)),
        dropout=float(cfg.get("dropout", 0.40)),
        n_cells=cell_n,
        contrast_proj_dim=int(cfg.get("contrast_proj_dim", 64)),
        contrast_hidden=int(cfg.get("contrast_hidden", 32)),
        contrast_tau=float(cfg.get("contrast_tau", 0.07)),
    )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    znp = np.load(dann_npz, allow_pickle=False)
    meth_mval = bool(cfg.get("meth_as_mvalues", bool(int(znp["meth_as_mvalues"][0])) if "meth_as_mvalues" in znp.files else True))
    model, _ = JointDannAesurvContrastive.from_checkpoints(
        str(dann_ckpt), str(dann_npz), head,
        n_cpg=int(cfg["n_cpg"]), n_snp=int(cfg["n_snp"]),
        meth_as_mvalues=meth_mval,
        map_location=device,
    )
    state = raw.get("state_dict", raw)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dann-encoder-ckpt", type=str, default="runs/mini_vae_dann_mmd_rich/mini_dann_model.pt")
    p.add_argument("--dann-preprocess-npz", type=str, default="runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz")
    p.add_argument("--combined-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--meta-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--test-combined-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--test-meta-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--cache-dir", type=str, default="vae_cox_cache")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--config-from",
        type=str,
        default=None,
        help="joint_dann_aesurv_model.pt with config (for whi_topk ckpts)",
    )
    p.add_argument("--out-json", type=str, default=None, help="Write metrics JSON to this path")
    p.add_argument("--eval-fhs-train", action="store_true", help="Also compute FHS training-set C-index")
    args = p.parse_args()

    log = _setup_logger(None)
    device = _resolve_device(args.device, log)
    ckpt = _resolve_data_path(args.checkpoint)
    cfg_from = _resolve_data_path(args.config_from) if args.config_from else None
    model = build_model_from_joint_ckpt(
        ckpt,
        _resolve_data_path(args.dann_encoder_ckpt),
        _resolve_data_path(args.dann_preprocess_npz),
        device,
        config_from=cfg_from,
    )
    model.eval()

    cache = _resolve_data_path(args.cache_dir)
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        "FHS", _resolve_data_path(args.combined_npz), _resolve_data_path(args.meta_parquet),
        None, None, None, cache, True,
    )
    X_meth_whi, X_snp_whi, t_whi, e_whi, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
        "WHI_raw", _resolve_data_path(args.test_combined_npz), _resolve_data_path(args.test_meta_parquet),
        None, None, None, cache, True,
    )
    fhs_npz = _resolve_data_path(args.combined_npz)
    whi_npz = _resolve_data_path(args.test_combined_npz)
    meth_fhs_names, snp_fhs_names = _get_truncated_feature_names(fhs_npz, None, n_cpg_fhs, n_snp_fhs)
    meth_whi_names, snp_whi_names = _get_truncated_feature_names(whi_npz, None, n_cpg_whi, n_snp_whi)
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, _, _ = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_fhs_names, snp_fhs_names,
        X_meth_whi, X_snp_whi, meth_whi_names, snp_whi_names,
    )
    tr_idx, va_idx = _split_indices(
        X_meth_fhs.shape[0], args.val_frac, args.seed, stratify_event=e_fhs.astype(np.int32),
    )

    c_va = _eval_cindex_raw(
        model, X_meth_fhs[va_idx], X_snp_fhs[va_idx], t_fhs[va_idx], e_fhs[va_idx], device, args.batch_size,
    )
    c_whi = _eval_cindex_raw(model, X_meth_whi, X_snp_whi, t_whi, e_whi, device, args.batch_size)
    raw_ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)

    out = {
        "checkpoint": str(ckpt),
        "fhs_val_cindex": float(c_va),
        "whi_cindex": float(c_whi),
        "epoch": raw_ckpt.get("epoch") or raw_ckpt.get("best_epoch"),
        "whi_cindex_saved": raw_ckpt.get("whi_cindex"),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_fhs_train": int(len(tr_idx)),
        "n_fhs_val": int(len(va_idx)),
        "n_whi_test": int(X_meth_whi.shape[0]),
    }
    if args.eval_fhs_train:
        out["fhs_train_cindex"] = float(_eval_cindex_raw(
            model, X_meth_fhs[tr_idx], X_snp_fhs[tr_idx], t_fhs[tr_idx], e_fhs[tr_idx], device, args.batch_size,
        ))
    if args.out_json:
        out_path = _resolve_data_path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
