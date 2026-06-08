#!/usr/bin/env python3
"""Rank-mean ensemble of contrastive AESURV log_h on WHI and FHS val."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dann_aesurv_bridge import DannLatentEncoder, InvariantPreprocessor, load_mini_dann_for_fusion
from experiments.aesurv_contrastive.aesurv_head_aux_contrastive import AESurvHeadAuxContrastive
from train_aesurv_dann_latent_aux import _apply_dann_input_modality
from train_dann_survival import harrell_c_index, load_or_extract_latent, _split_indices
from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    _resolve_data_path,
    load_bundle_with_cache,
)


def _load_head(ckpt_path: Path, latent_dim: int, device: torch.device) -> AESurvHeadAuxContrastive:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw.get("config", {})
    z_dim = int(cfg.get("z_dim", 8)) if isinstance(cfg, dict) else 8
    proj = int(cfg.get("contrast_proj_dim", 64)) if isinstance(cfg, dict) else 64
    tau = float(cfg.get("contrast_tau", 0.07)) if isinstance(cfg, dict) else 0.07
    model = AESurvHeadAuxContrastive(
        latent_dim, z_dim=z_dim, n_cells=6,
        contrast_proj_dim=proj, contrast_tau=tau,
    ).to(device)
    model.load_state_dict(raw["state_dict"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_log_h(model: AESurvHeadAuxContrastive, mu: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    out = []
    for s in range(0, len(mu), batch):
        x = torch.from_numpy(mu[s:s + batch]).float().to(device)
        log_h = model(x, sample_z=False)[2].cpu().numpy()
        out.append(log_h)
    return np.concatenate(out, axis=0)


def resolve_torch_device(device_str: str, log: logging.Logger | None = None) -> torch.device:
    """Use CPU when cuda is requested but PyTorch has no CUDA build."""
    req = device_str.strip().lower()
    if req == "cuda" or req.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        msg = f"CUDA requested but unavailable ({torch.__version__}); using CPU."
        if log is not None:
            log.warning(msg)
        else:
            print("WARNING:", msg, file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_str)


def rank_mean_ensemble(risk_mat: np.ndarray) -> np.ndarray:
    n_models, n = risk_mat.shape
    ranks = np.zeros((n_models, n), dtype=np.float64)
    for i in range(n_models):
        order = np.argsort(risk_mat[i])
        ranks[i, order] = np.arange(n, dtype=np.float64)
    return ranks.mean(axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, nargs="+", required=True)
    p.add_argument("--dann-encoder-ckpt", type=str, default="runs/mini_vae_dann_mmd_rich/mini_dann_model.pt")
    p.add_argument("--dann-preprocess-npz", type=str, default="runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz")
    p.add_argument("--combined-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--meta-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--test-combined-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--test-meta-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="runs/aesurv_contrastive_ensemble/ensemble_summary.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    log = logging.getLogger("ensemble")
    device = resolve_torch_device(args.device, log)
    log.info("ensemble device=%s", device)
    ckpt_path = _resolve_data_path(args.dann_encoder_ckpt)
    npz_path = _resolve_data_path(args.dann_preprocess_npz)
    preproc = InvariantPreprocessor(npz_path).to(device).eval()
    mini, cfg = load_mini_dann_for_fusion(ckpt_path, map_location=device)
    encoder = DannLatentEncoder(mini.to(device)).eval()
    latent_dim = int(cfg["latent_dim"])

    fhs_npz = _resolve_data_path(args.combined_npz)
    fhs_pq = _resolve_data_path(args.meta_parquet)
    whi_npz = _resolve_data_path(args.test_combined_npz)
    whi_pq = _resolve_data_path(args.test_meta_parquet)
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        "FHS", fhs_npz, fhs_pq, None, None, None, Path("vae_cox_cache"), True,
    )
    X_meth_whi, X_snp_whi, t_whi, e_whi, _, _ = load_bundle_with_cache(
        "WHI_raw", whi_npz, whi_pq, None, None, None, Path("vae_cox_cache"), True,
    )
    meth_fhs_names, snp_fhs_names = _get_truncated_feature_names(fhs_npz, None, n_cpg_fhs, n_snp_fhs)
    meth_whi_names, snp_whi_names = _get_truncated_feature_names(whi_npz, None, X_meth_whi.shape[1], X_snp_whi.shape[1])
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, n_cpg, n_snp = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_fhs_names, snp_fhs_names,
        X_meth_whi, X_snp_whi, meth_whi_names, snp_whi_names,
    )
    mean_np = preproc.mean_.detach().cpu().numpy().astype(np.float32)
    X_meth_fhs, X_snp_fhs = _apply_dann_input_modality(X_meth_fhs, X_snp_fhs, "both", mean_np, n_cpg, n_snp, log, "FHS")
    X_meth_whi, X_snp_whi = _apply_dann_input_modality(X_meth_whi, X_snp_whi, "both", mean_np, n_cpg, n_snp, log, "WHI")
    latent_cache = Path("vae_cox_cache/dann_latent")
    mu_fhs = load_or_extract_latent(
        "FHS", X_meth_fhs, X_snp_fhs, preproc, encoder,
        meth_as_mvalues=True, latent_dim=latent_dim, device=device,
        cache_dir=latent_cache, ckpt_path=ckpt_path, npz_path=npz_path,
        use_cache=True, log=log,
    )
    mu_whi = load_or_extract_latent(
        "WHI", X_meth_whi, X_snp_whi, preproc, encoder,
        meth_as_mvalues=True, latent_dim=latent_dim, device=device,
        cache_dir=latent_cache, ckpt_path=ckpt_path, npz_path=npz_path,
        use_cache=True, log=log,
    )
    va_idx, _ = _split_indices(mu_fhs.shape[0], args.val_frac, args.seed, stratify_event=e_fhs.astype(np.int32))

    risks_whi, risks_va, per_model = [], [], []
    for ck in args.checkpoints:
        model = _load_head(_resolve_data_path(ck), latent_dim, device)
        log_h_whi = predict_log_h(model, mu_whi, device)
        log_h_va = predict_log_h(model, mu_fhs[va_idx], device)
        risks_whi.append(log_h_whi)
        risks_va.append(log_h_va)
        per_model.append({
            "checkpoint": str(ck),
            "whi_cindex": harrell_c_index(t_whi, e_whi.astype(np.int32), log_h_whi),
            "fhs_val_cindex": harrell_c_index(t_fhs[va_idx], e_fhs[va_idx].astype(np.int32), log_h_va),
        })

    whi_mat = np.stack(risks_whi, axis=0)
    va_mat = np.stack(risks_va, axis=0)
    ens_whi = rank_mean_ensemble(whi_mat)
    ens_va = rank_mean_ensemble(va_mat)

    out = {
        "n_models": len(args.checkpoints),
        "per_model": per_model,
        "ensemble_whi_cindex": harrell_c_index(t_whi, e_whi.astype(np.int32), ens_whi),
        "ensemble_fhs_val_cindex": harrell_c_index(t_fhs[va_idx], e_fhs[va_idx].astype(np.int32), ens_va),
    }
    out_path = _resolve_data_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
