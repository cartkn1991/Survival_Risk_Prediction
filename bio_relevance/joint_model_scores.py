"""Log-hazard and age predictions from a packaged joint epoch checkpoint (raw omics)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.aesurv_contrastive.eval_joint_checkpoint import build_model_from_joint_ckpt
from experiments.aesurv_contrastive.joint_dann_aesurv_model import apply_modality_torch
from train_aesurv_dann_latent_aux import load_aux_targets
from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    _resolve_data_path,
    load_bundle_with_cache,
)


def _torch_load_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class JointProjAgeRisk(nn.Module):
    """Gradients w.r.t. JL-projected input through joint DANN + head."""

    def __init__(self, joint, age_mu: float, age_sd: float):
        super().__init__()
        self.joint = joint
        self.age_mu = float(age_mu)
        self.age_sd = float(age_sd)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x_proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, _logvar = self.joint.dann.encode(x_proj)
        _, _, log_h, _, _, age_z, _, _, _ = self.joint.forward_head(mu, mu, mu, sample_z=False)
        age_pred = age_z * self.age_sd + self.age_mu
        return age_pred, log_h


def load_joint_from_analysis_dir(analysis_dir: Path, device: torch.device):
    analysis_dir = Path(analysis_dir)
    manifest = json.loads((analysis_dir / "model_manifest.json").read_text(encoding="utf-8"))
    ckpt = analysis_dir / manifest["weights_file"]
    cfg_from = (
        analysis_dir.parent / "aesurv_joint_dann_contrastive_best" / "joint_dann_aesurv_model.pt"
    )
    if not cfg_from.is_file():
        cfg_from = Path(manifest.get("config_from", cfg_from))
    model = build_model_from_joint_ckpt(
        ckpt,
        Path(manifest["dann_encoder_ckpt"]),
        Path(manifest["dann_preprocess_npz"]),
        device,
        config_from=cfg_from if cfg_from.is_file() else None,
    )
    return model, manifest


def _age_norm_global(device: torch.device) -> Tuple[float, float]:
    log = logging.getLogger("joint_scores")
    age_fhs, _ = load_aux_targets(
        Path("FHS_methylation_with_snp_1milfeatures.parquet"),
        Path("FHS_cell_composition.parquet"),
        "Share_ID",
        ["B", "NK", "CD4T", "CD8T", "Mono", "Neutro"],
        log,
        "FHS",
    )
    age_whi, _ = load_aux_targets(
        Path("WHI_methylation_with_snp_merged_1milfeatures.parquet"),
        Path("WHI_cell_composition.parquet"),
        "sample_ID",
        ["B", "NK", "CD4T", "CD8T", "Mono", "Neutro"],
        log,
        "WHI",
    )
    age_all = np.concatenate([age_fhs, age_whi])
    return float(age_all.mean()), float(age_all.std() + 1e-6)


@torch.no_grad()
def joint_pooled_predictions(
    analysis_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    cache_dir: str = "vae_cox_cache",
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Returns ``(log_h, age_pred, n_fhs, n_whi)`` in FHS-then-WHI order."""
    model, _manifest = load_joint_from_analysis_dir(analysis_dir, device)
    model.eval()
    age_mu, age_sd = _age_norm_global(device)

    cache = _resolve_data_path(cache_dir)
    X_meth_fhs, X_snp_fhs, _, _, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        "FHS", _resolve_data_path("FHS_methylation_with_snp_1milfeatures_combined_training.npz"),
        _resolve_data_path("FHS_methylation_with_snp_1milfeatures.parquet"),
        None, None, None, cache, True,
    )
    X_meth_whi, X_snp_whi, _, _, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
        "WHI_raw",
        _resolve_data_path("WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz"),
        _resolve_data_path("WHI_methylation_with_snp_merged_1milfeatures.parquet"),
        None, None, None, cache, True,
    )
    fhs_npz = _resolve_data_path("FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    whi_npz = _resolve_data_path("WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    meth_fhs, snp_fhs = _get_truncated_feature_names(fhs_npz, None, n_cpg_fhs, n_snp_fhs)
    meth_whi, snp_whi = _get_truncated_feature_names(whi_npz, None, n_cpg_whi, n_snp_whi)
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, _, _ = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_fhs, snp_fhs, X_meth_whi, X_snp_whi, meth_whi, snp_whi,
    )
    n_fhs, n_whi = X_meth_fhs.shape[0], X_meth_whi.shape[0]
    n_total = n_fhs + n_whi
    log_h = np.empty((n_total,), dtype=np.float32)
    age_pred = np.empty((n_total,), dtype=np.float32)

    def _score_block(xm: np.ndarray, xs: np.ndarray, sl: slice) -> None:
        for s in range(0, xm.shape[0], batch_size):
            e = min(s + batch_size, xm.shape[0])
            m = torch.from_numpy(np.ascontiguousarray(xm[s:e])).to(device)
            v = torch.from_numpy(np.ascontiguousarray(xs[s:e])).to(device)
            mu, _, _ = model.encode_raw(m, v, "both")
            _, _, lh, _, _, age_z, _, _, _ = model.forward_head(mu, mu, mu, sample_z=False)
            log_h[sl][s:e] = lh.squeeze(-1).cpu().numpy()
            age_pred[sl][s:e] = (age_z.squeeze(-1) * age_sd + age_mu).cpu().numpy()

    _score_block(X_meth_fhs, X_snp_fhs, slice(0, n_fhs))
    _score_block(X_meth_whi, X_snp_whi, slice(n_fhs, n_total))
    return log_h, age_pred, n_fhs, n_whi


def joint_projected_features(
    analysis_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    cache_dir: str = "vae_cox_cache",
) -> Tuple[np.ndarray, int, int]:
    """Pooled JL projections (FHS then WHI) from the joint preprocessor."""
    model, _ = load_joint_from_analysis_dir(analysis_dir, device)
    model.eval()
    cache = _resolve_data_path(cache_dir)
    X_meth_fhs, X_snp_fhs, _, _, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        "FHS", _resolve_data_path("FHS_methylation_with_snp_1milfeatures_combined_training.npz"),
        _resolve_data_path("FHS_methylation_with_snp_1milfeatures.parquet"),
        None, None, None, cache, True,
    )
    X_meth_whi, X_snp_whi, _, _, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
        "WHI_raw",
        _resolve_data_path("WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz"),
        _resolve_data_path("WHI_methylation_with_snp_merged_1milfeatures.parquet"),
        None, None, None, cache, True,
    )
    fhs_npz = _resolve_data_path("FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    whi_npz = _resolve_data_path("WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    meth_fhs, snp_fhs = _get_truncated_feature_names(fhs_npz, None, n_cpg_fhs, n_snp_fhs)
    meth_whi, snp_whi = _get_truncated_feature_names(whi_npz, None, n_cpg_whi, n_snp_whi)
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, _, _ = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_fhs, snp_fhs, X_meth_whi, X_snp_whi, meth_whi, snp_whi,
    )
    n_fhs, n_whi = X_meth_fhs.shape[0], X_meth_whi.shape[0]
    proj = np.empty((n_fhs + n_whi, 0), dtype=np.float32)

    def _proj_block(xm: np.ndarray, xs: np.ndarray, sl: slice) -> None:
        nonlocal proj
        for s in range(0, xm.shape[0], batch_size):
            e = min(s + batch_size, xm.shape[0])
            m = torch.from_numpy(np.ascontiguousarray(xm[s:e])).to(device)
            v = torch.from_numpy(np.ascontiguousarray(xs[s:e])).to(device)
            xm_a, xs_a = apply_modality_torch(
                m, v, "both", model._mean_full, model.n_cpg, model.n_snp,
            )
            x_proj = model.preproc(xm_a, xs_a, use_m_values=model.meth_as_mvalues)
            if proj.shape[1] == 0:
                d_proj = int(x_proj.shape[1])
                proj = np.empty((n_fhs + n_whi, d_proj), dtype=np.float32)
            proj[sl][s:e] = x_proj.detach().cpu().numpy()

    _proj_block(X_meth_fhs, X_snp_fhs, slice(0, n_fhs))
    _proj_block(X_meth_whi, X_snp_whi, slice(n_fhs, n_fhs + n_whi))
    return proj, n_fhs, n_whi


def joint_grad_model(analysis_dir: Path, device: torch.device) -> JointProjAgeRisk:
    joint, _ = load_joint_from_analysis_dir(analysis_dir, device)
    age_mu, age_sd = _age_norm_global(device)
    return JointProjAgeRisk(joint, age_mu, age_sd).to(device)
