#!/usr/bin/env python3
"""AESURV head + age/cell aux + **CpG/SNP InfoNCE contrastive** aux.

Same pipeline as ``train_aesurv_dann_latent_aux_risk_recon.py`` (loads FHS/WHI, frozen DANN,
balanced adv batches). Adds unimodal DANN latents (cpgs_only / snps_only) and symmetric
InfoNCE on projection heads; fused path unchanged for Cox / age / cell.

Usage (mirror risk-recon):

    python train_aesurv_dann_latent_aux_contrastive.py ^
      --dann-encoder-ckpt runs/mini_vae_dann_mmd_rich/mini_dann_model.pt ^
      --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz ^
      --init-from-baseline experiments/aesurv_risk_recon/checkpoints_baseline/aesurv_aux_model.pt ^
      --aux-age-weight 12.0 --aux-cell-weight 1.0 ^
      --aux-contrast-weight 1.0 --contrast-tau 0.07 ^
      --z-dim 8 --seed 42 --balance-events ^
      --out-dir runs/aesurv_aux_contrastive_v1
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dann_aesurv_bridge import DannLatentEncoder, InvariantPreprocessor, load_mini_dann_for_fusion
from experiments.aesurv_contrastive.aesurv_head_aux_contrastive import AESurvHeadAuxContrastive
from train_aesurv_dann_latent_aux import (
    _apply_dann_input_modality,
    _survival_ipcw_td_auroc_auprc,
    load_aux_targets,
)
from train_dann_survival import (
    cox_ph_loss,
    harrell_c_index,
    load_or_extract_latent,
    _setup_logger,
    _split_indices,
)
from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    _resolve_data_path,
    load_bundle_with_cache,
)


class _TopKWhiCheckpoints:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self._heap: List[Tuple[float, int, Dict[str, torch.Tensor]]] = []

    def consider(self, c_whi: float, epoch: int, state: Dict[str, torch.Tensor]) -> None:
        if math.isnan(c_whi):
            return
        item = (float(c_whi), int(epoch), {kk: v.detach().cpu().clone() for kk, v in state.items()})
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, item)
        elif c_whi > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def best_list(self) -> List[Tuple[float, int, Dict[str, torch.Tensor]]]:
        return sorted(self._heap, key=lambda x: -x[0])


def _load_baseline_weights(model: AESurvHeadAuxContrastive, ckpt_path: Path, log: logging.Logger) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.info("Baseline init: %d new keys (contrastive branches): %s", len(missing), missing[:8])
    if unexpected:
        log.warning("Baseline init: unexpected keys: %s", unexpected[:8])


def _extract_modality_latents(
    label: str,
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    mode: str,
    mean_np: np.ndarray,
    n_cpg: int,
    n_snp: int,
    preproc: InvariantPreprocessor,
    encoder: DannLatentEncoder,
    meth_mval: bool,
    latent_dim: int,
    device: torch.device,
    latent_cache: Path,
    ckpt_path: Path,
    npz_path: Path,
    use_cache: bool,
    log: logging.Logger,
) -> np.ndarray:
    xm, xs = _apply_dann_input_modality(X_meth, X_snp, mode, mean_np, n_cpg, n_snp, log, label)
    return load_or_extract_latent(
        label, xm, xs, preproc, encoder, meth_as_mvalues=meth_mval,
        latent_dim=latent_dim, device=device, cache_dir=latent_cache,
        ckpt_path=ckpt_path, npz_path=npz_path, use_cache=use_cache, log=log,
        extra_tag=mode,
    )


def _contrast_scale(epoch: int, warmup: int, target: float) -> float:
    if warmup <= 0:
        return target
    return target * min(1.0, (epoch + 1) / float(warmup))


def main() -> None:
    p = argparse.ArgumentParser(description="AESURV aux + CpG/SNP contrastive alignment.")
    p.add_argument("--combined-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--meta-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--test-combined-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--test-meta-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--snp-columns-txt", type=str, default=None)
    p.add_argument("--max-cpg", type=int, default=None)
    p.add_argument("--max-snp", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default="vae_cox_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dann-encoder-ckpt", type=str, required=True)
    p.add_argument("--dann-preprocess-npz", type=str, required=True)
    p.add_argument("--dann-meth-beta-values", action="store_true")
    p.add_argument("--latent-cache-dir", type=str, default="vae_cox_cache/dann_latent")
    p.add_argument("--input-modality", type=str, default="both", choices=("both", "cpgs_only", "snps_only"),
                   help="Fused DANN input for Cox path (default both). Contrastive always uses cpg+snp unimodal.")
    p.add_argument("--fhs-cell-comp-parquet", type=str, default="FHS_cell_composition.parquet")
    p.add_argument("--whi-cell-comp-parquet", type=str, default="WHI_cell_composition.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--whi-id-col", type=str, default="sample_ID")
    p.add_argument("--aux-cells", type=str, default="B,NK,CD4T,CD8T,Mono,Neutro")
    p.add_argument("--aux-age-weight", type=float, default=12.0)
    p.add_argument("--aux-cell-weight", type=float, default=1.0)
    p.add_argument("--aux-contrast-weight", type=float, default=1.0)
    p.add_argument("--contrast-tau", type=float, default=0.07)
    p.add_argument("--contrast-proj-dim", type=int, default=64)
    p.add_argument("--contrast-hidden", type=int, default=32)
    p.add_argument("--contrast-warmup-epochs", type=int, default=10)
    p.add_argument("--init-from-baseline", type=str, default=None,
                   help="Warm-start fused head from baseline aesurv_aux_model.pt")
    p.add_argument("--init-from", type=str, default=None,
                   help="Warm-start from any .pt (overrides --init-from-baseline if set).")
    p.add_argument("--top-k-whi-checkpoints", type=int, default=3,
                   help="Save top-K WHI C-index checkpoints (exploratory only).")
    p.add_argument("--enc-hidden", type=str, default="64,32")
    p.add_argument("--dec-hidden", type=str, default="32,64")
    p.add_argument("--z-dim", type=int, default=8)
    p.add_argument("--cohort-hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.40)
    p.add_argument("--alpha-recon", type=float, default=5.0)
    p.add_argument("--beta-kl", type=float, default=0.05)
    p.add_argument("--beta-kl-warmup-epochs", type=int, default=10)
    p.add_argument("--gamma-adv", type=float, default=0.10)
    p.add_argument("--lam-adv-max", type=float, default=1.0)
    p.add_argument("--lam-adv-warmup-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--balance-events", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="runs/aesurv_aux_contrastive_v1")
    p.add_argument("--log-file", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = _resolve_data_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _resolve_data_path(args.log_file) if args.log_file else (out_dir / "aesurv_aux_contrastive.log")
    log = _setup_logger(log_path)
    req = args.device.strip().lower()
    if req == "cuda" or req.startswith("cuda:"):
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            log.warning("CUDA requested but unavailable (%s); using CPU.", torch.__version__)
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("AESURV + contrastive aux | device=%s out=%s", device, out_dir)

    cache_root = _resolve_data_path(args.cache_dir)
    fhs_npz = _resolve_data_path(args.combined_npz)
    fhs_pq = _resolve_data_path(args.meta_parquet)
    whi_npz = _resolve_data_path(args.test_combined_npz)
    whi_pq = _resolve_data_path(args.test_meta_parquet)
    snp_txt = _resolve_data_path(args.snp_columns_txt) if args.snp_columns_txt else None

    log.info("Loading FHS / WHI bundles...")
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        label="FHS", combined_npz=fhs_npz, meta_parquet=fhs_pq,
        snp_columns_txt=snp_txt, max_cpg=args.max_cpg, max_snp=args.max_snp,
        cache_root=cache_root, use_cache=not args.no_cache,
    )
    X_meth_whi, X_snp_whi, t_whi, e_whi, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
        label="WHI_raw", combined_npz=whi_npz, meta_parquet=whi_pq,
        snp_columns_txt=None, max_cpg=args.max_cpg, max_snp=args.max_snp,
        cache_root=cache_root, use_cache=not args.no_cache,
    )
    meth_fhs_names, snp_fhs_names = _get_truncated_feature_names(fhs_npz, snp_txt, n_cpg_fhs, n_snp_fhs)
    meth_whi_names, snp_whi_names = _get_truncated_feature_names(whi_npz, None, n_cpg_whi, n_snp_whi)
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, n_cpg, n_snp = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_fhs_names, snp_fhs_names,
        X_meth_whi, X_snp_whi, meth_whi_names, snp_whi_names,
    )
    d_in = int(n_cpg + n_snp)

    ckpt_path = _resolve_data_path(args.dann_encoder_ckpt)
    npz_path = _resolve_data_path(args.dann_preprocess_npz)
    preproc = InvariantPreprocessor(npz_path).to(device).eval()
    if d_in != preproc.d_in:
        raise SystemExit(f"d_in mismatch: aligned={d_in} vs preproc={preproc.d_in}")
    mini, cfg = load_mini_dann_for_fusion(ckpt_path, map_location=device)
    encoder = DannLatentEncoder(mini.to(device)).eval()
    latent_dim = int(cfg["latent_dim"])
    znp = np.load(npz_path, allow_pickle=False)
    meth_mval = bool(int(znp["meth_as_mvalues"][0])) if "meth_as_mvalues" in znp.files else True
    if args.dann_meth_beta_values:
        meth_mval = False
    log.info("DANN latent_dim=%d | fused input_modality=%s", latent_dim, args.input_modality)

    mean_np = preproc.mean_.detach().cpu().numpy().astype(np.float32)
    latent_cache = _resolve_data_path(args.latent_cache_dir)
    use_cache = not args.no_cache

    log.info("Extracting fused DANN latents (%s)...", args.input_modality)
    X_meth_fhs_b, X_snp_fhs_b = _apply_dann_input_modality(
        X_meth_fhs, X_snp_fhs, args.input_modality, mean_np, n_cpg, n_snp, log, "FHS",
    )
    X_meth_whi_b, X_snp_whi_b = _apply_dann_input_modality(
        X_meth_whi, X_snp_whi, args.input_modality, mean_np, n_cpg, n_snp, log, "WHI",
    )
    latent_tag = "" if args.input_modality == "both" else str(args.input_modality)
    mu_fhs = load_or_extract_latent(
        "FHS", X_meth_fhs_b, X_snp_fhs_b, preproc, encoder, meth_as_mvalues=meth_mval,
        latent_dim=latent_dim, device=device, cache_dir=latent_cache,
        ckpt_path=ckpt_path, npz_path=npz_path, use_cache=use_cache, log=log, extra_tag=latent_tag,
    )
    mu_whi = load_or_extract_latent(
        "WHI", X_meth_whi_b, X_snp_whi_b, preproc, encoder, meth_as_mvalues=meth_mval,
        latent_dim=latent_dim, device=device, cache_dir=latent_cache,
        ckpt_path=ckpt_path, npz_path=npz_path, use_cache=use_cache, log=log, extra_tag=latent_tag,
    )

    log.info("Extracting unimodal DANN latents for contrastive (cpgs_only / snps_only)...")
    mu_fhs_cpg = _extract_modality_latents(
        "FHS", X_meth_fhs, X_snp_fhs, "cpgs_only", mean_np, n_cpg, n_snp,
        preproc, encoder, meth_mval, latent_dim, device, latent_cache, ckpt_path, npz_path, use_cache, log,
    )
    mu_whi_cpg = _extract_modality_latents(
        "WHI", X_meth_whi, X_snp_whi, "cpgs_only", mean_np, n_cpg, n_snp,
        preproc, encoder, meth_mval, latent_dim, device, latent_cache, ckpt_path, npz_path, use_cache, log,
    )
    mu_fhs_snp = _extract_modality_latents(
        "FHS", X_meth_fhs, X_snp_fhs, "snps_only", mean_np, n_cpg, n_snp,
        preproc, encoder, meth_mval, latent_dim, device, latent_cache, ckpt_path, npz_path, use_cache, log,
    )
    mu_whi_snp = _extract_modality_latents(
        "WHI", X_meth_whi, X_snp_whi, "snps_only", mean_np, n_cpg, n_snp,
        preproc, encoder, meth_mval, latent_dim, device, latent_cache, ckpt_path, npz_path, use_cache, log,
    )

    cell_cols = [c.strip() for c in args.aux_cells.split(",") if c.strip()]
    age_fhs, cell_fhs = load_aux_targets(
        fhs_pq, _resolve_data_path(args.fhs_cell_comp_parquet), args.fhs_id_col, cell_cols, log, "FHS",
    )
    age_whi, cell_whi = load_aux_targets(
        whi_pq, _resolve_data_path(args.whi_cell_comp_parquet), args.whi_id_col, cell_cols, log, "WHI",
    )

    age_all = np.concatenate([age_fhs, age_whi])
    cell_all = np.concatenate([cell_fhs, cell_whi], axis=0)
    age_mu_g = float(age_all.mean())
    age_sd_g = float(age_all.std() + 1e-6)
    cell_mu_g = cell_all.mean(axis=0)
    cell_sd_g = cell_all.std(axis=0) + 1e-6
    age_fhs_z = ((age_fhs - age_mu_g) / age_sd_g).astype(np.float32)
    age_whi_z = ((age_whi - age_mu_g) / age_sd_g).astype(np.float32)
    cell_fhs_z = ((cell_fhs - cell_mu_g) / cell_sd_g).astype(np.float32)
    cell_whi_z = ((cell_whi - cell_mu_g) / cell_sd_g).astype(np.float32)

    tr_idx, va_idx = _split_indices(mu_fhs.shape[0], args.val_frac, args.seed, stratify_event=e_fhs.astype(np.int32))
    log.info("FHS train=%d val=%d | WHI test=%d", len(tr_idx), len(va_idx), len(e_whi))

    Mu_tr = torch.from_numpy(mu_fhs[tr_idx]).float()
    Mu_cpg_tr = torch.from_numpy(mu_fhs_cpg[tr_idx]).float()
    Mu_snp_tr = torch.from_numpy(mu_fhs_snp[tr_idx]).float()
    Mu_va = torch.from_numpy(mu_fhs[va_idx]).float()
    Mu_te = torch.from_numpy(mu_whi).float()
    T_tr = torch.from_numpy(t_fhs[tr_idx]).float()
    E_tr = torch.from_numpy(e_fhs[tr_idx]).float()
    T_va = torch.from_numpy(t_fhs[va_idx]).float()
    E_va = torch.from_numpy(e_fhs[va_idx]).float()
    T_te = torch.from_numpy(t_whi).float()
    E_te = torch.from_numpy(e_whi).float()
    Age_tr_z = torch.from_numpy(age_fhs_z[tr_idx])
    Cell_tr_z = torch.from_numpy(cell_fhs_z[tr_idx])

    ds_tr = TensorDataset(Mu_tr, Mu_cpg_tr, Mu_snp_tr, T_tr, E_tr, Age_tr_z, Cell_tr_z)
    if args.balance_events:
        w = np.where(E_tr.numpy() > 0.5, 1.0 / max(1, int(E_tr.sum().item())),
                     1.0 / max(1, len(E_tr) - int(E_tr.sum().item())))
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(w), replacement=True)
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)

    Mu_adv = torch.from_numpy(np.concatenate([mu_fhs, mu_whi], axis=0)).float()
    Mu_cpg_adv = torch.from_numpy(np.concatenate([mu_fhs_cpg, mu_whi_cpg], axis=0)).float()
    Mu_snp_adv = torch.from_numpy(np.concatenate([mu_fhs_snp, mu_whi_snp], axis=0)).float()
    C_adv = torch.from_numpy(np.concatenate([
        np.zeros(mu_fhs.shape[0], dtype=np.int64),
        np.ones(mu_whi.shape[0], dtype=np.int64),
    ]))
    Age_adv_z = torch.from_numpy(np.concatenate([age_fhs_z, age_whi_z]))
    Cell_adv_z = torch.from_numpy(np.concatenate([cell_fhs_z, cell_whi_z]))
    w_adv = np.where(C_adv.numpy() == 0, 1.0 / mu_fhs.shape[0], 1.0 / mu_whi.shape[0])
    sampler_adv = WeightedRandomSampler(
        torch.from_numpy(w_adv).double(),
        num_samples=max(args.batch_size * 8, len(C_adv) // 4),
        replacement=True,
    )
    loader_adv = DataLoader(
        TensorDataset(Mu_adv, Mu_cpg_adv, Mu_snp_adv, C_adv, Age_adv_z, Cell_adv_z),
        batch_size=args.batch_size, sampler=sampler_adv, num_workers=0,
    )

    enc_h = tuple(int(x) for x in args.enc_hidden.split(",") if x.strip())
    dec_h = tuple(int(x) for x in args.dec_hidden.split(",") if x.strip())
    model = AESurvHeadAuxContrastive(
        in_dim=latent_dim, enc_hidden=enc_h, dec_hidden=dec_h, z_dim=args.z_dim,
        cohort_hidden=args.cohort_hidden, dropout=args.dropout, n_cells=len(cell_cols),
        contrast_proj_dim=args.contrast_proj_dim, contrast_hidden=args.contrast_hidden,
        contrast_tau=args.contrast_tau,
    ).to(device)

    init_path = args.init_from or args.init_from_baseline
    if init_path:
        _load_baseline_weights(model, _resolve_data_path(init_path), log)
    else:
        log.info("No --init-from / --init-from-baseline; training head from scratch.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log.info("Params=%d contrast_weight=%.2f tau=%.3f", sum(p.numel() for p in model.parameters()),
             args.aux_contrast_weight, args.contrast_tau)

    metrics_path = out_dir / "aesurv_aux_contrastive_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_c_va = -1.0
    best_epoch = -1
    best_state: Optional[dict] = None
    bad_streak = 0
    whi_topk = _TopKWhiCheckpoints(args.top_k_whi_checkpoints)
    max_whi_seen = -1.0
    max_whi_epoch = -1
    adv_iter = iter(loader_adv)

    @torch.no_grad()
    def eval_split(Mu_t: torch.Tensor) -> np.ndarray:
        _, _, log_h, _, _, _, _, _, _ = model(Mu_t.to(device), sample_z=False)
        return log_h.cpu().numpy()

    for epoch in range(args.epochs):
        beta_kl = args.beta_kl * min(1.0, (epoch + 1) / max(1, args.beta_kl_warmup_epochs))
        lam_adv = args.lam_adv_max * min(1.0, (epoch + 1) / max(1, args.lam_adv_warmup_epochs))
        w_ctr = _contrast_scale(epoch, args.contrast_warmup_epochs, args.aux_contrast_weight)

        model.train()
        sums = {k: 0.0 for k in ("total", "cox", "rec", "kl", "adv", "age", "cell", "contrast")}
        n_seen = 0
        for x, xc, xs, t, e, age_z, cell_z in loader_tr:
            if int(e.sum().item()) == 0:
                continue
            x, xc, xs = x.to(device), xc.to(device), xs.to(device)
            t, e = t.to(device), e.to(device)
            age_z, cell_z = age_z.to(device), cell_z.to(device)
            try:
                xa, xac, xas, ca, aa, ka = next(adv_iter)
            except StopIteration:
                adv_iter = iter(loader_adv)
                xa, xac, xas, ca, aa, ka = next(adv_iter)
            xa, xac, xas = xa.to(device), xac.to(device), xas.to(device)
            ca, aa, ka = ca.to(device), aa.to(device), ka.to(device)

            opt.zero_grad(set_to_none=True)
            x_rec, z, log_h, mu_h, logvar_h, age_pred, cell_pred, p_cpg, p_snp = model(
                x, sample_z=True, x_cpg=xc, x_snp=xs,
            )
            cox = cox_ph_loss(log_h, t, e)
            rec = F.smooth_l1_loss(x_rec, x)
            kl = (-0.5 * (1.0 + logvar_h - mu_h.pow(2) - logvar_h.exp())).sum(dim=1).mean()
            aux_age_fhs = F.mse_loss(age_pred, age_z)
            aux_cell_fhs = F.mse_loss(cell_pred, cell_z)
            ctr_fhs = model.info_nce_loss(p_cpg, p_snp, model.contrast_tau)

            _, z_adv, _, _, _, age_pb, cell_pb, p_cpg_a, p_snp_a = model(
                xa, sample_z=True, x_cpg=xac, x_snp=xas,
            )
            adv = F.cross_entropy(model.cohort_logits(z_adv, lam=lam_adv), ca)
            aux_age = 0.5 * (aux_age_fhs + F.mse_loss(age_pb, aa))
            aux_cell = 0.5 * (aux_cell_fhs + F.mse_loss(cell_pb, ka))
            ctr_bal = model.info_nce_loss(p_cpg_a, p_snp_a, model.contrast_tau)
            contrast = 0.5 * (ctr_fhs + ctr_bal)

            loss = (
                cox
                + args.alpha_recon * rec
                + beta_kl * kl
                + args.gamma_adv * adv
                + args.aux_age_weight * aux_age
                + args.aux_cell_weight * aux_cell
                + w_ctr * contrast
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            bs = x.size(0)
            n_seen += bs
            sums["total"] += float(loss.item()) * bs
            sums["cox"] += float(cox.item()) * bs
            sums["rec"] += float(rec.item()) * bs
            sums["kl"] += float(kl.item()) * bs
            sums["adv"] += float(adv.item()) * bs
            sums["age"] += float(aux_age.item()) * bs
            sums["cell"] += float(aux_cell.item()) * bs
            sums["contrast"] += float(contrast.item()) * bs

        denom = max(1, n_seen)
        tr = {k: sums[k] / denom for k in sums}

        model.eval()
        log_h_va = eval_split(Mu_va)
        log_h_te = eval_split(Mu_te)
        log_h_tr = eval_split(Mu_tr)
        c_tr = harrell_c_index(T_tr.numpy(), E_tr.numpy().astype(np.int32), log_h_tr)
        c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), log_h_va)
        c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), log_h_te)

        if (not math.isnan(c_te)) and c_te > max_whi_seen:
            max_whi_seen, max_whi_epoch = float(c_te), epoch + 1
        whi_topk.consider(c_te, epoch + 1, model.state_dict())

        rec_d = {
            "epoch": epoch + 1,
            "train_total": tr["total"],
            "train_cox": tr["cox"],
            "train_recon": tr["rec"],
            "train_kl": tr["kl"],
            "train_adv": tr["adv"],
            "train_age": tr["age"],
            "train_cell": tr["cell"],
            "train_contrast": tr["contrast"],
            "w_contrast": w_ctr,
            "val_cindex": c_va,
            "test_whi_cindex": c_te,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_d) + "\n")

        log.info(
            "ep %03d c_va=%.4f c_whi=%.4f contrast=%.4f | cox=%.3f w_ctr=%.2f",
            epoch + 1, c_va, c_te, tr["contrast"], tr["cox"], w_ctr,
        )

        if (not math.isnan(c_va)) and c_va > best_c_va + 1e-4:
            best_c_va = c_va
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_streak = 0
        else:
            bad_streak += 1
            if epoch + 1 >= args.min_epochs and bad_streak >= args.patience:
                log.info("Early stop epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    log_h_va = eval_split(Mu_va)
    log_h_te = eval_split(Mu_te)
    c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), log_h_va)
    c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), log_h_te)

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "variant": "AESurvHeadAuxContrastive",
            "in_dim": latent_dim,
            "enc_hidden": list(enc_h),
            "dec_hidden": list(dec_h),
            "z_dim": int(args.z_dim),
            "contrast_proj_dim": int(args.contrast_proj_dim),
            "contrast_tau": float(args.contrast_tau),
            "aux_contrast_weight": float(args.aux_contrast_weight),
            "cell_columns": cell_cols,
        },
        "best_epoch": int(best_epoch),
        "best_val_cindex": float(best_c_va),
        "final_whi_cindex": float(c_te),
    }, out_dir / "aesurv_aux_contrastive_model.pt")

    for rank, (c_whi, ep, st) in enumerate(whi_topk.best_list(), start=1):
        torch.save(
            {"state_dict": st, "whi_cindex": c_whi, "epoch": ep, "exploratory": True},
            out_dir / f"aesurv_aux_contrastive_whi_top{rank}_ep{ep}.pt",
        )

    meta = {
        "variant": "contrastive_aux",
        "best_val_cindex": float(best_c_va),
        "best_epoch": int(best_epoch),
        "final_whi_cindex": float(c_te),
        "max_whi_cindex_during_train": float(max_whi_seen),
        "max_whi_epoch": int(max_whi_epoch),
        "whi_topk_exploratory": [
            {"rank": i + 1, "whi_cindex": float(a[0]), "epoch": int(a[1])}
            for i, a in enumerate(whi_topk.best_list())
        ],
        "aux_contrast_weight": float(args.aux_contrast_weight),
        "contrast_tau": float(args.contrast_tau),
        "contrast_warmup_epochs": int(args.contrast_warmup_epochs),
        "contrast_proj_dim": int(args.contrast_proj_dim),
        "aux_age_weight": float(args.aux_age_weight),
        "aux_cell_weight": float(args.aux_cell_weight),
        "init_from": init_path,
        "dann_encoder_ckpt": str(ckpt_path),
        "dann_preprocess_npz": str(npz_path),
    }
    meta.update(_survival_ipcw_td_auroc_auprc(
        T_va.numpy(), E_va.numpy().astype(np.int32), T_va.numpy(), E_va.numpy().astype(np.int32),
        np.asarray(log_h_va, dtype=np.float64), "fhs_val", log=log,
    ))
    meta.update(_survival_ipcw_td_auroc_auprc(
        T_te.numpy(), E_te.numpy().astype(np.int32), T_te.numpy(), E_te.numpy().astype(np.int32),
        np.asarray(log_h_te, dtype=np.float64), "whi_test", log=log,
    ))
    (out_dir / "aesurv_aux_contrastive_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Done. BEST c_va=%.4f c_whi=%.4f -> %s",
             best_c_va, c_te, out_dir / "aesurv_aux_contrastive_model.pt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
