#!/usr/bin/env python3
"""Joint full MiniVAEDANN + AESurv contrastive head (multi-task, raw omics forward)."""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from aesurv_domain import vae_kl_loss
from experiments.aesurv_contrastive.aesurv_head_aux_contrastive import AESurvHeadAuxContrastive
from experiments.aesurv_contrastive.joint_dann_aesurv_model import JointDannAesurvContrastive
from mini_vae_dann_pipeline import cohort_mmd, fhs_batch_ids, fhs_batch_mmd
from train_aesurv_dann_latent_aux import (
    _survival_ipcw_td_auroc_auprc,
    load_aux_targets,
)
from train_aesurv_dann_latent_aux_contrastive import _TopKWhiCheckpoints, _contrast_scale
from train_dann_survival import cox_ph_loss, harrell_c_index, _setup_logger, _split_indices
from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    _resolve_data_path,
    load_bundle_with_cache,
)


def _resolve_device(device_str: str, log: logging.Logger) -> torch.device:
    req = device_str.strip().lower()
    if req == "cuda" or req.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        log.warning("CUDA requested but unavailable (%s); using CPU.", torch.__version__)
        return torch.device("cpu")
    return torch.device(device_str)


def _parse_bandwidths(s: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _load_head_init(head: AESurvHeadAuxContrastive, path: Path, log: logging.Logger) -> None:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    if any(k.startswith("head.") for k in state):
        state = {k.replace("head.", "", 1): v for k, v in state.items() if k.startswith("head.")}
    elif any(k.startswith("dann.") for k in state):
        state = {k: v for k, v in state.items() if k.startswith("head.")}
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing:
        log.info("Head init: %d missing keys (expected for new DANN): %s", len(missing), missing[:6])
    if unexpected:
        log.warning("Head init: unexpected keys: %s", unexpected[:6])


def _load_joint_init(model: JointDannAesurvContrastive, path: Path, log: logging.Logger) -> None:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info(
        "Joint init from %s | missing=%d unexpected=%d",
        path, len(missing), len(unexpected),
    )
    if missing:
        log.info("  missing (first): %s", missing[:8])
    if unexpected:
        log.warning("  unexpected (first): %s", unexpected[:8])


@torch.no_grad()
def _eval_cindex_raw(
    model: JointDannAesurvContrastive,
    x_meth: np.ndarray,
    x_snp: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> float:
    model.eval()
    risks: List[np.ndarray] = []
    for s in range(0, x_meth.shape[0], batch_size):
        e_idx = min(s + batch_size, x_meth.shape[0])
        m = torch.from_numpy(np.ascontiguousarray(x_meth[s:e_idx])).to(device)
        v = torch.from_numpy(np.ascontiguousarray(x_snp[s:e_idx])).to(device)
        log_h = model.predict_log_h(m, v, input_modality="both")
        risks.append(log_h.cpu().numpy())
    return harrell_c_index(t, e.astype(np.int32), np.concatenate(risks, axis=0))


def main() -> None:
    p = argparse.ArgumentParser(description="Joint DANN + AESurv contrastive training.")
    p.add_argument("--combined-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--meta-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--test-combined-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--test-meta-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--max-cpg", type=int, default=None)
    p.add_argument("--max-snp", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default="vae_cox_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dann-encoder-ckpt", type=str, required=True)
    p.add_argument("--dann-preprocess-npz", type=str, required=True)
    p.add_argument("--dann-meth-beta-values", action="store_true")
    p.add_argument("--init-head", type=str, default=None, help="Warm-start AESurv head (.pt)")
    p.add_argument(
        "--init-joint",
        type=str,
        default=None,
        help="Warm-start full JointDannAesurvContrastive state_dict (.pt); overrides --init-head",
    )
    p.add_argument("--fhs-cell-comp-parquet", type=str, default="FHS_cell_composition.parquet")
    p.add_argument("--whi-cell-comp-parquet", type=str, default="WHI_cell_composition.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--whi-id-col", type=str, default="sample_ID")
    p.add_argument("--batch-label-col", type=str, default="batch")
    p.add_argument("--aux-cells", type=str, default="B,NK,CD4T,CD8T,Mono,Neutro")
    p.add_argument("--aux-age-weight", type=float, default=12.0)
    p.add_argument("--aux-cell-weight", type=float, default=0.5)
    p.add_argument("--aux-contrast-weight", type=float, default=1.0)
    p.add_argument("--contrast-tau", type=float, default=0.07)
    p.add_argument("--contrast-proj-dim", type=int, default=64)
    p.add_argument("--contrast-hidden", type=int, default=32)
    p.add_argument("--contrast-warmup-epochs", type=int, default=10)
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
    p.add_argument("--w-recon", type=float, default=0.5)
    p.add_argument("--w-dom", type=float, default=0.6)
    p.add_argument("--w-mmd", type=float, default=5.0)
    p.add_argument("--w-mmd-batch", type=float, default=2.0)
    p.add_argument("--w-kl-dann", type=float, default=0.0)
    p.add_argument("--w-batch-adv", type=float, default=1.0)
    p.add_argument("--domain-loss-scale", type=float, default=1.0)
    p.add_argument("--domain-pos-weight", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mmd-bandwidths", type=str, default="1,2,4,8,16")
    p.add_argument("--lr-dann", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--balance-events", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="runs/aesurv_joint_dann_contrastive_a3")
    p.add_argument("--top-k-whi-checkpoints", type=int, default=3)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = _resolve_data_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logger(out_dir / "joint_dann_aesurv.log")
    device = _resolve_device(args.device, log)
    log.info("Joint DANN+AESurv contrastive | device=%s out=%s", device, out_dir)

    cache_root = _resolve_data_path(args.cache_dir)
    fhs_npz = _resolve_data_path(args.combined_npz)
    fhs_pq = _resolve_data_path(args.meta_parquet)
    whi_npz = _resolve_data_path(args.test_combined_npz)
    whi_pq = _resolve_data_path(args.test_meta_parquet)

    log.info("Loading FHS / WHI bundles...")
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        label="FHS", combined_npz=fhs_npz, meta_parquet=fhs_pq,
        snp_columns_txt=None, max_cpg=args.max_cpg, max_snp=args.max_snp,
        cache_root=cache_root, use_cache=not args.no_cache,
    )
    X_meth_whi, X_snp_whi, t_whi, e_whi, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
        label="WHI_raw", combined_npz=whi_npz, meta_parquet=whi_pq,
        snp_columns_txt=None, max_cpg=args.max_cpg, max_snp=args.max_snp,
        cache_root=cache_root, use_cache=not args.no_cache,
    )
    meth_fhs_names, snp_fhs_names = _get_truncated_feature_names(fhs_npz, None, n_cpg_fhs, n_snp_fhs)
    meth_whi_names, snp_whi_names = _get_truncated_feature_names(whi_npz, None, n_cpg_whi, n_snp_whi)
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, n_cpg, n_snp = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_fhs_names, snp_fhs_names,
        X_meth_whi, X_snp_whi, meth_whi_names, snp_whi_names,
    )

    npz_path = _resolve_data_path(args.dann_preprocess_npz)
    znp = np.load(npz_path, allow_pickle=False)
    meth_mval = bool(int(znp["meth_as_mvalues"][0])) if "meth_as_mvalues" in znp.files else True
    if args.dann_meth_beta_values:
        meth_mval = False

    enc_h = tuple(int(x) for x in args.enc_hidden.split(",") if x.strip())
    dec_h = tuple(int(x) for x in args.dec_hidden.split(",") if x.strip())
    cell_cols = [c.strip() for c in args.aux_cells.split(",") if c.strip()]

    ckpt_path = _resolve_data_path(args.dann_encoder_ckpt)
    _mini_cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    latent_dim_init = int(_mini_cfg["latent_dim"])

    head = AESurvHeadAuxContrastive(
        in_dim=latent_dim_init, enc_hidden=enc_h, dec_hidden=dec_h, z_dim=args.z_dim,
        cohort_hidden=args.cohort_hidden, dropout=args.dropout, n_cells=len(cell_cols),
        contrast_proj_dim=args.contrast_proj_dim, contrast_hidden=args.contrast_hidden,
        contrast_tau=args.contrast_tau,
    )
    if args.init_head and not args.init_joint:
        _load_head_init(head, _resolve_data_path(args.init_head), log)

    model, cfg = JointDannAesurvContrastive.from_checkpoints(
        str(ckpt_path), str(npz_path), head,
        n_cpg=n_cpg, n_snp=n_snp, meth_as_mvalues=meth_mval,
        map_location=device,
    )
    latent_dim = int(cfg["latent_dim"])
    if head.in_dim != latent_dim:
        raise SystemExit(f"head in_dim {head.in_dim} != dann latent_dim {latent_dim}")
    model = model.to(device)
    if args.init_joint:
        _load_joint_init(model, _resolve_data_path(args.init_joint), log)

    age_fhs, cell_fhs = load_aux_targets(
        fhs_pq, _resolve_data_path(args.fhs_cell_comp_parquet), args.fhs_id_col, cell_cols, log, "FHS",
    )
    age_whi, cell_whi = load_aux_targets(
        whi_pq, _resolve_data_path(args.whi_cell_comp_parquet), args.whi_id_col, cell_cols, log, "WHI",
    )
    age_all = np.concatenate([age_fhs, age_whi])
    cell_all = np.concatenate([cell_fhs, cell_whi], axis=0)
    age_mu_g, age_sd_g = float(age_all.mean()), float(age_all.std() + 1e-6)
    cell_mu_g, cell_sd_g = cell_all.mean(axis=0), cell_all.std(axis=0) + 1e-6
    age_fhs_z = ((age_fhs - age_mu_g) / age_sd_g).astype(np.float32)
    age_whi_z = ((age_whi - age_mu_g) / age_sd_g).astype(np.float32)
    cell_fhs_z = ((cell_fhs - cell_mu_g) / cell_sd_g).astype(np.float32)
    cell_whi_z = ((cell_whi - cell_mu_g) / cell_sd_g).astype(np.float32)

    try:
        fhs_batch_all, _ = fhs_batch_ids(fhs_pq, args.batch_label_col, X_meth_fhs.shape[0])
    except (SystemExit, ValueError, KeyError) as ex:
        log.warning("FHS batch ids unavailable (%s); batch MMD/adv disabled.", ex)
        fhs_batch_all = np.zeros(X_meth_fhs.shape[0], dtype=np.int64)

    tr_idx, va_idx = _split_indices(X_meth_fhs.shape[0], args.val_frac, args.seed, stratify_event=e_fhs.astype(np.int32))
    log.info("FHS train=%d val=%d | WHI test=%d", len(tr_idx), len(va_idx), len(e_whi))

    Xm_tr = torch.from_numpy(np.ascontiguousarray(X_meth_fhs[tr_idx])).float()
    Xs_tr = torch.from_numpy(np.ascontiguousarray(X_snp_fhs[tr_idx])).float()
    T_tr = torch.from_numpy(t_fhs[tr_idx]).float()
    E_tr = torch.from_numpy(e_fhs[tr_idx]).float()
    Age_tr = torch.from_numpy(age_fhs_z[tr_idx])
    Cell_tr = torch.from_numpy(cell_fhs_z[tr_idx])

    ds_tr = TensorDataset(Xm_tr, Xs_tr, T_tr, E_tr, Age_tr, Cell_tr)
    if args.balance_events:
        w = np.where(E_tr.numpy() > 0.5, 1.0 / max(1, int(E_tr.sum().item())),
                     1.0 / max(1, len(E_tr) - int(E_tr.sum().item())))
        loader_tr = DataLoader(
            ds_tr, batch_size=args.batch_size,
            sampler=WeightedRandomSampler(torch.from_numpy(w).double(), len(w), replacement=True),
            num_workers=0,
        )
    else:
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)

    n_fhs, n_whi = X_meth_fhs.shape[0], X_meth_whi.shape[0]
    Xm_adv = torch.from_numpy(np.ascontiguousarray(
        np.concatenate([X_meth_fhs, X_meth_whi], axis=0)
    )).float()
    Xs_adv = torch.from_numpy(np.ascontiguousarray(
        np.concatenate([X_snp_fhs, X_snp_whi], axis=0)
    )).float()
    C_adv = torch.from_numpy(np.concatenate([
        np.zeros(n_fhs, dtype=np.int64), np.ones(n_whi, dtype=np.int64),
    ]))
    B_adv = torch.from_numpy(np.concatenate([
        fhs_batch_all.astype(np.int64), np.full(n_whi, -1, dtype=np.int64),
    ]))
    Age_adv = torch.from_numpy(np.concatenate([age_fhs_z, age_whi_z]))
    Cell_adv = torch.from_numpy(np.concatenate([cell_fhs_z, cell_whi_z]))
    w_adv = np.where(C_adv.numpy() == 0, 1.0 / n_fhs, 1.0 / n_whi)
    loader_adv = DataLoader(
        TensorDataset(Xm_adv, Xs_adv, C_adv, B_adv, Age_adv, Cell_adv),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            torch.from_numpy(w_adv).double(),
            max(args.batch_size * 8, (n_fhs + n_whi) // 4),
            replacement=True,
        ),
        num_workers=0,
    )

    dom_pw = args.domain_pos_weight
    if dom_pw is None:
        dom_pw = float(n_fhs) / max(1.0, float(n_whi))
    domain_pw_t = torch.tensor([dom_pw], dtype=torch.float32, device=device)
    mmd_bw = _parse_bandwidths(args.mmd_bandwidths)
    n_fhs_batches = int(model.dann.n_fhs_batches)

    opt = torch.optim.AdamW(
        [
            {"params": model.dann.parameters(), "lr": args.lr_dann},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    n_dann = sum(p.numel() for p in model.dann.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    log.info("Trainable dann=%d head=%d | w_recon=%.2f w_mmd=%.2f", n_dann, n_head, args.w_recon, args.w_mmd)

    metrics_path = out_dir / "joint_dann_aesurv_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_c_va, best_epoch, best_state = -1.0, -1, None
    bad_streak = 0
    whi_topk = _TopKWhiCheckpoints(args.top_k_whi_checkpoints)
    max_whi_seen, max_whi_epoch = -1.0, -1
    adv_iter = iter(loader_adv)

    for epoch in range(args.epochs):
        beta_kl = args.beta_kl * min(1.0, (epoch + 1) / max(1, args.beta_kl_warmup_epochs))
        lam_adv = args.lam_adv_max * min(1.0, (epoch + 1) / max(1, args.lam_adv_warmup_epochs))
        w_ctr = _contrast_scale(epoch, args.contrast_warmup_epochs, args.aux_contrast_weight)
        model.dann.grl.set_lambda(lam_adv)

        model.train()
        sums = {k: 0.0 for k in (
            "total", "cox", "rec", "kl", "adv", "age", "cell", "contrast",
            "dann_recon", "dann_dom", "dann_mmd", "dann_batch", "dann_kl",
        )}
        n_seen = 0

        for xm, xs, t, e, age_z, cell_z in loader_tr:
            if int(e.sum().item()) == 0:
                continue
            xm, xs = xm.to(device), xs.to(device)
            t, e = t.to(device), e.to(device)
            age_z, cell_z = age_z.to(device), cell_z.to(device)
            try:
                xma, xsa, ca, ba, aa, ka = next(adv_iter)
            except StopIteration:
                adv_iter = iter(loader_adv)
                xma, xsa, ca, ba, aa, ka = next(adv_iter)
            xma, xsa = xma.to(device), xsa.to(device)
            ca = ca.to(device)
            ba = ba.to(device)
            aa, ka = aa.to(device), ka.to(device)

            opt.zero_grad(set_to_none=True)

            mu_b, _, _ = model.encode_raw(xm, xs, "both")
            mu_c, _, _ = model.encode_raw(xm, xs, "cpgs_only")
            mu_s, _, _ = model.encode_raw(xm, xs, "snps_only")
            x_rec, z, log_h, mu_h, logvar_h, age_pred, cell_pred, p_cpg, p_snp = model.forward_head(
                mu_b, mu_c, mu_s, sample_z=True,
            )
            cox = cox_ph_loss(log_h, t, e)
            rec = F.smooth_l1_loss(x_rec, mu_b)
            kl = (-0.5 * (1.0 + logvar_h - mu_h.pow(2) - logvar_h.exp())).sum(dim=1).mean()
            ctr = model.head.info_nce_loss(p_cpg, p_snp, model.head.contrast_tau)
            aux_age = F.mse_loss(age_pred, age_z)
            aux_cell = F.mse_loss(cell_pred, cell_z)

            mu_ab, _, x_proj_a = model.encode_raw(xma, xsa, "both")
            mu_ac, _, _ = model.encode_raw(xma, xsa, "cpgs_only")
            mu_as, _, _ = model.encode_raw(xma, xsa, "snps_only")
            _, z_adv, _, _, _, age_pb, cell_pb, p_cpg_a, p_snp_a = model.forward_head(
                mu_ab, mu_ac, mu_as, sample_z=True,
            )
            adv = F.cross_entropy(model.head.cohort_logits(z_adv, lam=lam_adv), ca)
            aux_age_b = 0.5 * (aux_age + F.mse_loss(age_pb, aa))
            aux_cell_b = 0.5 * (aux_cell + F.mse_loss(cell_pb, ka))
            ctr_b = model.head.info_nce_loss(p_cpg_a, p_snp_a, model.head.contrast_tau)
            contrast = 0.5 * (ctr + ctr_b)

            recon_d, mu_d, logvar_d, dom_logit, batch_logits = model.forward_dann_on_proj(x_proj_a)
            dann_recon = F.mse_loss(recon_d, x_proj_a)
            y_f = ca.float()
            if args.label_smoothing > 0:
                y_f = y_f * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            dann_dom = F.binary_cross_entropy_with_logits(dom_logit, y_f, pos_weight=domain_pw_t)
            dann_mmd = cohort_mmd(mu_d, ca, mmd_bw, use_median=True)
            mmd_batch_only = fhs_batch_mmd(mu_d, ca, ba, n_fhs_batches, mmd_bw, use_median=True)
            batch_ce = torch.tensor(0.0, device=device)
            if args.w_batch_adv > 0 and batch_logits is not None and n_fhs_batches > 1:
                m_fhs = (ca == 0) & (ba >= 0)
                if torch.any(m_fhs):
                    batch_ce = F.cross_entropy(batch_logits[m_fhs], ba[m_fhs].long())

            dann_kl = vae_kl_loss(mu_d, logvar_d) if args.w_kl_dann > 0 else torch.tensor(0.0, device=device)

            loss_head = (
                cox
                + args.alpha_recon * rec
                + beta_kl * kl
                + args.gamma_adv * adv
                + args.aux_age_weight * aux_age_b
                + args.aux_cell_weight * aux_cell_b
                + w_ctr * contrast
            )
            loss_dann = (
                args.w_recon * dann_recon
                + args.w_dom * args.domain_loss_scale * dann_dom
                + args.w_mmd * dann_mmd
                + args.w_mmd_batch * mmd_batch_only
                + args.w_batch_adv * batch_ce
                + args.w_kl_dann * dann_kl
            )
            loss = loss_head + loss_dann
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            bs = xm.size(0)
            n_seen += bs
            sums["total"] += float(loss.item()) * bs
            sums["cox"] += float(cox.item()) * bs
            sums["rec"] += float(rec.item()) * bs
            sums["kl"] += float(kl.item()) * bs
            sums["adv"] += float(adv.item()) * bs
            sums["age"] += float(aux_age_b.item()) * bs
            sums["cell"] += float(aux_cell_b.item()) * bs
            sums["contrast"] += float(contrast.item()) * bs
            sums["dann_recon"] += float(dann_recon.item()) * bs
            sums["dann_dom"] += float(dann_dom.item()) * bs
            sums["dann_mmd"] += float(dann_mmd.item()) * bs
            sums["dann_batch"] += float((batch_ce + mmd_batch_only).item()) * bs
            sums["dann_kl"] += float(dann_kl.item()) * bs

        denom = max(1, n_seen)
        tr = {k: sums[k] / denom for k in sums}

        c_va = _eval_cindex_raw(
            model, X_meth_fhs[va_idx], X_snp_fhs[va_idx], t_fhs[va_idx], e_fhs[va_idx], device, args.batch_size,
        )
        c_te = _eval_cindex_raw(
            model, X_meth_whi, X_snp_whi, t_whi, e_whi, device, args.batch_size,
        )
        c_tr = _eval_cindex_raw(
            model, X_meth_fhs[tr_idx], X_snp_fhs[tr_idx], t_fhs[tr_idx], e_fhs[tr_idx], device, args.batch_size,
        )

        if (not math.isnan(c_te)) and c_te > max_whi_seen:
            max_whi_seen, max_whi_epoch = float(c_te), epoch + 1
        whi_topk.consider(c_te, epoch + 1, model.state_dict())

        rec = {
            "epoch": epoch + 1,
            "train_total": tr["total"],
            "train_cox": tr["cox"],
            "train_contrast": tr["contrast"],
            "train_dann_mmd": tr["dann_mmd"],
            "val_cindex": c_va,
            "test_whi_cindex": c_te,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log.info(
            "ep %03d c_va=%.4f c_whi=%.4f cox=%.3f dann_mmd=%.4f w_ctr=%.2f",
            epoch + 1, c_va, c_te, tr["cox"], tr["dann_mmd"], w_ctr,
        )

        if (not math.isnan(c_va)) and c_va > best_c_va + 1e-4:
            best_c_va, best_epoch = c_va, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_streak = 0
        else:
            bad_streak += 1
            if epoch + 1 >= args.min_epochs and bad_streak >= args.patience:
                log.info("Early stop epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    c_va = _eval_cindex_raw(model, X_meth_fhs[va_idx], X_snp_fhs[va_idx], t_fhs[va_idx], e_fhs[va_idx], device)
    c_te = _eval_cindex_raw(model, X_meth_whi, X_snp_whi, t_whi, e_whi, device)
    log_h_va_np = []
    va_meth, va_snp = X_meth_fhs[va_idx], X_snp_fhs[va_idx]
    for s in range(0, va_meth.shape[0], args.batch_size):
        e_idx = min(s + args.batch_size, va_meth.shape[0])
        m = torch.from_numpy(np.ascontiguousarray(va_meth[s:e_idx])).float().to(device)
        v = torch.from_numpy(np.ascontiguousarray(va_snp[s:e_idx])).float().to(device)
        log_h_va_np.append(model.predict_log_h(m, v).cpu().numpy())
    log_h_va_np = np.concatenate(log_h_va_np)

    log_h_te_np = []
    for s in range(0, X_meth_whi.shape[0], args.batch_size):
        e_idx = min(s + args.batch_size, X_meth_whi.shape[0])
        m = torch.from_numpy(np.ascontiguousarray(X_meth_whi[s:e_idx])).float().to(device)
        v = torch.from_numpy(np.ascontiguousarray(X_snp_whi[s:e_idx])).float().to(device)
        log_h_te_np.append(model.predict_log_h(m, v).cpu().numpy())
    log_h_te_np = np.concatenate(log_h_te_np)

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "variant": "JointDannAesurvContrastive",
            "latent_dim": latent_dim,
            "n_cpg": n_cpg,
            "n_snp": n_snp,
            "meth_as_mvalues": meth_mval,
            "z_dim": args.z_dim,
            "contrast_tau": args.contrast_tau,
            "w_mmd": args.w_mmd,
            "w_dom": args.w_dom,
            "lr_dann": args.lr_dann,
            "lr_head": args.lr_head,
        },
        "best_epoch": int(best_epoch),
        "best_val_cindex": float(best_c_va),
        "final_whi_cindex": float(c_te),
        "dann_encoder_ckpt": str(ckpt_path),
        "dann_preprocess_npz": str(npz_path),
    }, out_dir / "joint_dann_aesurv_model.pt")

    for rank, (c_whi, ep, st) in enumerate(whi_topk.best_list(), start=1):
        torch.save(
            {"state_dict": st, "whi_cindex": c_whi, "epoch": ep, "exploratory": True},
            out_dir / f"joint_dann_aesurv_whi_top{rank}_ep{ep}.pt",
        )

    meta = {
        "variant": "joint_dann_contrastive",
        "best_val_cindex": float(best_c_va),
        "best_epoch": int(best_epoch),
        "final_whi_cindex": float(c_te),
        "max_whi_cindex_during_train": float(max_whi_seen),
        "max_whi_epoch": int(max_whi_epoch),
        "aux_age_weight": args.aux_age_weight,
        "aux_cell_weight": args.aux_cell_weight,
        "w_recon": args.w_recon,
        "w_mmd": args.w_mmd,
        "w_dom": args.w_dom,
        "init_head": args.init_head,
        "init_joint": args.init_joint,
        "dann_encoder_ckpt": str(ckpt_path),
    }
    meta.update(_survival_ipcw_td_auroc_auprc(
        t_fhs[va_idx], e_fhs[va_idx].astype(np.int32), t_fhs[va_idx], e_fhs[va_idx].astype(np.int32),
        log_h_va_np.astype(np.float64), "fhs_val", log=log,
    ))
    meta.update(_survival_ipcw_td_auroc_auprc(
        t_whi, e_whi.astype(np.int32), t_whi, e_whi.astype(np.int32),
        log_h_te_np.astype(np.float64), "whi_test", log=log,
    ))
    (out_dir / "joint_dann_aesurv_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Done. BEST c_va=%.4f c_whi=%.4f -> %s", best_c_va, c_te, out_dir / "joint_dann_aesurv_model.pt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
