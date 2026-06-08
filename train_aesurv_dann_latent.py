#!/usr/bin/env python3
"""
AESURV-style head on top of frozen DANN latent (single-stage, clean).

Pipeline (input -> output, with grad flow):
    (meth, SNP)  --[FROZEN]-->  DANN mu (128-d)
                                   |
                                   v  (trainable)
                          Head encoder MLP -> mu_h, logvar_h
                                   |
                              reparameterize -> z (small bottleneck)
                                   |          \
                                   |           \-> Cox head -> log-hazard
                                   |           \-> Cohort adversary (GRL)
                                   v
                          Head decoder MLP -> reconstruct DANN-mu

Loss = Cox + alpha * SmoothL1(z_decoded, dann_mu) + beta * KL + gamma * cohort_adv + L2

This adds three regularizers absent from train_dann_survival.py:
  - recon(decoded, dann_mu): preserves DANN info, blocks FHS-only shortcuts
  - KL on z:                  smooth, calibrated bottleneck
  - cohort adversary on z:    one more layer of FHS<->WHI invariance

WHI samples are used **only** for the cohort adversary (no time/event leak).

Usage:
    python train_aesurv_dann_latent.py ^
      --dann-encoder-ckpt   runs/mini_vae_dann_mmd_rich/mini_dann_model.pt ^
      --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz ^
      --out-dir runs/aesurv_dann_rich_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dann_aesurv_bridge import (
    DannLatentEncoder,
    InvariantPreprocessor,
    load_mini_dann_for_fusion,
)
from train_dann_survival import (
    cox_ph_loss,
    extract_latent_mu,
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


# --------------------------------------------------------------------------- #
# Gradient Reversal (for cohort adversary)                                    #
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


# --------------------------------------------------------------------------- #
# AESURV head                                                                  #
# --------------------------------------------------------------------------- #
class AESurvHead(nn.Module):
    """Small VAE + Cox head over a frozen-DANN latent.

    Parameters
    ----------
    in_dim : int
        DANN latent dimension (e.g. 128).
    enc_hidden, dec_hidden : tuple of ints
        Encoder / decoder MLP widths.
    z_dim : int
        Bottleneck size (small; e.g. 16).
    cohort_hidden : int
        Width of the adversarial cohort head.
    dropout : float
        Dropout in encoder / decoder MLPs.
    """

    def __init__(
        self,
        in_dim: int,
        enc_hidden: Tuple[int, ...] = (64, 32),
        dec_hidden: Tuple[int, ...] = (32, 64),
        z_dim: int = 16,
        cohort_hidden: int = 8,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.z_dim = int(z_dim)

        enc_layers: List[nn.Module] = []
        prev = self.in_dim
        for h in enc_hidden:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(prev, self.z_dim)
        self.logvar_head = nn.Linear(prev, self.z_dim)

        dec_layers: List[nn.Module] = []
        prev = self.z_dim
        for h in dec_hidden:
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, self.in_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.cox_head = nn.Linear(self.z_dim, 1)
        self.cohort_head = nn.Sequential(
            nn.Linear(self.z_dim, cohort_hidden),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Dropout(dropout),
            nn.Linear(cohort_hidden, 2),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h).clamp(min=-8.0, max=8.0)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, sample_z: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar) if (sample_z and self.training) else mu
        x_rec = self.decoder(z)
        log_h = self.cox_head(z).squeeze(-1)
        return x_rec, z, log_h, mu, logvar

    def cohort_logits(self, z: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.cohort_head(grad_reverse(z, lam))


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="AESURV head on frozen DANN latent.")
    # data
    p.add_argument("--combined-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--meta-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--test-combined-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--test-meta-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--snp-columns-txt", type=str, default=None)
    p.add_argument("--max-cpg", type=int, default=None)
    p.add_argument("--max-snp", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default="vae_cox_cache")
    p.add_argument("--no-cache", action="store_true")
    # DANN stem
    p.add_argument("--dann-encoder-ckpt", type=str, required=True)
    p.add_argument("--dann-preprocess-npz", type=str, required=True)
    p.add_argument("--dann-meth-beta-values", action="store_true")
    p.add_argument("--latent-cache-dir", type=str, default="vae_cox_cache/dann_latent")
    # AESURV head
    p.add_argument("--enc-hidden", type=str, default="64,32")
    p.add_argument("--dec-hidden", type=str, default="32,64")
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--cohort-hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.30)
    # loss weights
    p.add_argument("--alpha-recon", type=float, default=0.5,
                   help="Weight of SmoothL1 reconstruction of DANN-mu.")
    p.add_argument("--beta-kl", type=float, default=0.01,
                   help="Weight of KL(z | N(0,I)).")
    p.add_argument("--beta-kl-warmup-epochs", type=int, default=10,
                   help="Linear ramp epochs for beta-kl from 0 to its target.")
    p.add_argument("--gamma-adv", type=float, default=0.10,
                   help="Cohort adversary weight (CE).")
    p.add_argument("--lam-adv-max", type=float, default=1.0,
                   help="Max gradient-reversal coefficient.")
    p.add_argument("--lam-adv-warmup-epochs", type=int, default=15)
    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--balance-events", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # output
    p.add_argument("--out-dir", type=str, default="runs/aesurv_dann_latent")
    p.add_argument("--log-file", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    out_dir = _resolve_data_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _resolve_data_path(args.log_file) if args.log_file else (out_dir / "aesurv_dann.log")
    log = _setup_logger(log_path)
    log.info("device=%s out_dir=%s", device, out_dir)

    # --- data ---
    cache_root = _resolve_data_path(args.cache_dir)
    fhs_npz = _resolve_data_path(args.combined_npz); fhs_pq = _resolve_data_path(args.meta_parquet)
    whi_npz = _resolve_data_path(args.test_combined_npz); whi_pq = _resolve_data_path(args.test_meta_parquet)
    snp_txt = _resolve_data_path(args.snp_columns_txt) if args.snp_columns_txt else None

    log.info("Loading FHS bundle...")
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        label="FHS", combined_npz=fhs_npz, meta_parquet=fhs_pq,
        snp_columns_txt=snp_txt, max_cpg=args.max_cpg, max_snp=args.max_snp,
        cache_root=cache_root, use_cache=not args.no_cache,
    )
    log.info("Loading WHI bundle...")
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
    log.info("Aligned: n_cpg=%d n_snp=%d D=%d | FHS %d WHI %d",
             n_cpg, n_snp, d_in, X_meth_fhs.shape[0], X_meth_whi.shape[0])

    # --- frozen DANN ---
    ckpt_path = _resolve_data_path(args.dann_encoder_ckpt)
    npz_path = _resolve_data_path(args.dann_preprocess_npz)
    preproc = InvariantPreprocessor(npz_path).to(device).eval()
    if d_in != preproc.d_in:
        raise SystemExit(f"d_in mismatch: aligned={d_in} vs preproc={preproc.d_in}.")
    mini, cfg = load_mini_dann_for_fusion(ckpt_path, map_location=device)
    encoder = DannLatentEncoder(mini.to(device)).eval()
    latent_dim = int(cfg["latent_dim"])
    znp = np.load(npz_path, allow_pickle=False)
    meth_mval = bool(int(znp["meth_as_mvalues"][0])) if "meth_as_mvalues" in znp.files else True
    if args.dann_meth_beta_values:
        meth_mval = False
    log.info("DANN: latent_dim=%d proj_dim=%d meth_as_mvalues=%s",
             latent_dim, int(cfg["proj_dim"]), meth_mval)

    # --- extract DANN mu (cached) ---
    latent_cache = _resolve_data_path(args.latent_cache_dir)
    mu_fhs = load_or_extract_latent(
        "FHS", X_meth_fhs, X_snp_fhs, preproc, encoder,
        meth_as_mvalues=meth_mval, latent_dim=latent_dim, device=device,
        cache_dir=latent_cache, ckpt_path=ckpt_path, npz_path=npz_path,
        use_cache=not args.no_cache, log=log,
    )
    mu_whi = load_or_extract_latent(
        "WHI", X_meth_whi, X_snp_whi, preproc, encoder,
        meth_as_mvalues=meth_mval, latent_dim=latent_dim, device=device,
        cache_dir=latent_cache, ckpt_path=ckpt_path, npz_path=npz_path,
        use_cache=not args.no_cache, log=log,
    )

    # --- FHS train/val split ---
    tr_idx, va_idx = _split_indices(mu_fhs.shape[0], args.val_frac, args.seed,
                                    stratify_event=e_fhs.astype(np.int32))
    log.info("FHS train n=%d (events=%d), val n=%d (events=%d), WHI test n=%d (events=%d)",
             len(tr_idx), int(e_fhs[tr_idx].sum()),
             len(va_idx), int(e_fhs[va_idx].sum()),
             len(e_whi), int(e_whi.sum()))

    # --- tensors ---
    Mu_tr = torch.from_numpy(mu_fhs[tr_idx]).float()
    Mu_va = torch.from_numpy(mu_fhs[va_idx]).float()
    Mu_te = torch.from_numpy(mu_whi).float()
    T_tr = torch.from_numpy(t_fhs[tr_idx]).float(); E_tr = torch.from_numpy(e_fhs[tr_idx]).float()
    T_va = torch.from_numpy(t_fhs[va_idx]).float(); E_va = torch.from_numpy(e_fhs[va_idx]).float()
    T_te = torch.from_numpy(t_whi).float(); E_te = torch.from_numpy(e_whi).float()

    # --- FHS supervised loader (Cox + recon) ---
    ds_tr = TensorDataset(Mu_tr, T_tr, E_tr)
    if args.balance_events:
        w = np.where(E_tr.numpy() > 0.5, 1.0 / max(1, int(E_tr.sum().item())),
                     1.0 / max(1, len(E_tr) - int(E_tr.sum().item())))
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(w), replacement=True)
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=False)
        log.info("Train sampling: balanced events")
    else:
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        log.info("Train sampling: shuffle")

    # --- adversary loader: FHS + WHI features only, with cohort label ---
    Mu_adv = torch.from_numpy(np.concatenate([mu_fhs, mu_whi], axis=0)).float()
    C_adv = torch.from_numpy(np.concatenate([
        np.zeros(mu_fhs.shape[0], dtype=np.int64),
        np.ones(mu_whi.shape[0], dtype=np.int64),
    ], axis=0))
    # Balance FHS/WHI per batch so adversary sees both cohorts every step.
    w_adv = np.where(C_adv.numpy() == 0, 1.0 / mu_fhs.shape[0], 1.0 / mu_whi.shape[0])
    sampler_adv = WeightedRandomSampler(torch.from_numpy(w_adv).double(),
                                        num_samples=max(args.batch_size * 8, len(C_adv) // 4),
                                        replacement=True)
    loader_adv = DataLoader(TensorDataset(Mu_adv, C_adv),
                            batch_size=args.batch_size, sampler=sampler_adv, num_workers=0, drop_last=False)
    log.info("Adversary loader: n_FHS=%d n_WHI=%d (balanced sampling)",
             mu_fhs.shape[0], mu_whi.shape[0])

    # --- model / optim ---
    enc_h = tuple(int(x) for x in args.enc_hidden.split(",") if x.strip())
    dec_h = tuple(int(x) for x in args.dec_hidden.split(",") if x.strip())
    model = AESurvHead(
        in_dim=latent_dim, enc_hidden=enc_h, dec_hidden=dec_h,
        z_dim=args.z_dim, cohort_hidden=args.cohort_hidden, dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("AESURV head: in=%d enc=%s dec=%s z=%d cohort_hidden=%d dropout=%.2f params=%d",
             latent_dim, enc_h, dec_h, args.z_dim, args.cohort_hidden, args.dropout, n_params)

    # --- training loop ---
    metrics_path = out_dir / "aesurv_dann_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_c_va = -1.0; best_epoch = -1; best_state: Optional[dict] = None; bad_streak = 0

    adv_iter = iter(loader_adv)
    for epoch in range(args.epochs):
        # warmups
        beta_kl = args.beta_kl * min(1.0, (epoch + 1) / max(1, args.beta_kl_warmup_epochs))
        lam_adv = args.lam_adv_max * min(1.0, (epoch + 1) / max(1, args.lam_adv_warmup_epochs))

        model.train()
        sum_total = 0.0; sum_cox = 0.0; sum_rec = 0.0; sum_kl = 0.0; sum_adv = 0.0
        adv_correct = 0; adv_seen = 0
        n_seen = 0
        for x, t, e in loader_tr:
            if int(e.sum().item()) == 0:
                continue
            x = x.to(device); t = t.to(device); e = e.to(device)
            try:
                xa, ca = next(adv_iter)
            except StopIteration:
                adv_iter = iter(loader_adv); xa, ca = next(adv_iter)
            xa = xa.to(device); ca = ca.to(device)

            opt.zero_grad(set_to_none=True)
            x_rec, z, log_h, mu_h, logvar_h = model(x, sample_z=True)
            cox = cox_ph_loss(log_h, t, e)
            rec = F.smooth_l1_loss(x_rec, x)
            kl = (-0.5 * (1.0 + logvar_h - mu_h.pow(2) - logvar_h.exp())).sum(dim=1).mean()

            # adversary uses its own forward to get z (no Cox/recon path here)
            _, z_adv, _, mu_a, logvar_a = model(xa, sample_z=True)
            adv_logits = model.cohort_logits(z_adv, lam=lam_adv)
            adv = F.cross_entropy(adv_logits, ca)

            loss = cox + args.alpha_recon * rec + beta_kl * kl + args.gamma_adv * adv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            bs = x.size(0); n_seen += bs
            sum_total += float(loss.item()) * bs
            sum_cox += float(cox.item()) * bs
            sum_rec += float(rec.item()) * bs
            sum_kl += float(kl.item()) * bs
            sum_adv += float(adv.item()) * bs
            with torch.no_grad():
                pred = adv_logits.argmax(dim=1)
                adv_correct += int((pred == ca).sum().item())
                adv_seen += int(ca.numel())

        denom = max(1, n_seen)
        tr_total = sum_total / denom; tr_cox = sum_cox / denom
        tr_rec = sum_rec / denom; tr_kl = sum_kl / denom; tr_adv = sum_adv / denom
        adv_acc = adv_correct / max(1, adv_seen)

        # eval (deterministic forward: z = mu_h)
        model.eval()
        with torch.no_grad():
            r_tr = model(Mu_tr.to(device), sample_z=False)[2].cpu().numpy()
            r_va = model(Mu_va.to(device), sample_z=False)[2].cpu().numpy()
            r_te = model(Mu_te.to(device), sample_z=False)[2].cpu().numpy()
        c_tr = harrell_c_index(T_tr.numpy(), E_tr.numpy().astype(np.int32), r_tr)
        c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), r_va)
        c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), r_te)

        rec_d = {
            "epoch": epoch + 1,
            "lam_adv": lam_adv, "beta_kl": beta_kl,
            "train_total": tr_total, "train_cox": tr_cox,
            "train_recon": tr_rec, "train_kl": tr_kl, "train_adv": tr_adv,
            "train_adv_acc": adv_acc,
            "train_cindex": c_tr, "val_cindex": c_va, "test_whi_cindex": c_te,
        }

        def _clean(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: _clean(v) for k, v in rec_d.items()}) + "\n")

        log.info("ep %03d/%d  lam=%.2f bkl=%.3f  cox=%.4f rec=%.4f kl=%.4f adv=%.4f advAcc=%.3f  c_tr=%.4f c_va=%.4f c_whi=%.4f",
                 epoch + 1, args.epochs, lam_adv, beta_kl, tr_cox, tr_rec, tr_kl, tr_adv,
                 adv_acc, c_tr, c_va, c_te)

        improved = (not math.isnan(c_va)) and (c_va > best_c_va + 1e-4)
        if improved:
            best_c_va = c_va; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_streak = 0
        else:
            bad_streak += 1
            if epoch + 1 >= args.min_epochs and bad_streak >= args.patience:
                log.info("Early stop at epoch %d (no val improvement for %d epochs).",
                         epoch + 1, bad_streak)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        r_tr = model(Mu_tr.to(device), sample_z=False)[2].cpu().numpy()
        r_va = model(Mu_va.to(device), sample_z=False)[2].cpu().numpy()
        r_te = model(Mu_te.to(device), sample_z=False)[2].cpu().numpy()
    c_tr = harrell_c_index(T_tr.numpy(), E_tr.numpy().astype(np.int32), r_tr)
    c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), r_va)
    c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), r_te)

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "in_dim": latent_dim, "enc_hidden": list(enc_h), "dec_hidden": list(dec_h),
            "z_dim": int(args.z_dim), "cohort_hidden": int(args.cohort_hidden),
            "dropout": float(args.dropout),
        },
        "best_epoch": int(best_epoch), "best_val_cindex": float(best_c_va),
    }, out_dir / "aesurv_dann_model.pt")
    log.info("BEST  c_tr=%.4f  c_val_fhs=%.4f  c_whi=%.4f  (epoch %d)",
             c_tr, c_va, c_te, best_epoch)

    np.savez(
        out_dir / "aesurv_dann_risk.npz",
        risk_fhs_train=r_tr.astype(np.float32),
        risk_fhs_val=r_va.astype(np.float32),
        risk_whi_test=r_te.astype(np.float32),
        time_fhs_train=T_tr.numpy().astype(np.float32),
        time_fhs_val=T_va.numpy().astype(np.float32),
        time_whi_test=T_te.numpy().astype(np.float32),
        event_fhs_train=E_tr.numpy().astype(np.int32),
        event_fhs_val=E_va.numpy().astype(np.int32),
        event_whi_test=E_te.numpy().astype(np.int32),
    )

    meta: Dict[str, object] = {
        "device": str(device),
        "n_fhs_train": int(len(tr_idx)), "n_fhs_val": int(len(va_idx)),
        "n_whi_test": int(len(T_te)),
        "events_fhs_train": int(E_tr.sum().item()),
        "events_fhs_val": int(E_va.sum().item()),
        "events_whi_test": int(E_te.sum().item()),
        "dann_encoder_ckpt": str(ckpt_path),
        "dann_preprocess_npz": str(npz_path),
        "dann_latent_dim": int(latent_dim),
        "head_enc_hidden": list(enc_h), "head_dec_hidden": list(dec_h),
        "z_dim": int(args.z_dim), "cohort_hidden": int(args.cohort_hidden),
        "dropout": float(args.dropout),
        "alpha_recon": float(args.alpha_recon),
        "beta_kl": float(args.beta_kl),
        "beta_kl_warmup_epochs": int(args.beta_kl_warmup_epochs),
        "gamma_adv": float(args.gamma_adv),
        "lam_adv_max": float(args.lam_adv_max),
        "lam_adv_warmup_epochs": int(args.lam_adv_warmup_epochs),
        "lr": float(args.lr), "weight_decay": float(args.weight_decay),
        "best_epoch": int(best_epoch), "best_val_cindex": float(best_c_va),
        "final_train_cindex": float(c_tr),
        "final_val_cindex": float(c_va),
        "final_whi_cindex": float(c_te),
    }
    (out_dir / "aesurv_dann_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Wrote %s and %s", out_dir / "aesurv_dann_model.pt", out_dir / "aesurv_dann_run_meta.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
