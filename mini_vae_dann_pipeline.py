#!/usr/bin/env python3
"""
Mini pipeline: load FHS + WHI combined NPZs (same path contract as PCA / train_vae_cox_lite),
align common methylation + SNP columns, then train a **projected** VAE with a
domain-adversarial head (Gradient Reversal Layer, DANN-style).

Why projection
  A full decoder over ~1e6 SNPs + CpGs is not practical. This script pools a
  ``StandardScaler`` on [methylation | SNP] (optional M-values for meth), then applies a
  **fixed** Johnson–Lindenstrauss random matrix ``W ∈ R^{D×K}`` (``K = --proj-dim``).
  The VAE reconstructs in **R^K**; the domain classifier sees the latent ``z = μ`` behind a GRL.

Objective
  Minimize reconstruction + KL while **confusing** a cohort discriminator (FHS=0, WHI=1).
  Use **val_balanced_domain_acc** near **0.5** together with **val_domain_acc** *not* hugging
  the majority-FHS baseline: balanced 0.5 alone can mean a collapsed **always-FHS** predictor
  (raw acc ≈ share of FHS in val), which is **not** domain invariance.

Outputs (``--out-dir``)
  ``mini_dann_model.pt``  — state_dict + minimal config
  ``mini_dann_preprocess.npz`` — scaler mean/scale, ``W``, dims, seed, ``meth_as_mvalues``
  ``mini_dann_metrics.jsonl`` — per epoch: recon, domain accs, **val_cbal_auc** (balanced-val ROC)
  ``mini_dann_run_meta.json`` — counts, sampling, ``domain_pos_weight``, etc.
  Plots: ``python plot_mini_dann_metrics.py --run-dir <out-dir>``

Training defaults (anti-collapse): **balanced** train sampling, **BCE pos_weight** for WHI,
**cohort-balanced val ROC-AUC** for checkpoint + early-stop (not raw imbalanced val acc).

Downstream AESURV (frozen stem + trainable ``meth_from_dann``):
  ``train_aesurv_highperf.py --dann-encoder-ckpt .../mini_dann_model.pt \\
      --dann-preprocess-npz .../mini_dann_preprocess.npz`` (same NPZ column counts as pretrain).

Example (smoke test):
  python mini_vae_dann_pipeline.py --max-snp 2000 --max-cpg 500 --epochs 5 --proj-dim 256

Full-scale (high RAM; tune ``--row-chunk`` if needed):
  python mini_vae_dann_pipeline.py --proj-dim 4096 --epochs 80
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from aesurv_domain import GradientReversal, vae_kl_loss

from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    _resolve_data_path,
    load_bundle_with_cache,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_to_m(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    b = np.clip(beta.astype(np.float64, copy=False), eps, 1.0 - eps)
    return np.log2(b / (1.0 - b))


def concat_rows_chunk(
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    start: int,
    end: int,
    meth_as_mvalues: bool,
) -> np.ndarray:
    """Rows [start:end) of hstack(meth, snp), float32."""
    xm = X_meth[start:end]
    xs = X_snp[start:end]
    if meth_as_mvalues:
        xm = beta_to_m(np.asarray(xm, dtype=np.float64))
    else:
        xm = np.asarray(xm, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    return np.hstack([xm, xs]).astype(np.float32, copy=False)


def fit_scaler_on_chunks(
    scaler: StandardScaler,
    X_meth_list: List[np.ndarray],
    X_snp_list: List[np.ndarray],
    row_chunk: int,
    meth_as_mvalues: bool,
) -> None:
    for Xm, Xs in zip(X_meth_list, X_snp_list):
        n = Xm.shape[0]
        for i in range(0, n, row_chunk):
            block = concat_rows_chunk(Xm, Xs, i, min(i + row_chunk, n), meth_as_mvalues)
            scaler.partial_fit(block)


def transform_and_project(
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    scaler: StandardScaler,
    W: np.ndarray,
    row_chunk: int,
    meth_as_mvalues: bool,
) -> np.ndarray:
    """Return (n, K) float32 projected matrix."""
    n = X_meth.shape[0]
    k = W.shape[1]
    out = np.zeros((n, k), dtype=np.float32)
    for i in range(0, n, row_chunk):
        j = min(i + row_chunk, n)
        raw = concat_rows_chunk(X_meth, X_snp, i, j, meth_as_mvalues)
        z = scaler.transform(raw).astype(np.float32, copy=False)
        out[i:j] = z @ W
    return out


def make_random_projection(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((d, k)).astype(np.float32)
    w /= np.sqrt(float(d))
    return w


class MiniVAEDANN(nn.Module):
    """
    VAE in projected space + domain head on μ with GRL (training only).
    """

    def __init__(
        self,
        proj_dim: int,
        hidden: Tuple[int, ...],
        latent_dim: int,
        dropout: float,
        n_fhs_batches: int = 0,
    ) -> None:
        super().__init__()
        enc_layers: List[nn.Module] = []
        prev = proj_dim
        for h in hidden:
            enc_layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden):
            dec_layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        dec_layers.append(nn.Linear(prev, proj_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.grl = GradientReversal(0.0)
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.n_fhs_batches = int(max(0, n_fhs_batches))
        self.batch_head: Optional[nn.Module]
        if self.n_fhs_batches > 1:
            self.batch_head = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout),
                nn.Linear(128, self.n_fhs_batches),
            )
        else:
            self.batch_head = None

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = torch.clamp(self.fc_mu(h), -6.0, 6.0)
        logvar = torch.clamp(self.fc_logvar(h), -6.0, 2.0)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mu, logvar = self.encode(x)
        z = mu
        recon = self.decode(z)
        dom_logit = self.domain_head(self.grl(z)).squeeze(-1)
        batch_logits = self.batch_head(self.grl(z)) if self.batch_head is not None else None
        return recon, mu, logvar, dom_logit, batch_logits

    @torch.no_grad()
    def domain_logits_clean(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminator logits without GRL (for monitoring accuracy)."""
        mu, _ = self.encode(x)
        return self.domain_head(mu).squeeze(-1)


@torch.no_grad()
def domain_accuracy(model: MiniVAEDANN, x: torch.Tensor, y: torch.Tensor) -> float:
    logit = model.domain_logits_clean(x)
    pred = (torch.sigmoid(logit) >= 0.5).float()
    return (pred == y).float().mean().item()


def _accumulate_domain_confusion(
    model: MiniVAEDANN,
    xb: torch.Tensor,
    yb: torch.Tensor,
    n0: int,
    n1: int,
    c0: int,
    c1: int,
) -> Tuple[int, int, int, int]:
    """Update counts for balanced accuracy: class 0=FHS, 1=WHI."""
    logit = model.domain_logits_clean(xb)
    pred = (torch.sigmoid(logit) >= 0.5).float()
    m0 = yb == 0
    m1 = yb == 1
    n0 += int(m0.sum().item())
    n1 += int(m1.sum().item())
    c0 += int(((pred == yb) & m0).sum().item())
    c1 += int(((pred == yb) & m1).sum().item())
    return n0, n1, c0, c1


def read_parquet_fast(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    try:
        df = pd.read_parquet(path, engine="fastparquet")
    except Exception:
        df = pd.read_parquet(path, engine="pyarrow")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fhs_batch_ids(meta_parquet: Path, batch_col: str, n_rows_expected: int) -> Tuple[np.ndarray, List[str]]:
    df = read_parquet_fast(meta_parquet)
    if batch_col not in df.columns:
        raise ValueError(
            f"--batch-label-col {batch_col!r} not in parquet columns. First 40 cols: {list(df.columns)[:40]}"
        )
    if len(df) != n_rows_expected:
        raise ValueError(
            f"FHS meta rows ({len(df)}) != loaded FHS feature rows ({n_rows_expected})."
        )
    raw = df[batch_col].fillna("__missing_batch__").astype(str).to_numpy()
    levels = sorted(pd.unique(pd.Series(raw)))
    idx = {k: i for i, k in enumerate(levels)}
    ids = np.array([idx[x] for x in raw], dtype=np.int64)
    return ids, levels


def dann_lambda(epoch: int, warmup_epochs: int, max_lambda: float) -> float:
    if warmup_epochs <= 0:
        return max_lambda
    t = min(1.0, (epoch + 1) / float(warmup_epochs))
    return max_lambda * t


def _median_distance(a: torch.Tensor, b: torch.Tensor, max_n: int = 256) -> float:
    """Median pairwise L2 distance between subsamples of two tensors (numerically stable scale).

    Returns a positive scalar; falls back to 1.0 if undefined or tiny.
    """
    with torch.no_grad():
        if a.shape[0] > max_n:
            idx_a = torch.randperm(a.shape[0], device=a.device)[:max_n]
            a = a[idx_a]
        if b.shape[0] > max_n:
            idx_b = torch.randperm(b.shape[0], device=b.device)[:max_n]
            b = b[idx_b]
        z = torch.cat([a, b], dim=0)
        d = torch.cdist(z, z, p=2)
        mask = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
        vals = d[mask]
        if vals.numel() == 0:
            return 1.0
        m = float(vals.median().item())
        return max(m, 1e-3)


def _gauss_mmd2(
    a: torch.Tensor,
    b: torch.Tensor,
    bandwidths: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
    use_median: bool = True,
    median_scales: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Multi-bandwidth Gaussian MMD^2 between two latent sample sets.

    a: [n_a, d], b: [n_b, d]. Returns a scalar tensor (>=0).
    Uses biased estimator (fine as a regularizer).

    If ``use_median`` is True, bandwidths are computed as ``median_scales * sigma_med``
    where ``sigma_med`` is the median pairwise L2 distance over a subsample of the union.
    This is crucial when the two clouds are far apart: fixed-bandwidth kernels saturate
    to zero on cross-terms and the MMD gradient vanishes.
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(0.0, device=a.device, dtype=a.dtype)
    if use_median:
        sigma_med = _median_distance(a, b)
        eff_bandwidths = tuple(float(s) * sigma_med for s in median_scales)
    else:
        eff_bandwidths = bandwidths

    aa = (a * a).sum(dim=1, keepdim=True)
    bb = (b * b).sum(dim=1, keepdim=True)
    d_aa = aa + aa.t() - 2.0 * (a @ a.t())
    d_bb = bb + bb.t() - 2.0 * (b @ b.t())
    d_ab = aa + bb.t() - 2.0 * (a @ b.t())
    d_aa = d_aa.clamp_min(0.0)
    d_bb = d_bb.clamp_min(0.0)
    d_ab = d_ab.clamp_min(0.0)

    mmd2 = a.new_zeros(())
    for s in eff_bandwidths:
        gamma = 1.0 / (2.0 * float(s) * float(s))
        k_aa = torch.exp(-gamma * d_aa)
        k_bb = torch.exp(-gamma * d_bb)
        k_ab = torch.exp(-gamma * d_ab)
        mmd2 = mmd2 + k_aa.mean() + k_bb.mean() - 2.0 * k_ab.mean()
    return mmd2


def cohort_mmd(
    mu: torch.Tensor,
    y_cohort: torch.Tensor,
    bandwidths: Tuple[float, ...],
    use_median: bool = True,
) -> torch.Tensor:
    """MMD between FHS (y=0) and WHI (y=1) latents in this minibatch."""
    m_f = y_cohort == 0
    m_w = y_cohort == 1
    if int(m_f.sum().item()) < 2 or int(m_w.sum().item()) < 2:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
    return _gauss_mmd2(mu[m_f], mu[m_w], bandwidths, use_median=use_median)


def fhs_batch_mmd(
    mu: torch.Tensor,
    y_cohort: torch.Tensor,
    b_fhs: torch.Tensor,
    n_classes: int,
    bandwidths: Tuple[float, ...],
    use_median: bool = True,
) -> torch.Tensor:
    """Sum of one-vs-rest MMD across FHS batches (within FHS only)."""
    m_fhs = (y_cohort == 0) & (b_fhs >= 0)
    if int(m_fhs.sum().item()) < 4 or n_classes <= 1:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
    mu_f = mu[m_fhs]
    b_f = b_fhs[m_fhs]
    total = mu.new_zeros(())
    n_used = 0
    for c in range(n_classes):
        m_in = b_f == c
        m_out = b_f != c
        if int(m_in.sum().item()) < 2 or int(m_out.sum().item()) < 2:
            continue
        total = total + _gauss_mmd2(
            mu_f[m_in], mu_f[m_out], bandwidths, use_median=use_median
        )
        n_used += 1
    if n_used == 0:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
    return total / float(n_used)


@torch.no_grad()
def domain_roc_auc_from_loader(
    model: MiniVAEDANN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """WHI = positive label (1). Returns NaN if undefined."""
    ys: List[float] = []
    ps: List[float] = []
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logit = model.domain_logits_clean(xb)
        pr = torch.sigmoid(logit)
        ys.extend(yb.detach().float().cpu().numpy().tolist())
        ps.extend(pr.detach().float().cpu().numpy().tolist())
    if len(set(int(round(y)) for y in ys)) < 2:
        return float("nan")
    return float(roc_auc_score(np.asarray(ys, dtype=np.int32), np.asarray(ps, dtype=np.float64)))


def main() -> None:
    p = argparse.ArgumentParser(description="Mini VAE + DANN cohort-invariance pipeline.")
    p.add_argument("--combined-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--meta-parquet", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--test-combined-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--test-meta-parquet", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--snp-columns-txt", type=str, default=None)
    p.add_argument("--max-cpg", type=int, default=None)
    p.add_argument("--max-snp", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default="vae_cox_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--meth-beta-values", action="store_true", help="Use beta instead of M-value for methylation.")
    p.add_argument("--proj-dim", type=int, default=2048)
    p.add_argument("--hidden", type=str, default="1024,512")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--lambda-domain-max",
        type=float,
        default=2.0,
        help="Max GRL strength (lower e.g. 1.0–1.5 if loss oscillates or val_auc sticks at 1.0).",
    )
    p.add_argument("--lambda-warmup-epochs", type=int, default=25)
    p.add_argument("--kl-weight", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.05, help="Domain BCE label smoothing.")
    p.add_argument(
        "--recon-weight",
        type=float,
        default=0.3,
        help="Weight on reconstruction MSE. Lower this if the latent keeps cohort info "
        "(typical: 0.2-0.5).",
    )
    p.add_argument(
        "--mmd-weight",
        type=float,
        default=1.0,
        help="Weight for cohort MMD (FHS vs WHI) on latent mu. 0 to disable.",
    )
    p.add_argument(
        "--mmd-batch-weight",
        type=float,
        default=0.5,
        help="Weight for FHS-batch MMD (one-vs-rest) on latent mu. 0 to disable.",
    )
    p.add_argument(
        "--mmd-bandwidths",
        type=str,
        default="1,2,4,8,16",
        help="Fixed Gaussian kernel bandwidths for MMD (used only if --mmd-no-median is set).",
    )
    p.add_argument(
        "--mmd-no-median",
        action="store_true",
        help="Disable median-heuristic bandwidth (use fixed --mmd-bandwidths). Not recommended.",
    )
    p.add_argument(
        "--batch-adv-weight",
        type=float,
        default=0.7,
        help="Weight for FHS batch adversary cross-entropy (set 0 to disable batch adversary).",
    )
    p.add_argument(
        "--batch-label-col",
        type=str,
        default="batch",
        help="FHS meta parquet column for batch labels (e.g., Gen3/JHU/UMN).",
    )
    p.add_argument(
        "--domain-loss-scale",
        type=float,
        default=1.0,
        help="Multiply domain BCE before adding to total loss (try 0.5–0.8 if encoder ignores recon).",
    )
    p.add_argument(
        "--domain-pos-weight",
        type=float,
        default=None,
        help="BCE pos_weight for WHI=1 (default: min(n_fhs_train/n_whi_train, --domain-pos-weight-cap)).",
    )
    p.add_argument(
        "--domain-pos-weight-cap",
        type=float,
        default=40.0,
        help="Cap auto pos_weight so huge FHS:WHI imbalance does not explode domain gradients.",
    )
    p.add_argument(
        "--train-sampling",
        type=str,
        choices=("balanced", "shuffle"),
        default="balanced",
        help="balanced = cohort-balanced draws (recommended); shuffle = classic shuffle (minority often missing per batch).",
    )
    p.add_argument(
        "--val-cbal-min-per-cohort",
        type=int,
        default=32,
        help="Per-cohort val samples for ROC-AUC pool (min of FHS val, WHI val, this cap). Set 0 to disable AUC / cbal early-stop.",
    )
    p.add_argument(
        "--auc-early-stop-tol",
        type=float,
        default=0.06,
        help="Early-stop when |val_cbal_auc - 0.5| <= this for --patience epochs (requires val-cbal-min-per-cohort > 0).",
    )
    p.add_argument("--fhs-val-frac", type=float, default=0.12)
    p.add_argument("--whi-val-frac", type=float, default=0.1)
    p.add_argument("--row-chunk", type=int, default=128, help="Rows per chunk for scaler / projection I/O.")
    p.add_argument("--proj-seed", type=int, default=42, help="RNG seed for W (independent of --seed).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="runs/mini_vae_dann")
    p.add_argument(
        "--domain-chance-margin",
        type=float,
        default=0.03,
        help="Early-stop when val |domain_acc-0.5| <= this for --patience consecutive epochs.",
    )
    p.add_argument("--patience", type=int, default=8)
    p.add_argument(
        "--min-epochs",
        type=int,
        default=5,
        help="Minimum epochs before chance-domain early stop can trigger.",
    )
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = _resolve_data_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = _resolve_data_path(args.cache_dir)
    fhs_npz = _resolve_data_path(args.combined_npz)
    fhs_pq = _resolve_data_path(args.meta_parquet)
    whi_npz = _resolve_data_path(args.test_combined_npz)
    whi_pq = _resolve_data_path(args.test_meta_parquet)
    snp_txt = _resolve_data_path(args.snp_columns_txt) if args.snp_columns_txt else None

    print("Loading FHS bundle...", flush=True)
    # Labels must match train_aesurv_highperf / train_vae_cox_lite so ``vae_cox_cache/bundles/*.npz`` is shared.
    X_meth_fhs, X_snp_fhs, _t_fhs, _e_fhs, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        label="FHS",
        combined_npz=fhs_npz,
        meta_parquet=fhs_pq,
        snp_columns_txt=snp_txt,
        max_cpg=args.max_cpg,
        max_snp=args.max_snp,
        cache_root=cache_root,
        use_cache=not args.no_cache,
    )
    print("Loading WHI bundle...", flush=True)
    X_meth_whi, X_snp_whi, _t_whi, _e_whi, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
        label="WHI_raw",
        combined_npz=whi_npz,
        meta_parquet=whi_pq,
        snp_columns_txt=None,
        max_cpg=args.max_cpg,
        max_snp=args.max_snp,
        cache_root=cache_root,
        use_cache=not args.no_cache,
    )

    meth_fhs_names, snp_fhs_names = _get_truncated_feature_names(fhs_npz, snp_txt, n_cpg_fhs, n_snp_fhs)
    meth_whi_names, snp_whi_names = _get_truncated_feature_names(whi_npz, None, n_cpg_whi, n_snp_whi)

    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, n_cpg, n_snp = _align_common_features(
        X_meth_fhs,
        X_snp_fhs,
        meth_fhs_names,
        snp_fhs_names,
        X_meth_whi,
        X_snp_whi,
        meth_whi_names,
        snp_whi_names,
    )
    d_in = int(n_cpg + n_snp)
    print(f"Aligned: n_cpg={n_cpg} n_snp={n_snp} D={d_in} | FHS {X_meth_fhs.shape[0]} WHI {X_meth_whi.shape[0]}", flush=True)
    if d_in == 0:
        raise SystemExit("No features after alignment.")

    meth_mval = not args.meth_beta_values

    # --- train/val index splits (FHS and WHI separately) ---
    n_fhs = X_meth_fhs.shape[0]
    n_whi = X_meth_whi.shape[0]
    idx_fhs = np.arange(n_fhs)
    idx_whi = np.arange(n_whi)
    fhs_tr, fhs_va = train_test_split(
        idx_fhs, test_size=args.fhs_val_frac, random_state=args.seed, shuffle=True
    )
    whi_tr, whi_va = train_test_split(
        idx_whi, test_size=args.whi_val_frac, random_state=args.seed, shuffle=True
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    print("Fitting StandardScaler (pooled FHS-train + WHI-train chunks)...", flush=True)
    t0 = time.perf_counter()
    fit_scaler_on_chunks(
        scaler,
        [X_meth_fhs[fhs_tr], X_meth_whi[whi_tr]],
        [X_snp_fhs[fhs_tr], X_snp_whi[whi_tr]],
        row_chunk=args.row_chunk,
        meth_as_mvalues=meth_mval,
    )
    print(f"  scaler fit in {time.perf_counter() - t0:.1f}s", flush=True)

    W = make_random_projection(d_in, args.proj_dim, args.proj_seed)
    print(f"Projecting to K={args.proj_dim} (JL random matrix)...", flush=True)
    t1 = time.perf_counter()
    P_fhs = transform_and_project(X_meth_fhs, X_snp_fhs, scaler, W, args.row_chunk, meth_mval)
    P_whi = transform_and_project(X_meth_whi, X_snp_whi, scaler, W, args.row_chunk, meth_mval)
    print(f"  projection done in {time.perf_counter() - t1:.1f}s", flush=True)

    P_fhs_tr, P_fhs_va = P_fhs[fhs_tr], P_fhs[fhs_va]
    P_whi_tr, P_whi_va = P_whi[whi_tr], P_whi[whi_va]

    fhs_batch_all, fhs_batch_levels = fhs_batch_ids(fhs_pq, args.batch_label_col, X_meth_fhs.shape[0])
    b_fhs_tr = fhs_batch_all[fhs_tr]
    b_fhs_va = fhs_batch_all[fhs_va]

    x_tr = np.vstack([P_fhs_tr, P_whi_tr])
    y_tr = np.concatenate([np.zeros(len(fhs_tr)), np.ones(len(whi_tr))]).astype(np.float32)
    b_tr = np.concatenate([b_fhs_tr, -1 * np.ones(len(whi_tr), dtype=np.int64)]).astype(np.int64)
    x_va = np.vstack([P_fhs_va, P_whi_va])
    y_va = np.concatenate([np.zeros(len(fhs_va)), np.ones(len(whi_va))]).astype(np.float32)
    b_va = np.concatenate([b_fhs_va, -1 * np.ones(len(whi_va), dtype=np.int64)]).astype(np.int64)

    n_fhs_tr, n_whi_tr = len(fhs_tr), len(whi_tr)
    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr), torch.from_numpy(b_tr))
    if args.train_sampling == "balanced":
        w = np.zeros(len(y_tr), dtype=np.float64)
        w[:n_fhs_tr] = 1.0 / max(1, n_fhs_tr)
        w[n_fhs_tr:] = 1.0 / max(1, n_whi_tr)
        sampler_tr = WeightedRandomSampler(
            torch.from_numpy(w).double(),
            num_samples=len(y_tr),
            replacement=True,
        )
        loader_tr = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            sampler=sampler_tr,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        print(
            f"Train sampling: balanced (WeightedRandomSampler) | n_fhs_tr={n_fhs_tr} n_whi_tr={n_whi_tr}",
            flush=True,
        )
    else:
        loader_tr = DataLoader(
            ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0
        )
        print("Train sampling: shuffle (legacy; minority cohort often under-represented per batch).", flush=True)

    ds_va = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va), torch.from_numpy(b_va))
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    loader_va_cbal: Optional[DataLoader] = None
    k_cbal = 0
    if int(args.val_cbal_min_per_cohort) > 0:
        k_cbal = int(min(len(fhs_va), len(whi_va), int(args.val_cbal_min_per_cohort)))
        if k_cbal >= 8:
            rng_c = np.random.default_rng(int(args.seed) + 424242)
            sub_f = rng_c.choice(np.arange(len(fhs_va)), size=k_cbal, replace=False)
            sub_w = rng_c.choice(np.arange(len(whi_va)), size=k_cbal, replace=False)
            fi = fhs_va[sub_f]
            wi = whi_va[sub_w]
            x_cbal = np.vstack([P_fhs[fi], P_whi[wi]])
            y_cbal = np.concatenate([np.zeros(k_cbal), np.ones(k_cbal)]).astype(np.float32)
            ds_cbal = TensorDataset(torch.from_numpy(x_cbal), torch.from_numpy(y_cbal))
            loader_va_cbal = DataLoader(
                ds_cbal, batch_size=min(256, 2 * k_cbal), shuffle=False, num_workers=0
            )
            print(
                f"Val ROC pool: {k_cbal} FHS + {k_cbal} WHI (fixed seed) for val_cbal_auc / early-stop.",
                flush=True,
            )

    n_fhs_va, n_whi_va = len(fhs_va), len(whi_va)
    maj_val = n_fhs_va / max(1, (n_fhs_va + n_whi_va))
    if args.domain_pos_weight is not None:
        pw_used = float(args.domain_pos_weight)
    elif args.train_sampling == "balanced":
        # Sampler already presents ~50/50 batches; an imbalance-derived pos_weight
        # collapses the head to predicting the minority class and kills the adversary.
        pw_used = 1.0
    else:
        pw_used = float(min(n_fhs_tr / max(1, n_whi_tr), float(args.domain_pos_weight_cap)))
    domain_pw = torch.tensor([pw_used], dtype=torch.float32, device=device)

    run_meta = {
        "n_fhs_train": int(n_fhs_tr),
        "n_whi_train": int(n_whi_tr),
        "n_fhs_val": int(n_fhs_va),
        "n_whi_val": int(n_whi_va),
        "val_majority_fhs_baseline_acc": float(maj_val),
        "train_sampling": str(args.train_sampling),
        "domain_pos_weight": float(pw_used),
        "domain_loss_scale": float(args.domain_loss_scale),
        "batch_adv_weight": float(args.batch_adv_weight),
        "batch_levels": [str(x) for x in fhs_batch_levels],
        "recon_weight": float(args.recon_weight),
        "mmd_weight": float(args.mmd_weight),
        "mmd_batch_weight": float(args.mmd_batch_weight),
        "mmd_bandwidths": str(args.mmd_bandwidths),
        "val_cbal_k_per_cohort": int(k_cbal),
        "note": "Prefer val_cbal_auc near 0.5 AND small val_mmd_cohort + val_mmd_batch.",
    }
    (out_dir / "mini_dann_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(
        f"Val split: FHS={n_fhs_va} WHI={n_whi_va} - majority-class baseline acc if always-FHS: {maj_val:.3f}",
        flush=True,
    )
    print(
        f"Domain BCE pos_weight (WHI positive)={pw_used:.2f}  domain_loss_scale={args.domain_loss_scale}",
        flush=True,
    )
    print(
        f"FHS batch adversary: levels={len(fhs_batch_levels)} weight={args.batch_adv_weight:.3f} "
        f"column={args.batch_label_col}",
        flush=True,
    )

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    model = MiniVAEDANN(
        proj_dim=args.proj_dim,
        hidden=hidden,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        n_fhs_batches=len(fhs_batch_levels) if args.batch_adv_weight > 0 else 0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_path = out_dir / "mini_dann_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    best_margin = float("inf")
    best_state: Optional[dict] = None
    near_chance_run = 0

    mmd_bandwidths = tuple(float(x) for x in str(args.mmd_bandwidths).split(",") if x.strip())
    if len(mmd_bandwidths) == 0:
        mmd_bandwidths = (1.0, 2.0, 4.0, 8.0, 16.0)
    use_median = not bool(args.mmd_no_median)
    print(
        f"MMD: cohort_w={args.mmd_weight:.3f} batch_w={args.mmd_batch_weight:.3f} "
        f"median_heuristic={use_median} fixed_bandwidths={list(mmd_bandwidths)}",
        flush=True,
    )
    print(
        f"Recon weight={args.recon_weight:.3f}  (lower this if latent still encodes cohort)",
        flush=True,
    )

    for epoch in range(args.epochs):
        lam = dann_lambda(epoch, args.lambda_warmup_epochs, args.lambda_domain_max)
        model.grl.set_lambda(lam)
        model.train()
        sum_recon = sum_kl = sum_dom = sum_batch = sum_mmd_c = sum_mmd_b = 0.0
        n_seen = 0
        dom_correct_tr = 0.0
        n_dom_tr = 0
        tr_n0 = tr_n1 = tr_c0 = tr_c1 = 0
        tr_bn = tr_bc = 0

        for xb, yb, bb in loader_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            bb = bb.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar, dom_logit, batch_logits = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            kl = vae_kl_loss(mu, logvar)
            if args.label_smoothing > 0:
                y_smooth = yb * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            else:
                y_smooth = yb
            dom_loss = F.binary_cross_entropy_with_logits(
                dom_logit, y_smooth, pos_weight=domain_pw.to(device)
            )
            batch_loss = torch.tensor(0.0, device=device)
            if args.batch_adv_weight > 0 and batch_logits is not None:
                m_fhs = bb >= 0
                if torch.any(m_fhs):
                    batch_loss = F.cross_entropy(batch_logits[m_fhs], bb[m_fhs].long())

            mmd_c = torch.tensor(0.0, device=device)
            if args.mmd_weight > 0:
                mmd_c = cohort_mmd(mu, yb, mmd_bandwidths, use_median=use_median)
            mmd_b = torch.tensor(0.0, device=device)
            if args.mmd_batch_weight > 0 and model.batch_head is not None:
                mmd_b = fhs_batch_mmd(
                    mu, yb, bb, model.n_fhs_batches, mmd_bandwidths, use_median=use_median
                )

            loss = (
                float(args.recon_weight) * recon_loss
                + args.kl_weight * kl
                + args.domain_loss_scale * dom_loss
                + float(args.batch_adv_weight) * batch_loss
                + float(args.mmd_weight) * mmd_c
                + float(args.mmd_batch_weight) * mmd_b
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            bs = xb.size(0)
            sum_recon += recon_loss.item() * bs
            sum_kl += kl.item() * bs
            sum_dom += dom_loss.item() * bs
            sum_batch += batch_loss.item() * bs
            sum_mmd_c += mmd_c.item() * bs
            sum_mmd_b += mmd_b.item() * bs
            n_seen += bs
            with torch.no_grad():
                dom_correct_tr += domain_accuracy(model, xb, yb) * bs
                n_dom_tr += bs
                tr_n0, tr_n1, tr_c0, tr_c1 = _accumulate_domain_confusion(
                    model, xb, yb, tr_n0, tr_n1, tr_c0, tr_c1
                )
                if args.batch_adv_weight > 0 and batch_logits is not None:
                    m_fhs = bb >= 0
                    if torch.any(m_fhs):
                        pred_b = torch.argmax(batch_logits[m_fhs], dim=1)
                        tr_bc += int((pred_b == bb[m_fhs]).sum().item())
                        tr_bn += int(m_fhs.sum().item())

        model.eval()
        va_recon = va_kl = va_dom = va_batch = va_mmd_c = va_mmd_b = 0.0
        n_va = 0
        dom_correct_va = 0.0
        n_dom_va = 0
        va_n0 = va_n1 = va_c0 = va_c1 = 0
        va_bn = va_bc = 0
        va_batch_class_n = np.zeros(len(fhs_batch_levels), dtype=np.int64)
        va_batch_class_c = np.zeros(len(fhs_batch_levels), dtype=np.int64)
        with torch.no_grad():
            for xb, yb, bb in loader_va:
                xb = xb.to(device)
                yb = yb.to(device)
                bb = bb.to(device)
                recon, mu, logvar, dom_logit, batch_logits = model(xb)
                bs = xb.size(0)
                va_recon += F.mse_loss(recon, xb).item() * bs
                va_kl += vae_kl_loss(mu, logvar).item() * bs
                if args.label_smoothing > 0:
                    ys = yb * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                else:
                    ys = yb
                va_dom += (
                    F.binary_cross_entropy_with_logits(dom_logit, ys, pos_weight=domain_pw.to(device)).item()
                    * bs
                )
                batch_loss = torch.tensor(0.0, device=device)
                if args.batch_adv_weight > 0 and batch_logits is not None:
                    m_fhs = bb >= 0
                    if torch.any(m_fhs):
                        batch_loss = F.cross_entropy(batch_logits[m_fhs], bb[m_fhs].long())
                va_batch += batch_loss.item() * bs
                va_mmd_c += cohort_mmd(mu, yb, mmd_bandwidths, use_median=use_median).item() * bs
                if model.batch_head is not None:
                    va_mmd_b += fhs_batch_mmd(
                        mu, yb, bb, model.n_fhs_batches, mmd_bandwidths, use_median=use_median
                    ).item() * bs
                n_va += bs
                dom_correct_va += domain_accuracy(model, xb, yb) * bs
                n_dom_va += bs
                va_n0, va_n1, va_c0, va_c1 = _accumulate_domain_confusion(
                    model, xb, yb, va_n0, va_n1, va_c0, va_c1
                )
                if args.batch_adv_weight > 0 and batch_logits is not None:
                    m_fhs = bb >= 0
                    if torch.any(m_fhs):
                        pred_b = torch.argmax(batch_logits[m_fhs], dim=1)
                        bb_f = bb[m_fhs].long()
                        va_bc += int((pred_b == bb_f).sum().item())
                        va_bn += int(bb_f.numel())
                        for cls in range(len(fhs_batch_levels)):
                            m_cls = bb_f == cls
                            n_cls = int(m_cls.sum().item())
                            if n_cls > 0:
                                va_batch_class_n[cls] += n_cls
                                va_batch_class_c[cls] += int((pred_b[m_cls] == bb_f[m_cls]).sum().item())

        tr_recon_m = sum_recon / max(1, n_seen)
        tr_dom_acc = dom_correct_tr / max(1, n_dom_tr)
        va_dom_acc = dom_correct_va / max(1, n_dom_va)
        tr_bal = (
            0.5 * (tr_c0 / tr_n0 + tr_c1 / tr_n1) if tr_n0 > 0 and tr_n1 > 0 else float("nan")
        )
        va_bal = (
            0.5 * (va_c0 / va_n0 + va_c1 / va_n1) if va_n0 > 0 and va_n1 > 0 else float("nan")
        )
        tr_batch_acc = (tr_bc / tr_bn) if tr_bn > 0 else float("nan")
        va_batch_acc = (va_bc / va_bn) if va_bn > 0 else float("nan")
        per_cls_recalls = [
            (va_batch_class_c[i] / va_batch_class_n[i]) for i in range(len(fhs_batch_levels)) if va_batch_class_n[i] > 0
        ]
        va_batch_bal = float(np.mean(per_cls_recalls)) if len(per_cls_recalls) > 0 else float("nan")
        batch_chance = (1.0 / len(fhs_batch_levels)) if len(fhs_batch_levels) > 0 else float("nan")
        margin = abs(va_dom_acc - 0.5)
        margin_bal = abs(va_bal - 0.5) if not math.isnan(va_bal) else float("nan")

        val_cbal_auc = float("nan")
        if loader_va_cbal is not None:
            val_cbal_auc = domain_roc_auc_from_loader(model, loader_va_cbal, device)

        rec = {
            "epoch": epoch,
            "lambda_grl": lam,
            "train_recon_mse": tr_recon_m,
            "train_domain_acc": tr_dom_acc,
            "train_balanced_domain_acc": tr_bal,
            "train_batch_acc": tr_batch_acc,
            "train_mmd_cohort": sum_mmd_c / max(1, n_seen),
            "train_mmd_batch": sum_mmd_b / max(1, n_seen),
            "val_recon_mse": va_recon / max(1, n_va),
            "val_kl": va_kl / max(1, n_va),
            "val_domain_acc": va_dom_acc,
            "val_balanced_domain_acc": va_bal,
            "val_batch_acc": va_batch_acc,
            "val_batch_balanced_acc": va_batch_bal,
            "val_batch_chance_acc": batch_chance,
            "val_batch_loss": va_batch / max(1, n_va),
            "val_mmd_cohort": va_mmd_c / max(1, n_va),
            "val_mmd_batch": va_mmd_b / max(1, n_va),
            "val_abs_acc_minus_half": margin,
            "val_abs_balanced_minus_half": margin_bal,
            "val_cbal_auc": val_cbal_auc,
        }
        def _json_floats(obj: dict) -> dict:
            out = {}
            for k, v in obj.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out[k] = None
                else:
                    out[k] = v
            return out

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_floats(rec)) + "\n")

        auc_s = f"{val_cbal_auc:.3f}" if not math.isnan(val_cbal_auc) else "nan"
        print(
            f"epoch {epoch+1:03d}/{args.epochs}  lambda={lam:.3f}  "
            f"train_dom_acc={tr_dom_acc:.3f}  val_dom_acc={va_dom_acc:.3f}  "
            f"val_bal={va_bal:.3f}  val_auc={auc_s}  val_batch_bal={va_batch_bal:.3f}  "
            f"mmd_c={va_mmd_c/max(1,n_va):.4f}  mmd_b={va_mmd_b/max(1,n_va):.4f}  "
            f"val_recon={va_recon/max(1,n_va):.4f}",
            flush=True,
        )

        # Balanced acc = 0.5 with raw acc ≈ "always FHS" majority baseline means **collapsed**
        # constant classifier, not a confused domain head (same degeneracy as trivial baseline).
        collapse_majority = (
            not math.isnan(va_bal)
            and abs(va_bal - 0.5) < 0.02
            and abs(va_dom_acc - maj_val) < 0.03
        )

        # Best checkpoint: prefer cohort AUC -> 0.5, FHS batch balanced acc -> chance,
        # AND small MMD (cohort + batch). MMD is added on a clamped scale to keep weights stable.
        m_ckpt = margin_bal if not math.isnan(margin_bal) else margin
        batch_dev = (
            abs(va_batch_bal - batch_chance)
            if (not math.isnan(va_batch_bal) and not math.isnan(batch_chance))
            else 0.0
        )
        mmd_pen = 0.25 * float(min(1.0, va_mmd_c / max(1, n_va))) + 0.15 * float(
            min(1.0, va_mmd_b / max(1, n_va))
        )
        if not math.isnan(val_cbal_auc):
            dev_auc = abs(val_cbal_auc - 0.5) + 0.5 * batch_dev + mmd_pen
            if dev_auc < best_margin - 1e-4:
                best_margin = dev_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif (not collapse_majority) and (m_ckpt + 0.5 * batch_dev + mmd_pen) < best_margin - 1e-5:
            best_margin = m_ckpt + 0.5 * batch_dev + mmd_pen
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if not math.isnan(val_cbal_auc) and int(args.val_cbal_min_per_cohort) > 0:
            near_stop = abs(val_cbal_auc - 0.5)
            if near_stop <= args.auc_early_stop_tol:
                near_chance_run += 1
                print(
                    f"  -> val_cbal_auc near 0.5 (|auc-0.5|={near_stop:.3f} <= {args.auc_early_stop_tol}) "
                    f"[{near_chance_run}/{args.patience}]",
                    flush=True,
                )
                if epoch + 1 >= args.min_epochs and near_chance_run >= args.patience:
                    print("Early stop: domain ROC ~ chance on cohort-balanced val pool.", flush=True)
                    break
            else:
                near_chance_run = 0
        else:
            near_m = margin_bal if not math.isnan(margin_bal) else margin
            if near_m <= args.domain_chance_margin and not collapse_majority:
                near_chance_run += 1
                print(
                    f"  -> val near chance (|metric-0.5|={near_m:.3f} <= {args.domain_chance_margin}) "
                    f"[{near_chance_run}/{args.patience}]",
                    flush=True,
                )
                if epoch + 1 >= args.min_epochs and near_chance_run >= args.patience:
                    print("Early stop: domain signal near chance on val.", flush=True)
                    break
            else:
                near_chance_run = 0

        if collapse_majority and (epoch < 4 or (epoch + 1) % 25 == 0):
            print(
                "  (imb-val: val_bal~0.5 & val_dom~majority baseline -> likely always-FHS; see val_auc on balanced pool.)",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "proj_dim": args.proj_dim,
                "hidden": hidden,
                "latent_dim": args.latent_dim,
                "dropout": args.dropout,
                "d_input_raw": d_in,
                "n_cpg": n_cpg,
                "n_snp": n_snp,
                "meth_as_mvalues": meth_mval,
                "train_sampling": str(args.train_sampling),
                "domain_pos_weight": float(pw_used),
                "domain_loss_scale": float(args.domain_loss_scale),
                "batch_adv_weight": float(args.batch_adv_weight),
                "batch_label_col": str(args.batch_label_col),
                "fhs_batch_levels": [str(x) for x in fhs_batch_levels],
                "n_fhs_batches": int(len(fhs_batch_levels)),
                "recon_weight": float(args.recon_weight),
                "mmd_weight": float(args.mmd_weight),
                "mmd_batch_weight": float(args.mmd_batch_weight),
                "mmd_bandwidths": str(args.mmd_bandwidths),
                "val_cbal_min_per_cohort": int(args.val_cbal_min_per_cohort),
            },
        },
        out_dir / "mini_dann_model.pt",
    )
    np.savez(
        out_dir / "mini_dann_preprocess.npz",
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        W=W.astype(np.float32),
        d_in=np.int64(d_in),
        proj_dim=np.int64(args.proj_dim),
        proj_seed=np.int64(args.proj_seed),
        meth_as_mvalues=np.array([0 if args.meth_beta_values else 1], dtype=np.int8),
        fhs_train_idx=fhs_tr.astype(np.int64),
        fhs_val_idx=fhs_va.astype(np.int64),
        whi_train_idx=whi_tr.astype(np.int64),
        whi_val_idx=whi_va.astype(np.int64),
    )
    print(f"Wrote {out_dir / 'mini_dann_model.pt'} and preprocess bundle.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
