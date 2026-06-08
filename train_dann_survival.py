#!/usr/bin/env python3
"""
DANN-frozen-stem survival training (clean single-stage pipeline).

Purpose
-------
Train a Cox survival model on FHS using a **frozen** pretrained multi-adversary
DANN stem (cohort + FHS-batch invariant) to produce a domain-invariant 64-dim
latent, validated on a held-out FHS split and an external WHI test set.

The DANN stem (``mini_dann_model.pt`` + ``mini_dann_preprocess.npz``) is treated
as a deterministic feature extractor. We forward each sample **once** through:

    (X_meth, X_snp)  ->  scaler  ->  JL projection (W)  ->  encoder  ->  mu (64-d)

The 64-d ``mu`` is cached to disk. A small trainable Cox MLP head learns to
predict log-hazard from ``mu``. No two-stage pretraining, no adversarial
machinery, no cohort residual — the DANN stem already removes cohort/batch
effects from the latent.

Quick start
-----------
    python train_dann_survival.py ^
      --dann-encoder-ckpt   runs/mini_vae_dann_mmd_final/mini_dann_model.pt ^
      --dann-preprocess-npz runs/mini_vae_dann_mmd_final/mini_dann_preprocess.npz

Outputs (under --out-dir):
  - latent_cache_FHS.npz, latent_cache_WHI.npz
  - dann_surv_model.pt            (best-val checkpoint)
  - dann_surv_metrics.jsonl       (per-epoch metrics)
  - dann_surv_run_meta.json
  - dann_surv_predictions.csv     (FHS-val + WHI risk scores at best epoch)
"""
from __future__ import annotations

import argparse
import hashlib
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dann_aesurv_bridge import (
    DannLatentEncoder,
    InvariantPreprocessor,
    load_mini_dann_for_fusion,
)
from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    _resolve_data_path,
    load_bundle_with_cache,
)


# --------------------------------------------------------------------------- #
# C-index (Harrell)                                                            #
# --------------------------------------------------------------------------- #
def harrell_c_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    """Harrell's concordance index. Higher risk should mean shorter survival."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int32)
    r = np.asarray(risks, dtype=np.float64)
    n = t.shape[0]
    if n < 2:
        return float("nan")
    num = 0.0
    den = 0.0
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if t[j] > t[i]:
                den += 1.0
                if r[i] > r[j]:
                    num += 1.0
                elif r[i] == r[j]:
                    num += 0.5
    if den <= 0:
        return float("nan")
    return float(num / den)


# --------------------------------------------------------------------------- #
# Cox negative partial log-likelihood (Breslow)                                #
# --------------------------------------------------------------------------- #
def cox_ph_loss(log_h: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative log partial likelihood (Breslow). ``log_h`` is unconstrained log-hazard."""
    if event.sum() <= 0:
        return torch.tensor(0.0, device=log_h.device, dtype=log_h.dtype)
    order = torch.argsort(time, descending=True)
    log_h = log_h[order]
    event = event[order].float()
    # log cumulative hazard for risk sets (descending time => cumulative from start)
    log_cum = torch.logcumsumexp(log_h, dim=0)
    loss = -(log_h - log_cum) * event
    return loss.sum() / event.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Survival head                                                                #
# --------------------------------------------------------------------------- #
class CoxSurvivalHead(nn.Module):
    """MLP from DANN latent -> log-hazard scalar.

    The DANN stem already produces a domain-invariant representation, so we keep
    this head intentionally small to avoid overfitting on a few thousand FHS
    subjects.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# Latent extraction (cached)                                                   #
# --------------------------------------------------------------------------- #
def _latent_cache_key(
    bundle_label: str,
    n_rows: int,
    d_in: int,
    proj_dim: int,
    latent_dim: int,
    ckpt_path: Path,
    npz_path: Path,
    meth_as_mvalues: bool,
    extra_tag: str = "",
) -> str:
    h = hashlib.sha1()
    h.update(bundle_label.encode())
    h.update(f"|n={n_rows}|d={d_in}|p={proj_dim}|l={latent_dim}|m={int(meth_as_mvalues)}".encode())
    if extra_tag:
        h.update(f"|tag={extra_tag}".encode())
    try:
        h.update(str(ckpt_path.stat().st_mtime_ns).encode())
        h.update(str(npz_path.stat().st_mtime_ns).encode())
    except OSError:
        pass
    return h.hexdigest()[:10]


@torch.no_grad()
def extract_latent_mu(
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    preproc: InvariantPreprocessor,
    encoder: DannLatentEncoder,
    *,
    meth_as_mvalues: bool,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Forward each sample through scaler+JL+encoder, return [N, latent_dim] mu."""
    preproc.eval()
    encoder.eval()
    n = X_meth.shape[0]
    out: List[np.ndarray] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        m = torch.from_numpy(np.ascontiguousarray(X_meth[s:e])).to(device)
        v = torch.from_numpy(np.ascontiguousarray(X_snp[s:e])).to(device)
        z = preproc(m, v, use_m_values=meth_as_mvalues)
        mu = encoder(z)
        out.append(mu.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0)


def load_or_extract_latent(
    label: str,
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    preproc: InvariantPreprocessor,
    encoder: DannLatentEncoder,
    *,
    meth_as_mvalues: bool,
    latent_dim: int,
    device: torch.device,
    cache_dir: Path,
    ckpt_path: Path,
    npz_path: Path,
    use_cache: bool,
    log: logging.Logger,
    extra_tag: str = "",
) -> np.ndarray:
    key = _latent_cache_key(
        label,
        X_meth.shape[0],
        preproc.d_in,
        preproc.proj_dim,
        latent_dim,
        ckpt_path,
        npz_path,
        meth_as_mvalues,
        extra_tag=extra_tag,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag_s = f"_{extra_tag}" if extra_tag else ""
    cache_path = cache_dir / f"latent_{label}_{key}{tag_s}.npz"
    if use_cache and cache_path.is_file():
        t0 = time.perf_counter()
        z = np.load(cache_path, allow_pickle=False)
        mu = z["mu"]
        log.info("[%s] latent cache hit: %s (%.2fs) mu=%s", label, cache_path.name, time.perf_counter() - t0, mu.shape)
        return mu
    t0 = time.perf_counter()
    mu = extract_latent_mu(
        X_meth,
        X_snp,
        preproc,
        encoder,
        meth_as_mvalues=meth_as_mvalues,
        device=device,
    )
    log.info("[%s] extracted latent in %.1fs (mu=%s)", label, time.perf_counter() - t0, mu.shape)
    if use_cache:
        np.savez_compressed(cache_path, mu=mu.astype(np.float32))
        log.info("[%s] cached latent -> %s", label, cache_path)
    return mu


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def _setup_logger(log_file: Optional[Path]) -> logging.Logger:
    log = logging.getLogger("dann_surv")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    log.propagate = False
    return log


def _split_indices(
    n: int, val_frac: float, seed: int, stratify_event: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    strat = stratify_event if (stratify_event is not None and stratify_event.sum() >= 2) else None
    tr, va = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=strat, shuffle=True)
    return tr, va


def main() -> None:
    p = argparse.ArgumentParser(
        description="DANN-stem survival training (clean single-stage pipeline)."
    )
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
    p.add_argument("--dann-meth-beta-values", action="store_true",
                   help="Feed raw beta values to the DANN stem instead of M-values.")
    p.add_argument("--latent-cache-dir", type=str, default="vae_cox_cache/dann_latent")
    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--hidden", type=str, default="64,32", help="Comma-separated MLP widths.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--balance-events", action="store_true",
                   help="Use WeightedRandomSampler to balance events vs censors per minibatch.")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # output
    p.add_argument("--out-dir", type=str, default="runs/dann_survival")
    p.add_argument("--log-file", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    out_dir = _resolve_data_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _resolve_data_path(args.log_file) if args.log_file else (out_dir / "dann_surv.log")
    log = _setup_logger(log_path)
    log.info("device=%s out_dir=%s", device, out_dir)

    # --- load aligned FHS / WHI bundles (same as DANN pretrain) ---
    cache_root = _resolve_data_path(args.cache_dir)
    fhs_npz = _resolve_data_path(args.combined_npz)
    fhs_pq = _resolve_data_path(args.meta_parquet)
    whi_npz = _resolve_data_path(args.test_combined_npz)
    whi_pq = _resolve_data_path(args.test_meta_parquet)
    snp_txt = _resolve_data_path(args.snp_columns_txt) if args.snp_columns_txt else None

    log.info("Loading FHS bundle...")
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_fhs, n_snp_fhs = load_bundle_with_cache(
        label="FHS",
        combined_npz=fhs_npz, meta_parquet=fhs_pq,
        snp_columns_txt=snp_txt, max_cpg=args.max_cpg, max_snp=args.max_snp,
        cache_root=cache_root, use_cache=not args.no_cache,
    )
    log.info("Loading WHI bundle...")
    X_meth_whi, X_snp_whi, t_whi, e_whi, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
        label="WHI_raw",
        combined_npz=whi_npz, meta_parquet=whi_pq,
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

    # --- frozen DANN stem ---
    ckpt_path = _resolve_data_path(args.dann_encoder_ckpt)
    npz_path = _resolve_data_path(args.dann_preprocess_npz)
    preproc = InvariantPreprocessor(npz_path).to(device).eval()
    if d_in != preproc.d_in:
        raise SystemExit(
            f"d_in mismatch: aligned features={d_in} but DANN preprocess expects {preproc.d_in}. "
            "Use the same --max-cpg/--max-snp/--snp-columns-txt as mini_vae_dann_pipeline."
        )
    mini_model, cfg = load_mini_dann_for_fusion(ckpt_path, map_location=device)
    encoder = DannLatentEncoder(mini_model).to(device).eval()
    latent_dim = int(cfg["latent_dim"])
    meth_mval = bool(cfg.get("meth_as_mvalues", True))
    if args.dann_meth_beta_values:
        meth_mval = False
    log.info("DANN cfg: latent_dim=%d proj_dim=%d meth_as_mvalues=%s",
             latent_dim, int(cfg["proj_dim"]), meth_mval)

    # --- latent extraction (cached) ---
    latent_cache_dir = _resolve_data_path(args.latent_cache_dir)
    mu_fhs = load_or_extract_latent(
        "FHS", X_meth_fhs, X_snp_fhs, preproc, encoder,
        meth_as_mvalues=meth_mval, latent_dim=latent_dim, device=device,
        cache_dir=latent_cache_dir, ckpt_path=ckpt_path, npz_path=npz_path,
        use_cache=not args.no_cache, log=log,
    )
    mu_whi = load_or_extract_latent(
        "WHI", X_meth_whi, X_snp_whi, preproc, encoder,
        meth_as_mvalues=meth_mval, latent_dim=latent_dim, device=device,
        cache_dir=latent_cache_dir, ckpt_path=ckpt_path, npz_path=npz_path,
        use_cache=not args.no_cache, log=log,
    )

    # --- FHS train/val split (stratified on event) ---
    tr_idx, va_idx = _split_indices(
        mu_fhs.shape[0], args.val_frac, args.seed, stratify_event=e_fhs.astype(np.int32)
    )
    log.info("FHS train n=%d (events=%d), val n=%d (events=%d), WHI test n=%d (events=%d)",
             len(tr_idx), int(e_fhs[tr_idx].sum()),
             len(va_idx), int(e_fhs[va_idx].sum()),
             len(e_whi), int(e_whi.sum()))

    # --- torch dataloaders ---
    Mu_tr = torch.from_numpy(mu_fhs[tr_idx]).float()
    T_tr = torch.from_numpy(t_fhs[tr_idx]).float()
    E_tr = torch.from_numpy(e_fhs[tr_idx]).float()
    Mu_va = torch.from_numpy(mu_fhs[va_idx]).float()
    T_va = torch.from_numpy(t_fhs[va_idx]).float()
    E_va = torch.from_numpy(e_fhs[va_idx]).float()
    Mu_te = torch.from_numpy(mu_whi).float()
    T_te = torch.from_numpy(t_whi).float()
    E_te = torch.from_numpy(e_whi).float()

    ds_tr = TensorDataset(Mu_tr, T_tr, E_tr)
    if args.balance_events:
        # one event class is usually <20%; sample events more often to fill the
        # risk set per minibatch (Cox loss needs both events and censors).
        w = np.where(E_tr.numpy() > 0.5, 1.0 / max(1, int(E_tr.sum().item())),
                     1.0 / max(1, len(E_tr) - int(E_tr.sum().item())))
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(w), replacement=True)
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                               drop_last=False, num_workers=0)
        log.info("Train sampling: balanced events (WeightedRandomSampler)")
    else:
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                               drop_last=False, num_workers=0)
        log.info("Train sampling: shuffle")

    # --- model + optimizer ---
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    model = CoxSurvivalHead(latent_dim=latent_dim, hidden_dims=hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log.info("Survival head: latent_dim=%d hidden=%s dropout=%.2f params=%d",
             latent_dim, hidden, args.dropout,
             sum(p.numel() for p in model.parameters() if p.requires_grad))

    metrics_path = out_dir / "dann_surv_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_c_va = -1.0
    best_epoch = -1
    best_state: Optional[dict] = None
    bad_streak = 0

    for epoch in range(args.epochs):
        # train
        model.train()
        sum_loss = 0.0
        n_seen = 0
        for mu, t, e in loader_tr:
            mu = mu.to(device); t = t.to(device); e = e.to(device)
            if int(e.sum().item()) == 0:
                # Cox loss undefined without events; skip this minibatch
                continue
            opt.zero_grad(set_to_none=True)
            log_h = model(mu)
            loss = cox_ph_loss(log_h, t, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            sum_loss += float(loss.item()) * mu.size(0)
            n_seen += mu.size(0)
        tr_loss = sum_loss / max(1, n_seen)

        # eval
        model.eval()
        with torch.no_grad():
            r_tr = model(Mu_tr.to(device)).detach().cpu().numpy()
            r_va = model(Mu_va.to(device)).detach().cpu().numpy()
            r_te = model(Mu_te.to(device)).detach().cpu().numpy()
        c_tr = harrell_c_index(T_tr.numpy(), E_tr.numpy().astype(np.int32), r_tr)
        c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), r_va)
        c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), r_te)

        rec = {
            "epoch": epoch + 1,
            "train_cox_loss": tr_loss,
            "train_cindex": c_tr,
            "val_cindex": c_va,
            "test_whi_cindex": c_te,
        }

        def _clean(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: _clean(v) for k, v in rec.items()}) + "\n")

        log.info("epoch %03d/%d  cox=%.4f  c_tr=%.4f  c_va=%.4f  c_whi=%.4f",
                 epoch + 1, args.epochs, tr_loss, c_tr, c_va, c_te)

        improved = (not math.isnan(c_va)) and (c_va > best_c_va + 1e-4)
        if improved:
            best_c_va = c_va
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_streak = 0
        else:
            bad_streak += 1
            if epoch + 1 >= args.min_epochs and bad_streak >= args.patience:
                log.info("Early stop at epoch %d (no val improvement for %d epochs).",
                         epoch + 1, bad_streak)
                break

    # restore best & save
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        r_tr = model(Mu_tr.to(device)).detach().cpu().numpy()
        r_va = model(Mu_va.to(device)).detach().cpu().numpy()
        r_te = model(Mu_te.to(device)).detach().cpu().numpy()
    c_tr = harrell_c_index(T_tr.numpy(), E_tr.numpy().astype(np.int32), r_tr)
    c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), r_va)
    c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), r_te)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "latent_dim": latent_dim,
                "hidden": list(hidden),
                "dropout": float(args.dropout),
            },
            "best_epoch": int(best_epoch),
            "best_val_cindex": float(best_c_va),
        },
        out_dir / "dann_surv_model.pt",
    )
    log.info("BEST  c_tr=%.4f  c_val_fhs=%.4f  c_whi=%.4f  (epoch %d)",
             c_tr, c_va, c_te, best_epoch)

    # save risk predictions (FHS-val + WHI) for downstream analysis
    pred_rows = []
    for i, gi in enumerate(va_idx):
        pred_rows.append(
            f"FHS_val,{int(gi)},{float(T_va[i].item()):.6f},{int(E_va[i].item())},{float(r_va[i]):.6f}"
        )
    for i in range(len(T_te)):
        pred_rows.append(
            f"WHI,{i},{float(T_te[i].item()):.6f},{int(E_te[i].item())},{float(r_te[i]):.6f}"
        )
    (out_dir / "dann_surv_predictions.csv").write_text(
        "cohort,idx,time,event,risk\n" + "\n".join(pred_rows) + "\n", encoding="utf-8"
    )

    meta: Dict[str, object] = {
        "device": str(device),
        "n_fhs_train": int(len(tr_idx)),
        "n_fhs_val": int(len(va_idx)),
        "n_whi_test": int(len(T_te)),
        "events_fhs_train": int(E_tr.sum().item()),
        "events_fhs_val": int(E_va.sum().item()),
        "events_whi_test": int(E_te.sum().item()),
        "dann_encoder_ckpt": str(ckpt_path),
        "dann_preprocess_npz": str(npz_path),
        "dann_latent_dim": int(latent_dim),
        "dann_proj_dim": int(cfg["proj_dim"]),
        "dann_n_cpg": int(n_cpg),
        "dann_n_snp": int(n_snp),
        "dann_d_in": int(d_in),
        "head_hidden": list(hidden),
        "head_dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "best_epoch": int(best_epoch),
        "best_val_cindex": float(best_c_va),
        "final_train_cindex": float(c_tr),
        "final_val_cindex": float(c_va),
        "final_whi_cindex": float(c_te),
    }
    (out_dir / "dann_surv_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Wrote %s and %s", out_dir / "dann_surv_model.pt", out_dir / "dann_surv_run_meta.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
