#!/usr/bin/env python3
"""
AESURV head on frozen DANN latent **with cohort-portable auxiliary targets**
(age + Houseman cell composition).

Extension of train_aesurv_dann_latent.py. The head's small VAE bottleneck must
now also reconstruct two cohort-portable biological targets:

    age                       (1-d)
    cell composition          (default 6-d: B, NK, CD4T, CD8T, Mono, Neutro)

Both labels are available in FHS and WHI, so the auxiliary loss uses **both
cohorts** -- it adds supervised signal on WHI that does not leak the survival
label, and forces the bottleneck z to encode biology that transports across
cohorts.

Loss:
    L = L_cox  (FHS only)
      + alpha  * recon(decoded, dann_mu)              (FHS samples)
      + beta   * KL(z)                                (FHS samples)
      + gamma  * cohort_adv_CE                        (FHS + WHI, balanced)
      + delta_age  * MSE(age_pred, age_true)          (FHS + WHI, balanced)
      + delta_cell * MSE(cell_pred, cell_true)        (FHS + WHI, balanced)

Aux targets are z-scored using the pooled FHS+WHI mean/std so the MSE scale is
comparable to the Cox loss.

Usage:
    python train_aesurv_dann_latent_aux.py ^
      --dann-encoder-ckpt   runs/mini_vae_dann_mmd_rich/mini_dann_model.pt ^
      --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz ^
      --fhs-cell-comp-parquet FHS_cell_composition.parquet ^
      --whi-cell-comp-parquet WHI_cell_composition.parquet ^
      --aux-age-weight 1.0 --aux-cell-weight 1.0 ^
      --out-dir runs/aesurv_dann_rich_aux_v1

``--input-modality`` can be ``both`` (default), ``cpgs_only``, or ``snps_only``;
in the latter two, the masked block is set to the DANN scaler mean so that block
is all zeros after standardization.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dann_aesurv_bridge import (
    DannLatentEncoder,
    InvariantPreprocessor,
    load_mini_dann_for_fusion,
)
from train_aesurv_dann_latent import grad_reverse  # GRL helper
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

_THRIFT_LIMIT = 2_147_483_647


# --------------------------------------------------------------------------- #
# AESURV head with aux outputs                                                #
# --------------------------------------------------------------------------- #
class AESurvHeadAux(nn.Module):
    """AESURV head + cohort adversary + age head + cell-composition head.

    All heads share the same small VAE bottleneck ``z`` (z_dim).
    """

    def __init__(
        self,
        in_dim: int,
        enc_hidden: Tuple[int, ...] = (64, 32),
        dec_hidden: Tuple[int, ...] = (32, 64),
        z_dim: int = 16,
        cohort_hidden: int = 8,
        dropout: float = 0.30,
        n_cells: int = 6,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.z_dim = int(z_dim)
        self.n_cells = int(n_cells)

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
        self.age_head = nn.Linear(self.z_dim, 1)
        self.cell_head = nn.Linear(self.z_dim, self.n_cells)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h).clamp(min=-8.0, max=8.0)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, sample_z: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar) if (sample_z and self.training) else mu
        x_rec = self.decoder(z)
        log_h = self.cox_head(z).squeeze(-1)
        age_pred = self.age_head(z).squeeze(-1)
        cell_pred = self.cell_head(z)
        return x_rec, z, log_h, mu, logvar, age_pred, cell_pred

    def cohort_logits(self, z: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.cohort_head(grad_reverse(z, lam))


# --------------------------------------------------------------------------- #
# Aux loading                                                                  #
# --------------------------------------------------------------------------- #
def _open_parquet(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(str(path),
                              thrift_string_size_limit=_THRIFT_LIMIT,
                              thrift_container_size_limit=_THRIFT_LIMIT)
    except TypeError:
        return pq.ParquetFile(str(path))


def _read_id_age(parquet_path: Path, id_col: str) -> pd.DataFrame:
    """Read just the id + age columns from a (huge) parquet, preserving row order."""
    pf = _open_parquet(parquet_path)
    sch = getattr(pf, "schema_arrow", None) or pf.schema
    all_cols = list(sch.names)
    if id_col not in all_cols:
        raise SystemExit(f"id col {id_col!r} not in {parquet_path}")
    if "age" not in all_cols:
        raise SystemExit(f"'age' column not found in {parquet_path}")
    table = pf.read(columns=[id_col, "age"])
    df = table.to_pandas()
    df[id_col] = df[id_col].astype(str)
    return df


def _apply_dann_input_modality(
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    mode: str,
    mean_full: np.ndarray,
    n_cpg: int,
    n_snp: int,
    log: logging.Logger,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace a modality with the DANN scaler **mean** so its standardized block is all zeros.

    ``both``: no change. ``cpgs_only``: SNP block = population mean. ``snps_only``: meth = mean.
    """
    if mode == "both":
        return X_meth, X_snp
    mean_full = np.asarray(mean_full, dtype=np.float32).ravel()
    if mean_full.shape[0] != n_cpg + n_snp:
        raise SystemExit(f"mean_ length {mean_full.shape[0]} != n_cpg+n_snp={n_cpg + n_snp}")
    mmean = mean_full[:n_cpg]
    smean = mean_full[n_cpg:]
    if mode == "cpgs_only":
        Xm = np.asarray(X_meth, dtype=np.float32)
        Xs = np.broadcast_to(smean, (Xm.shape[0], n_snp)).astype(np.float32).copy()
        log.info("[%s] input modality=cpgs_only (SNP block fixed to scaler mean; n_cpg=%d n_snp=%d)",
                 label, n_cpg, n_snp)
        return Xm, Xs
    if mode == "snps_only":
        Xs = np.asarray(X_snp, dtype=np.float32)
        Xm = np.broadcast_to(mmean, (Xs.shape[0], n_cpg)).astype(np.float32).copy()
        log.info("[%s] input modality=snps_only (methylation block fixed to scaler mean)", label)
        return Xm, Xs
    raise SystemExit(f"Unknown --input-modality {mode!r} (use both|cpgs_only|snps_only)")


def load_aux_targets(
    parquet_path: Path,
    cell_comp_path: Path,
    id_col: str,
    cell_columns: List[str],
    log: logging.Logger,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (age_array, cell_matrix) aligned with parquet row order.

    Joins by id_col rather than positional index, so silent misalignment is
    impossible. Asserts no NaNs after the join.
    """
    log.info("[%s] reading id + age from %s ...", label, parquet_path.name)
    df_main = _read_id_age(parquet_path, id_col)
    log.info("[%s] reading cell composition from %s ...", label, cell_comp_path.name)
    df_cell = pd.read_parquet(cell_comp_path)
    if id_col not in df_cell.columns:
        raise SystemExit(
            f"{cell_comp_path} is missing id column {id_col!r}; cols={list(df_cell.columns)}"
        )
    df_cell[id_col] = df_cell[id_col].astype(str)
    missing_cells = [c for c in cell_columns if c not in df_cell.columns]
    if missing_cells:
        raise SystemExit(f"Cells missing in {cell_comp_path}: {missing_cells}")
    df_cell = df_cell[[id_col] + cell_columns].copy()

    df_join = df_main.merge(df_cell, on=id_col, how="left",
                            suffixes=("_main", "_cell"))
    n_missing_cell = int(df_join[cell_columns[0]].isna().sum())
    if n_missing_cell:
        raise SystemExit(
            f"[{label}] {n_missing_cell} rows missing cell composition after merge."
        )
    n_missing_age = int(df_join["age"].isna().sum())
    if n_missing_age:
        log.warning("[%s] %d rows have NaN age; will impute with cohort mean.",
                    label, n_missing_age)
        df_join["age"] = df_join["age"].fillna(df_join["age"].mean())

    age = df_join["age"].to_numpy(dtype=np.float32)
    cells = df_join[cell_columns].to_numpy(dtype=np.float32)
    log.info("[%s] age: mean=%.1f sd=%.1f range=[%.1f, %.1f]",
             label, float(age.mean()), float(age.std()), float(age.min()), float(age.max()))
    for i, c in enumerate(cell_columns):
        log.info("  %s  mean=%.3f sd=%.3f range=[%.3f, %.3f]",
                 c, float(cells[:, i].mean()), float(cells[:, i].std()),
                 float(cells[:, i].min()), float(cells[:, i].max()))
    return age, cells


def _json_safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _survival_ipcw_td_auroc_auprc(
    ipcw_times: np.ndarray,
    ipcw_events: np.ndarray,
    test_times: np.ndarray,
    test_events: np.ndarray,
    risk_te: np.ndarray,
    prefix: str,
    *,
    n_times: int = 48,
    log: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """IPCW time-dependent AUROC + landmark AUPRC.

    ``ipcw_times`` / ``ipcw_events`` define ``survival_train`` for Uno's weights.
    For FHS validation use FHS **training**; for WHI test use WHI itself when FHS-train
    IPCW has no support at WHI event times (longer follow-up than FHS train max).
    """
    out: Dict[str, object] = {}
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sklearn.metrics import average_precision_score
    except ImportError as err:
        if log:
            log.warning("Skipping %s TD-AUROC/AUPRC (need scikit-survival, sklearn): %s", prefix, err)
        return out

    t_ip = np.asarray(ipcw_times, dtype=np.float64).ravel()
    e_ip = np.asarray(ipcw_events, dtype=np.int32).ravel()
    t_te = np.asarray(test_times, dtype=np.float64).ravel()
    e_te = np.asarray(test_events, dtype=np.int32).ravel()
    risk_te = np.asarray(risk_te, dtype=np.float64).ravel()

    y_ip = np.empty(len(t_ip), dtype=[("event", np.bool_), ("time", np.float64)])
    y_ip["event"] = e_ip.astype(np.bool_)
    y_ip["time"] = t_ip
    y_te = np.empty(len(t_te), dtype=[("event", np.bool_), ("time", np.float64)])
    y_te["event"] = e_te.astype(np.bool_)
    y_te["time"] = t_te

    ev_t = t_te[e_te == 1]
    if len(ev_t) < 10:
        if log:
            log.warning("Skipping %s TD-AUROC: too few events in test (%d)", prefix, len(ev_t))
        return out

    qs = np.linspace(0.03, 0.97, min(n_times, max(10, len(ev_t))))
    eval_times = np.unique(np.clip(np.quantile(ev_t, qs), 1e-8, None))
    tmax_ip = float(np.max(t_ip))
    eval_times = eval_times[eval_times < tmax_ip - 1e-9]
    if len(eval_times) < 4:
        lo = float(np.min(ev_t))
        hi = min(float(np.max(ev_t)), tmax_ip * 0.999)
        if hi <= lo + 1e-9:
            return out
        eval_times = np.linspace(lo, hi, min(max(n_times, 8), 30))
        eval_times = np.unique(eval_times[eval_times < tmax_ip - 1e-9])
    if len(eval_times) < 4:
        return out

    try:
        aucs, mean_auc = cumulative_dynamic_auc(y_ip, y_te, risk_te, eval_times)
    except Exception as err:
        if log:
            log.warning("cumulative_dynamic_auc failed for %s: %s", prefix, err)
        return out

    auc_arr = np.asarray(aucs, dtype=np.float64).ravel()
    out[f"{prefix}_td_auroc_mean_of_times"] = _json_safe_float(float(np.nanmean(auc_arr)))
    out[f"{prefix}_td_auroc_integrated_mean"] = _json_safe_float(float(mean_auc))
    out[f"{prefix}_td_auroc_times"] = [_json_safe_float(t) for t in eval_times]
    out[f"{prefix}_td_auroc_values"] = [_json_safe_float(a) for a in auc_arr]

    med = float(np.median(ev_t))
    y_bin = ((t_te <= med) & (e_te == 1)).astype(np.int32)
    if int(y_bin.sum()) > 0 and int((y_bin == 0).sum()) > 0:
        auprc = float(average_precision_score(y_bin, risk_te))
        out[f"{prefix}_auprc_landmark"] = _json_safe_float(auprc)
        out[f"{prefix}_auprc_landmark_time"] = _json_safe_float(med)

    if log and out:
        log.info(
            "%s: TD-AUROC mean(times)=%s integrated_mean=%s AUPRC@median=%s",
            prefix,
            out.get(f"{prefix}_td_auroc_mean_of_times"),
            out.get(f"{prefix}_td_auroc_integrated_mean"),
            out.get(f"{prefix}_auprc_landmark"),
        )
    return out


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="AESURV head + cell/age aux on DANN latent.")
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
    p.add_argument(
        "--input-modality",
        type=str,
        default="both",
        choices=("both", "cpgs_only", "snps_only"),
        help="Frozen DANN input: both omics; CpGs only (SNPs = scaler mean); SNPs only (meth = mean).",
    )
    # aux targets
    p.add_argument("--fhs-cell-comp-parquet", type=str, default="FHS_cell_composition.parquet")
    p.add_argument("--whi-cell-comp-parquet", type=str, default="WHI_cell_composition.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--whi-id-col", type=str, default="sample_ID")
    p.add_argument("--aux-cells", type=str, default="B,NK,CD4T,CD8T,Mono,Neutro",
                   help="Comma-separated cell columns to use as aux targets (skip Eosino: ~0).")
    p.add_argument("--aux-age-weight", type=float, default=1.0)
    p.add_argument("--aux-cell-weight", type=float, default=1.0)
    # head
    p.add_argument("--enc-hidden", type=str, default="64,32")
    p.add_argument("--dec-hidden", type=str, default="32,64")
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--cohort-hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.40)
    # primary losses
    p.add_argument("--alpha-recon", type=float, default=5.0)
    p.add_argument("--beta-kl", type=float, default=0.05)
    p.add_argument("--beta-kl-warmup-epochs", type=int, default=10)
    p.add_argument("--gamma-adv", type=float, default=0.10)
    p.add_argument("--lam-adv-max", type=float, default=1.0)
    p.add_argument("--lam-adv-warmup-epochs", type=int, default=15)
    # optimization
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
    # output
    p.add_argument("--out-dir", type=str, default="runs/aesurv_dann_aux")
    p.add_argument("--log-file", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    out_dir = _resolve_data_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _resolve_data_path(args.log_file) if args.log_file else (out_dir / "aesurv_aux.log")
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
    log.info("DANN: latent_dim=%d proj_dim=%d meth_as_mvalues=%s input_modality=%s",
             latent_dim, int(cfg["proj_dim"]), meth_mval, args.input_modality)

    mean_np = preproc.mean_.detach().cpu().numpy().astype(np.float32)
    X_meth_fhs, X_snp_fhs = _apply_dann_input_modality(
        X_meth_fhs, X_snp_fhs, args.input_modality, mean_np, n_cpg, n_snp, log, "FHS",
    )
    X_meth_whi, X_snp_whi = _apply_dann_input_modality(
        X_meth_whi, X_snp_whi, args.input_modality, mean_np, n_cpg, n_snp, log, "WHI",
    )

    # --- DANN latent (cached) ---
    latent_cache = _resolve_data_path(args.latent_cache_dir)
    latent_tag = "" if args.input_modality == "both" else str(args.input_modality)
    mu_fhs = load_or_extract_latent("FHS", X_meth_fhs, X_snp_fhs, preproc, encoder,
                                    meth_as_mvalues=meth_mval, latent_dim=latent_dim, device=device,
                                    cache_dir=latent_cache, ckpt_path=ckpt_path, npz_path=npz_path,
                                    use_cache=not args.no_cache, log=log, extra_tag=latent_tag)
    mu_whi = load_or_extract_latent("WHI", X_meth_whi, X_snp_whi, preproc, encoder,
                                    meth_as_mvalues=meth_mval, latent_dim=latent_dim, device=device,
                                    cache_dir=latent_cache, ckpt_path=ckpt_path, npz_path=npz_path,
                                    use_cache=not args.no_cache, log=log, extra_tag=latent_tag)

    # --- aux targets (age + cell composition) ---
    cell_cols = [c.strip() for c in args.aux_cells.split(",") if c.strip()]
    age_fhs, cell_fhs = load_aux_targets(
        fhs_pq, _resolve_data_path(args.fhs_cell_comp_parquet),
        args.fhs_id_col, cell_cols, log, "FHS",
    )
    age_whi, cell_whi = load_aux_targets(
        whi_pq, _resolve_data_path(args.whi_cell_comp_parquet),
        args.whi_id_col, cell_cols, log, "WHI",
    )
    if age_fhs.shape[0] != mu_fhs.shape[0]:
        raise SystemExit(f"FHS aux ({age_fhs.shape[0]}) != FHS latent ({mu_fhs.shape[0]})")
    if age_whi.shape[0] != mu_whi.shape[0]:
        raise SystemExit(f"WHI aux ({age_whi.shape[0]}) != WHI latent ({mu_whi.shape[0]})")

    # z-score on pooled FHS+WHI for stable MSE scaling
    age_all = np.concatenate([age_fhs, age_whi])
    cell_all = np.concatenate([cell_fhs, cell_whi], axis=0)
    age_mu_g = float(age_all.mean()); age_sd_g = float(age_all.std() + 1e-6)
    cell_mu_g = cell_all.mean(axis=0)
    cell_sd_g = cell_all.std(axis=0) + 1e-6
    age_fhs_z = ((age_fhs - age_mu_g) / age_sd_g).astype(np.float32)
    age_whi_z = ((age_whi - age_mu_g) / age_sd_g).astype(np.float32)
    cell_fhs_z = ((cell_fhs - cell_mu_g) / cell_sd_g).astype(np.float32)
    cell_whi_z = ((cell_whi - cell_mu_g) / cell_sd_g).astype(np.float32)
    log.info("Aux z-score globals: age mu=%.2f sd=%.2f | cells mu=%s sd=%s",
             age_mu_g, age_sd_g, np.round(cell_mu_g, 3).tolist(),
             np.round(cell_sd_g, 3).tolist())

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

    Age_tr_z = torch.from_numpy(age_fhs_z[tr_idx])
    Age_va_z = torch.from_numpy(age_fhs_z[va_idx])
    Age_te_z = torch.from_numpy(age_whi_z)
    Cell_tr_z = torch.from_numpy(cell_fhs_z[tr_idx])
    Cell_va_z = torch.from_numpy(cell_fhs_z[va_idx])
    Cell_te_z = torch.from_numpy(cell_whi_z)

    # Cox loader (FHS-train, also carries aux targets for in-batch aux loss).
    ds_tr = TensorDataset(Mu_tr, T_tr, E_tr, Age_tr_z, Cell_tr_z)
    if args.balance_events:
        w = np.where(E_tr.numpy() > 0.5, 1.0 / max(1, int(E_tr.sum().item())),
                     1.0 / max(1, len(E_tr) - int(E_tr.sum().item())))
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(w), replacement=True)
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=False)
        log.info("Train sampling: balanced events")
    else:
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        log.info("Train sampling: shuffle")

    # Adversary + aux loader: FHS-full + WHI-full, balanced per-cohort sampling.
    Mu_adv = torch.from_numpy(np.concatenate([mu_fhs, mu_whi], axis=0)).float()
    C_adv = torch.from_numpy(np.concatenate(
        [np.zeros(mu_fhs.shape[0], dtype=np.int64),
         np.ones(mu_whi.shape[0], dtype=np.int64)], axis=0))
    Age_adv_z = torch.from_numpy(np.concatenate([age_fhs_z, age_whi_z], axis=0))
    Cell_adv_z = torch.from_numpy(np.concatenate([cell_fhs_z, cell_whi_z], axis=0))
    w_adv = np.where(C_adv.numpy() == 0, 1.0 / mu_fhs.shape[0], 1.0 / mu_whi.shape[0])
    sampler_adv = WeightedRandomSampler(
        torch.from_numpy(w_adv).double(),
        num_samples=max(args.batch_size * 8, len(C_adv) // 4),
        replacement=True,
    )
    loader_adv = DataLoader(
        TensorDataset(Mu_adv, C_adv, Age_adv_z, Cell_adv_z),
        batch_size=args.batch_size, sampler=sampler_adv, num_workers=0, drop_last=False,
    )
    log.info("Adversary+aux loader: n_FHS=%d n_WHI=%d (balanced)", mu_fhs.shape[0], mu_whi.shape[0])

    # --- model ---
    enc_h = tuple(int(x) for x in args.enc_hidden.split(",") if x.strip())
    dec_h = tuple(int(x) for x in args.dec_hidden.split(",") if x.strip())
    model = AESurvHeadAux(
        in_dim=latent_dim, enc_hidden=enc_h, dec_hidden=dec_h,
        z_dim=args.z_dim, cohort_hidden=args.cohort_hidden, dropout=args.dropout,
        n_cells=len(cell_cols),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("AESurvHeadAux: in=%d enc=%s dec=%s z=%d cohort_h=%d cells=%d dropout=%.2f params=%d",
             latent_dim, enc_h, dec_h, args.z_dim, args.cohort_hidden, len(cell_cols),
             args.dropout, n_params)

    # --- training loop ---
    metrics_path = out_dir / "aesurv_aux_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_c_va = -1.0; best_epoch = -1; best_state: Optional[dict] = None; bad_streak = 0
    adv_iter = iter(loader_adv)

    @torch.no_grad()
    def eval_split(Mu_t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_rec, z, log_h, mu_h, _, age_pred, cell_pred = model(Mu_t.to(device), sample_z=False)
        return log_h.cpu().numpy(), age_pred.cpu().numpy(), cell_pred.cpu().numpy()

    for epoch in range(args.epochs):
        beta_kl = args.beta_kl * min(1.0, (epoch + 1) / max(1, args.beta_kl_warmup_epochs))
        lam_adv = args.lam_adv_max * min(1.0, (epoch + 1) / max(1, args.lam_adv_warmup_epochs))

        model.train()
        sum_total = sum_cox = sum_rec = sum_kl = sum_adv = sum_age = sum_cell = 0.0
        adv_correct = adv_seen = 0; n_seen = 0
        for x, t, e, age_z, cell_z in loader_tr:
            if int(e.sum().item()) == 0:
                continue
            x = x.to(device); t = t.to(device); e = e.to(device)
            age_z = age_z.to(device); cell_z = cell_z.to(device)
            try:
                xa, ca, aa, ka = next(adv_iter)
            except StopIteration:
                adv_iter = iter(loader_adv); xa, ca, aa, ka = next(adv_iter)
            xa = xa.to(device); ca = ca.to(device); aa = aa.to(device); ka = ka.to(device)

            opt.zero_grad(set_to_none=True)
            # FHS forward: Cox + recon + KL + aux(FHS)
            x_rec, z, log_h, mu_h, logvar_h, age_pred, cell_pred = model(x, sample_z=True)
            cox = cox_ph_loss(log_h, t, e)
            rec = F.smooth_l1_loss(x_rec, x)
            kl = (-0.5 * (1.0 + logvar_h - mu_h.pow(2) - logvar_h.exp())).sum(dim=1).mean()
            # aux loss on FHS samples (weight halved; we'll also add aux on the balanced batch below)
            aux_age_fhs = F.mse_loss(age_pred, age_z)
            aux_cell_fhs = F.mse_loss(cell_pred, cell_z)

            # FHS+WHI forward (balanced): adversary + aux(both cohorts)
            _, z_adv, _, _, _, age_pred_b, cell_pred_b = model(xa, sample_z=True)
            adv = F.cross_entropy(model.cohort_logits(z_adv, lam=lam_adv), ca)
            aux_age_bal = F.mse_loss(age_pred_b, aa)
            aux_cell_bal = F.mse_loss(cell_pred_b, ka)

            # Aux loss combines both forward passes (equal weight; balanced batch already
            # sees WHI, FHS pass keeps gradient flow tied to the FHS Cox/recon path).
            aux_age = 0.5 * (aux_age_fhs + aux_age_bal)
            aux_cell = 0.5 * (aux_cell_fhs + aux_cell_bal)

            loss = (cox
                    + args.alpha_recon * rec
                    + beta_kl * kl
                    + args.gamma_adv * adv
                    + args.aux_age_weight * aux_age
                    + args.aux_cell_weight * aux_cell)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            bs = x.size(0); n_seen += bs
            sum_total += float(loss.item()) * bs
            sum_cox += float(cox.item()) * bs
            sum_rec += float(rec.item()) * bs
            sum_kl += float(kl.item()) * bs
            sum_adv += float(adv.item()) * bs
            sum_age += float(aux_age.item()) * bs
            sum_cell += float(aux_cell.item()) * bs
            with torch.no_grad():
                pred = model.cohort_logits(z_adv, lam=lam_adv).argmax(dim=1)
                adv_correct += int((pred == ca).sum().item())
                adv_seen += int(ca.numel())

        denom = max(1, n_seen)
        tr = {
            "total": sum_total / denom, "cox": sum_cox / denom,
            "rec": sum_rec / denom, "kl": sum_kl / denom, "adv": sum_adv / denom,
            "age": sum_age / denom, "cell": sum_cell / denom,
            "adv_acc": adv_correct / max(1, adv_seen),
        }

        # eval (deterministic forward: z = mu_h)
        model.eval()
        log_h_tr, age_tr_p, cell_tr_p = eval_split(Mu_tr)
        log_h_va, age_va_p, cell_va_p = eval_split(Mu_va)
        log_h_te, age_te_p, cell_te_p = eval_split(Mu_te)
        c_tr = harrell_c_index(T_tr.numpy(), E_tr.numpy().astype(np.int32), log_h_tr)
        c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), log_h_va)
        c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), log_h_te)

        # aux quality in *original* units (de-standardize) for interpretability
        age_rmse_va = float(np.sqrt(np.mean(
            ((age_va_p * age_sd_g + age_mu_g) - (Age_va_z.numpy() * age_sd_g + age_mu_g))**2
        )))
        age_rmse_te = float(np.sqrt(np.mean(
            ((age_te_p * age_sd_g + age_mu_g) - (Age_te_z.numpy() * age_sd_g + age_mu_g))**2
        )))
        cell_rmse_va = float(np.sqrt(np.mean(
            ((cell_va_p * cell_sd_g + cell_mu_g) - (Cell_va_z.numpy() * cell_sd_g + cell_mu_g))**2
        )))
        cell_rmse_te = float(np.sqrt(np.mean(
            ((cell_te_p * cell_sd_g + cell_mu_g) - (Cell_te_z.numpy() * cell_sd_g + cell_mu_g))**2
        )))

        rec_d = {
            "epoch": epoch + 1,
            "lam_adv": lam_adv, "beta_kl": beta_kl,
            "train_total": tr["total"], "train_cox": tr["cox"],
            "train_recon": tr["rec"], "train_kl": tr["kl"], "train_adv": tr["adv"],
            "train_aux_age": tr["age"], "train_aux_cell": tr["cell"],
            "train_adv_acc": tr["adv_acc"],
            "train_cindex": c_tr, "val_cindex": c_va, "test_whi_cindex": c_te,
            "val_age_rmse": age_rmse_va, "whi_age_rmse": age_rmse_te,
            "val_cell_rmse": cell_rmse_va, "whi_cell_rmse": cell_rmse_te,
        }

        def _clean(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: _clean(v) for k, v in rec_d.items()}) + "\n")

        log.info("ep %03d/%d  lam=%.2f bkl=%.3f  cox=%.3f rec=%.3f kl=%.3f adv=%.3f age=%.3f cell=%.3f advAcc=%.3f  "
                 "c_tr=%.4f c_va=%.4f c_whi=%.4f  ageRMSE va=%.2f whi=%.2f  cellRMSE va=%.3f whi=%.3f",
                 epoch + 1, args.epochs, lam_adv, beta_kl,
                 tr["cox"], tr["rec"], tr["kl"], tr["adv"], tr["age"], tr["cell"], tr["adv_acc"],
                 c_tr, c_va, c_te, age_rmse_va, age_rmse_te, cell_rmse_va, cell_rmse_te)

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

    # restore best & evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    log_h_tr, age_tr_p, cell_tr_p = eval_split(Mu_tr)
    log_h_va, age_va_p, cell_va_p = eval_split(Mu_va)
    log_h_te, age_te_p, cell_te_p = eval_split(Mu_te)
    c_tr = harrell_c_index(T_tr.numpy(), E_tr.numpy().astype(np.int32), log_h_tr)
    c_va = harrell_c_index(T_va.numpy(), E_va.numpy().astype(np.int32), log_h_va)
    c_te = harrell_c_index(T_te.numpy(), E_te.numpy().astype(np.int32), log_h_te)

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "in_dim": latent_dim, "enc_hidden": list(enc_h), "dec_hidden": list(dec_h),
            "z_dim": int(args.z_dim), "cohort_hidden": int(args.cohort_hidden),
            "dropout": float(args.dropout), "n_cells": len(cell_cols),
            "cell_columns": cell_cols,
            "aux_norm": {
                "age_mu": age_mu_g, "age_sd": age_sd_g,
                "cell_mu": cell_mu_g.tolist(), "cell_sd": cell_sd_g.tolist(),
            },
        },
        "best_epoch": int(best_epoch), "best_val_cindex": float(best_c_va),
    }, out_dir / "aesurv_aux_model.pt")
    log.info("BEST  c_tr=%.4f  c_val_fhs=%.4f  c_whi=%.4f  (epoch %d)",
             c_tr, c_va, c_te, best_epoch)

    np.savez(
        out_dir / "aesurv_aux_risk.npz",
        risk_fhs_train=log_h_tr.astype(np.float32),
        risk_fhs_val=log_h_va.astype(np.float32),
        risk_whi_test=log_h_te.astype(np.float32),
        time_fhs_train=T_tr.numpy().astype(np.float32),
        time_fhs_val=T_va.numpy().astype(np.float32),
        time_whi_test=T_te.numpy().astype(np.float32),
        event_fhs_train=E_tr.numpy().astype(np.int32),
        event_fhs_val=E_va.numpy().astype(np.int32),
        event_whi_test=E_te.numpy().astype(np.int32),
        age_pred_fhs_train=(age_tr_p * age_sd_g + age_mu_g).astype(np.float32),
        age_pred_fhs_val=(age_va_p * age_sd_g + age_mu_g).astype(np.float32),
        age_pred_whi=(age_te_p * age_sd_g + age_mu_g).astype(np.float32),
        cell_pred_fhs_train=(cell_tr_p * cell_sd_g + cell_mu_g).astype(np.float32),
        cell_pred_fhs_val=(cell_va_p * cell_sd_g + cell_mu_g).astype(np.float32),
        cell_pred_whi=(cell_te_p * cell_sd_g + cell_mu_g).astype(np.float32),
    )

    meta: Dict[str, object] = {
        "input_modality": str(args.input_modality),
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
        "cell_columns": cell_cols,
        "alpha_recon": float(args.alpha_recon),
        "beta_kl": float(args.beta_kl),
        "beta_kl_warmup_epochs": int(args.beta_kl_warmup_epochs),
        "gamma_adv": float(args.gamma_adv),
        "lam_adv_max": float(args.lam_adv_max),
        "lam_adv_warmup_epochs": int(args.lam_adv_warmup_epochs),
        "aux_age_weight": float(args.aux_age_weight),
        "aux_cell_weight": float(args.aux_cell_weight),
        "lr": float(args.lr), "weight_decay": float(args.weight_decay),
        "best_epoch": int(best_epoch), "best_val_cindex": float(best_c_va),
        "final_train_cindex": float(c_tr),
        "final_val_cindex": float(c_va),
        "final_whi_cindex": float(c_te),
    }
    meta.update(
        _survival_ipcw_td_auroc_auprc(
            T_tr.numpy(), E_tr.numpy().astype(np.int32),
            T_va.numpy(), E_va.numpy().astype(np.int32),
            np.asarray(log_h_va, dtype=np.float64).ravel(),
            "fhs_val",
            n_times=48,
            log=log,
        )
    )
    meta.update(
        _survival_ipcw_td_auroc_auprc(
            T_te.numpy(), E_te.numpy().astype(np.int32),
            T_te.numpy(), E_te.numpy().astype(np.int32),
            np.asarray(log_h_te, dtype=np.float64).ravel(),
            "whi_test",
            n_times=48,
            log=log,
        )
    )
    (out_dir / "aesurv_aux_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Wrote %s and %s", out_dir / "aesurv_aux_model.pt", out_dir / "aesurv_aux_run_meta.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
