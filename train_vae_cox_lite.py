#!/usr/bin/env python3
"""
Efficient training: VAE (bottleneck recon) + Cox loss on FHS/WHI-style combined .npz data.

Data layout (same as merge_parquet_with_snp combined_training):
  - NPZ: X_methylation (N x n_cpg), X_snp (N x n_snp), iid
  - Sibling txts: *_cpg_columns.txt, *_snp_genotypes.snp_columns.txt
  - Parquet: survival columns (time, event) + ID column aligned with iid row order

Efficiency choices:
  - Never materialize a single (N x 1M) feature matrix; two modality tensors only.
  - Optional --max-cpg / --max-snp column subsets for smoke tests.
  - np.load(..., mmap_mode='r') when arrays are stored uncompressed in npz.
  - DataLoader: num_workers, pin_memory, persistent_workers, prefetch factor.
  - torch.compile optional; AMP optional.

Defaults assume this repo is d:\\SNP_datasets (same directory as this file).

Exact data paths used by defaults:

  d:\\SNP_datasets\\FHS_methylation_with_snp_1milfeatures_combined_training.npz
  d:\\SNP_datasets\\FHS_methylation_with_snp_1milfeatures.parquet
  d:\\SNP_datasets\\vae_cox_lite_checkpoint.pt   (written by --save default)
  d:\\SNP_datasets\\vae_cox_lite_train.log       (append log file; use --no-log-file to disable)

Logging: console INFO by default; file DEBUG by default. Verbose console: add -v or --log-level DEBUG.
Per-batch training logs: --log-every-train-batch. Per-batch val: --log-every-val-batch.

Optional WHI external test (pass both flags together):

  d:\\SNP_datasets\\WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz
  d:\\SNP_datasets\\WHI_methylation_with_snp_merged_1milfeatures.parquet

Run (no arguments required if those files exist):

  cmd.exe:  cd /d d:\\SNP_datasets && python train_vae_cox_lite.py
  PowerShell:  Set-Location d:\\SNP_datasets; python .\\train_vae_cox_lite.py

Or run:  d:\\SNP_datasets\\run_vae_cox_lite.ps1

If your aligned FHS table is FHS_methylation_with_snp.parquet instead, run:

  python train_vae_cox_lite.py --meta-parquet d:\\SNP_datasets\\FHS_methylation_with_snp.parquet
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from vae_cox_lite_model import (
    VAECoxLiteFusion,
    cox_ph_loss,
    joint_loss,
    kl_gaussian,
)

# Repository root (on your machine this is d:\SNP_datasets when the script lives there).
_REPO_ROOT = Path(__file__).resolve().parent


def _resolve_data_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else _REPO_ROOT / pp


class _FlushStreamHandler(logging.StreamHandler):
    """Stdout/stderr handler that flushes after every record (live terminal output)."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_run_logging(
    *,
    log_file: Optional[Path],
    console_level: int,
    file_level: int,
) -> logging.Logger:
    """
    Root logger for this script: `vae_cox_lite`.
    Console and optional file; warnings captured to the same logger.
    """
    log = logging.getLogger("vae_cox_lite")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = _FlushStreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    logging.captureWarnings(True)
    warnings.filterwarnings("default")
    return log


def _parse_log_level(s: str) -> int:
    return getattr(logging, s.upper(), logging.INFO)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- small I/O helpers (avoid importing the 3k-line trainer) ---


def read_column_txt(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def stem_base_from_combined_training_npz(combined_path: Path) -> str:
    stem = combined_path.stem
    suf = "_combined_training"
    if not stem.endswith(suf):
        raise ValueError(
            f"Expected * _combined_training.npz from merge_parquet_with_snp, got: {combined_path.name}"
        )
    return stem[: -len(suf)].rstrip("_")


def paths_next_to_combined_training(
    combined_path: Path, snp_columns_txt_override: Optional[Path]
) -> Tuple[Path, Path]:
    base = stem_base_from_combined_training_npz(combined_path)
    parent = combined_path.parent
    cpg_txt = parent / f"{base}_cpg_columns.txt"
    if snp_columns_txt_override is not None:
        snp_txt = snp_columns_txt_override
    else:
        snp_txt = parent / f"{base}_snp_genotypes.snp_columns.txt"
    return cpg_txt, snp_txt


def detect_time_event(df: pd.DataFrame) -> Tuple[str, str]:
    df.columns = [str(c).strip() for c in df.columns]
    time_candidates = ["time", "followup_time", "follow_up_time"]
    event_candidates = ["event", "status", "mortality", "death"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    event_col = next((c for c in event_candidates if c in df.columns), None)
    if time_col is None or event_col is None:
        raise ValueError(f"Could not detect time/event. Columns: {list(df.columns)}")
    return time_col, event_col


def get_id_column(df: pd.DataFrame) -> str:
    for c in ["IID", "Share_ID", "sample_ID", "SUBJID", "Patient_ID", "SAMPLE_ID", "id"]:
        if c in df.columns:
            return c
    raise ValueError(f"No ID column found. Columns: {list(df.columns)}")


def concordance_index_safe(
    time: np.ndarray, risk: np.ndarray, event: np.ndarray
) -> Tuple[float, float]:
    """Harrell C for Cox linear predictor ``risk`` where larger = higher hazard / worse prognosis.

    Returns ``(ci_clinical, ci_flip)``:
      * ``ci_clinical`` — correct convention: higher risk ↔ shorter survival.
      * ``ci_flip`` — same scores passed with opposite sign (diagnostic).

    lifelines counts a pair correct when longer survival has a *larger* predicted score; their
    Cox example uses ``concordance_index(T, -partial_hazard, E)``. So for lifelines we pass
    ``-risk``. scikit-survival expects a risk score directly (higher = shorter survival), so we
    pass ``+risk``.
    """
    t = np.asarray(time, dtype=np.float64)
    r = np.asarray(risk, dtype=np.float64)
    e = np.asarray(event, dtype=np.int64)
    ci_clinical = float("nan")
    ci_flip = float("nan")
    try:
        from lifelines.utils import concordance_index as ci

        ci_clinical = float(ci(t, -r, e))
        ci_flip = float(ci(t, r, e))
    except ImportError:
        try:
            from sksurv.metrics import concordance_index_censored

            y_event = e.astype(bool)
            ci_clinical = float(concordance_index_censored(y_event, t, r)[0])
            ci_flip = float(concordance_index_censored(y_event, t, -r)[0])
        except ImportError:
            pass
    return ci_clinical, ci_flip


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SurvivalNPZDataset(Dataset):
    """Row-aligned meth + SNP + time + event."""

    def __init__(
        self,
        X_meth: np.ndarray,
        X_snp: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.X_meth = X_meth
        self.X_snp = X_snp
        self.time = time.astype(np.float32)
        self.event = event.astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        xm = torch.from_numpy(np.asarray(self.X_meth[j], dtype=np.float32))
        xs = torch.from_numpy(np.asarray(self.X_snp[j], dtype=np.float32))
        return xm, xs, torch.tensor(self.time[j]), torch.tensor(self.event[j])


def load_combined_bundle(
    combined_npz: Path,
    meta_parquet: Path,
    snp_columns_txt: Optional[Path],
    max_cpg: Optional[int],
    max_snp: Optional[int],
    label: str = "cohort",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    log = logging.getLogger("vae_cox_lite.data")
    log.info("[%s] Step: resolve paths next to combined NPZ", label)
    log.info("[%s] Combined NPZ: %s", label, combined_npz.resolve())
    log.info("[%s] Meta parquet: %s", label, meta_parquet.resolve())
    cpg_txt, snp_txt = paths_next_to_combined_training(combined_npz, snp_columns_txt)
    log.debug("[%s] CpG columns file: %s", label, cpg_txt.resolve())
    log.debug("[%s] SNP columns file: %s", label, snp_txt.resolve())
    if not cpg_txt.is_file():
        raise FileNotFoundError(f"Missing CpG list: {cpg_txt}")
    if not snp_txt.is_file():
        raise FileNotFoundError(f"Missing SNP list: {snp_txt}")

    meth_names = read_column_txt(cpg_txt)
    snp_names = read_column_txt(snp_txt)
    log.info(
        "[%s] Step: read column lists — n_cpg=%d n_snp=%d",
        label,
        len(meth_names),
        len(snp_names),
    )

    t_npz = time.perf_counter()
    z = np.load(combined_npz, allow_pickle=True, mmap_mode="r")
    log.info(
        "[%s] Step: opened NPZ in %.3fs; archive keys=%s",
        label,
        time.perf_counter() - t_npz,
        list(z.files),
    )
    if "X_methylation" not in z.files or "X_snp" not in z.files or "iid" not in z.files:
        raise ValueError(
            f"NPZ must contain X_methylation, X_snp, iid; got {list(z.files)}"
        )
    X_meth = z["X_methylation"]
    X_snp = z["X_snp"]
    iid_npz = np.asarray(z["iid"]).astype(str)
    log.info(
        "[%s] Step: array shapes X_methylation=%s X_snp=%s iid=%s dtype(meth)=%s dtype(snp)=%s",
        label,
        getattr(X_meth, "shape", None),
        getattr(X_snp, "shape", None),
        iid_npz.shape,
        getattr(X_meth, "dtype", None),
        getattr(X_snp, "dtype", None),
    )

    if X_meth.shape[1] != len(meth_names):
        raise ValueError(
            f"X_methylation cols {X_meth.shape[1]} != len(cpg txt) {len(meth_names)}"
        )
    if X_snp.shape[1] != len(snp_names):
        raise ValueError(f"X_snp cols {X_snp.shape[1]} != len(snp txt) {len(snp_names)}")
    if X_meth.shape[0] != X_snp.shape[0]:
        raise ValueError(f"Meth rows {X_meth.shape[0]} != SNP rows {X_snp.shape[0]}")

    t_pq = time.perf_counter()
    # For huge WHI tables, pyarrow's Thrift reader can hit size limits;
    # mirror the older pipeline: try fastparquet first, then fall back.
    try:
        df = pd.read_parquet(meta_parquet, engine="fastparquet")
        log.info("[%s] Step: read parquet with fastparquet", label)
    except Exception as e_fast:
        log.warning(
            "[%s] fastparquet failed (%s); falling back to pyarrow", label, e_fast
        )
        df = pd.read_parquet(meta_parquet, engine="pyarrow")
    log.info(
        "[%s] Step: read parquet in %.3fs — rows=%d cols=%d",
        label,
        time.perf_counter() - t_pq,
        len(df),
        len(df.columns),
    )
    time_col, event_col = detect_time_event(df)
    id_col = get_id_column(df)
    log.info(
        "[%s] Step: survival columns — time=%r event=%r id=%r",
        label,
        time_col,
        event_col,
        id_col,
    )
    ids_parquet = df[id_col].astype(str).to_numpy()
    if ids_parquet.shape[0] != iid_npz.shape[0] or not np.all(ids_parquet == iid_npz):
        n_show = min(5, len(ids_parquet))
        log.error(
            "[%s] ID mismatch: parquet vs NPZ iid (showing first %d): parquet=%s npz=%s",
            label,
            n_show,
            ids_parquet[:n_show].tolist(),
            iid_npz[:n_show].tolist(),
        )
        raise ValueError(
            "Parquet row order / IDs must match NPZ `iid` (merge_parquet_with_snp convention)."
        )
    log.info("[%s] Step: verified parquet IDs match NPZ iid (all %d rows)", label, len(ids_parquet))

    time_arr = df[time_col].to_numpy(dtype=np.float64)
    event_arr = df[event_col].to_numpy(dtype=np.float64)
    log.debug(
        "[%s] time stats: min=%.4f max=%.4f mean=%.4f | event rate=%.4f",
        label,
        float(np.min(time_arr)),
        float(np.max(time_arr)),
        float(np.mean(time_arr)),
        float(np.mean(event_arr)),
    )

    n_cpg_use = X_meth.shape[1] if max_cpg is None else min(int(max_cpg), X_meth.shape[1])
    n_snp_use = X_snp.shape[1] if max_snp is None else min(int(max_snp), X_snp.shape[1])
    if max_cpg is not None or max_snp is not None:
        log.info(
            "[%s] Step: column subset — using n_cpg=%d (of %d) n_snp=%d (of %d)",
            label,
            n_cpg_use,
            X_meth.shape[1],
            n_snp_use,
            X_snp.shape[1],
        )

    # Slice columns (creates a view for mmap; training still touches fewer cols)
    X_meth = X_meth[:, :n_cpg_use]
    X_snp = X_snp[:, :n_snp_use]

    log.info("[%s] Step: load complete — training tensor views ready", label)
    return X_meth, X_snp, time_arr, event_arr, n_cpg_use, n_snp_use


def stratified_train_val_split(
    event: np.ndarray, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(event))
    pos = idx[event > 0.5]
    neg = idx[event <= 0.5]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, int(round(val_frac * len(pos)))) if len(pos) else 0
    n_val_neg = max(1, int(round(val_frac * len(neg)))) if len(neg) else 0
    val_idx = np.concatenate([pos[:n_val_pos], neg[:n_val_neg]])
    train_mask = np.ones(len(event), dtype=bool)
    train_mask[val_idx] = False
    train_idx = idx[train_mask]
    val_idx = np.sort(val_idx)
    train_idx = np.sort(train_idx)
    return train_idx, val_idx


def _bundle_cache_path(
    cache_root: Optional[Path],
    label: str,
    combined_npz: Path,
    meta_parquet: Path,
    max_cpg: Optional[int],
    max_snp: Optional[int],
) -> Optional[Path]:
    if cache_root is None:
        return None
    cache_root = cache_root / "bundles"
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{combined_npz.stem}_{meta_parquet.stem}_cpg{max_cpg or 'all'}_snp{max_snp or 'all'}"
    return cache_root / f"{stem}.npz"


def load_bundle_with_cache(
    label: str,
    combined_npz: Path,
    meta_parquet: Path,
    snp_columns_txt: Optional[Path],
    max_cpg: Optional[int],
    max_snp: Optional[int],
    cache_root: Optional[Path],
    use_cache: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Wrapper over load_combined_bundle that optionally caches the aligned arrays
    (X_meth, X_snp, time, event, n_cpg, n_snp) to disk for much faster reloads.
    """
    log = logging.getLogger("vae_cox_lite.cache")
    cache_path = _bundle_cache_path(
        cache_root, label, combined_npz, meta_parquet, max_cpg, max_snp
    )
    if use_cache and cache_path is not None and cache_path.is_file():
        t0 = time.perf_counter()
        zc = np.load(cache_path, allow_pickle=False)
        X_meth = zc["X_meth"]
        X_snp = zc["X_snp"]
        time_arr = zc["time"]
        event_arr = zc["event"]
        n_cpg = int(zc["n_cpg"])
        n_snp = int(zc["n_snp"])
        log.info(
            "[%s] Cache hit: %s (loaded in %.3fs; X_meth=%s X_snp=%s)",
            label,
            cache_path,
            time.perf_counter() - t0,
            X_meth.shape,
            X_snp.shape,
        )
        return X_meth, X_snp, time_arr, event_arr, n_cpg, n_snp

    X_meth, X_snp, time_arr, event_arr, n_cpg, n_snp = load_combined_bundle(
        combined_npz,
        meta_parquet,
        snp_columns_txt,
        max_cpg,
        max_snp,
        label=label,
    )

    if use_cache and cache_path is not None:
        t0 = time.perf_counter()
        np.savez_compressed(
            cache_path,
            X_meth=X_meth,
            X_snp=X_snp,
            time=time_arr,
            event=event_arr,
            n_cpg=np.array(n_cpg, dtype=np.int64),
            n_snp=np.array(n_snp, dtype=np.int64),
        )
        log.info(
            "[%s] Cache write: %s (%.3fs; X_meth=%s X_snp=%s)",
            label,
            cache_path,
            time.perf_counter() - t0,
            X_meth.shape,
            X_snp.shape,
        )

    return X_meth, X_snp, time_arr, event_arr, n_cpg, n_snp


def _get_truncated_feature_names(
    combined_npz: Path,
    snp_columns_txt: Optional[Path],
    n_cpg_use: int,
    n_snp_use: int,
) -> Tuple[List[str], List[str]]:
    """
    Read CpG/SNP name txts next to a *_combined_training.npz and truncate
    to match the number of columns actually loaded (after any max_cpg/max_snp).
    """
    cpg_txt, snp_txt = paths_next_to_combined_training(combined_npz, snp_columns_txt)
    meth_names_full = read_column_txt(cpg_txt)
    snp_names_full = read_column_txt(snp_txt)
    meth_names = meth_names_full[:n_cpg_use]
    snp_names = snp_names_full[:n_snp_use]
    return meth_names, snp_names


def _align_common_features(
    X_fhs_meth: np.ndarray,
    X_fhs_snp: np.ndarray,
    meth_fhs_names: List[str],
    snp_fhs_names: List[str],
    X_whi_meth: np.ndarray,
    X_whi_snp: np.ndarray,
    meth_whi_names: List[str],
    snp_whi_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Intersect CpG and SNP feature sets between FHS and WHI and return
    aligned matrices in the *FHS feature order*.
    """
    log = logging.getLogger("vae_cox_lite.align")
    whi_meth_set = set(meth_whi_names)
    whi_snp_set = set(snp_whi_names)
    common_meth = [n for n in meth_fhs_names if n in whi_meth_set]
    common_snp = [n for n in snp_fhs_names if n in whi_snp_set]
    if not common_meth and not common_snp:
        raise ValueError("No common CpG or SNP features between FHS and WHI.")

    log.info(
        "Step: feature intersection — common CpGs=%d / FHS %d / WHI %d, "
        "common SNPs=%d / FHS %d / WHI %d",
        len(common_meth),
        len(meth_fhs_names),
        len(meth_whi_names),
        len(common_snp),
        len(snp_fhs_names),
        len(snp_whi_names),
    )

    fhs_meth_idx = {n: i for i, n in enumerate(meth_fhs_names)}
    whi_meth_idx = {n: i for i, n in enumerate(meth_whi_names)}
    fhs_snp_idx = {n: i for i, n in enumerate(snp_fhs_names)}
    whi_snp_idx = {n: i for i, n in enumerate(snp_whi_names)}

    fhs_m_cols = [fhs_meth_idx[n] for n in common_meth]
    whi_m_cols = [whi_meth_idx[n] for n in common_meth]
    fhs_s_cols = [fhs_snp_idx[n] for n in common_snp]
    whi_s_cols = [whi_snp_idx[n] for n in common_snp]

    X_fhs_meth_aligned = X_fhs_meth[:, fhs_m_cols] if common_meth else np.zeros(
        (X_fhs_meth.shape[0], 0), dtype=np.float32
    )
    X_whi_meth_aligned = X_whi_meth[:, whi_m_cols] if common_meth else np.zeros(
        (X_whi_meth.shape[0], 0), dtype=np.float32
    )
    X_fhs_snp_aligned = X_fhs_snp[:, fhs_s_cols] if common_snp else np.zeros(
        (X_fhs_snp.shape[0], 0), dtype=np.float32
    )
    X_whi_snp_aligned = X_whi_snp[:, whi_s_cols] if common_snp else np.zeros(
        (X_whi_snp.shape[0], 0), dtype=np.float32
    )

    n_cpg = X_fhs_meth_aligned.shape[1]
    n_snp = X_fhs_snp_aligned.shape[1]
    log.info(
        "Step: aligned feature dims — n_cpg=%d n_snp=%d (FHS rows=%d WHI rows=%d)",
        n_cpg,
        n_snp,
        X_fhs_meth.shape[0],
        X_whi_meth.shape[0],
    )
    return (
        X_fhs_meth_aligned,
        X_fhs_snp_aligned,
        X_whi_meth_aligned,
        X_whi_snp_aligned,
        n_cpg,
        n_snp,
    )


def _non_bias_l2_sum(model: nn.Module, device: torch.device) -> torch.Tensor:
    l2 = torch.tensor(0.0, device=device)
    for n, p in model.named_parameters():
        if p.requires_grad and "bias" not in n:
            l2 = l2 + (p**2).sum()
    return l2


@torch.no_grad()
def eval_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    beta: float,
    gamma_kl: float,
    lambda_l2: float,
    free_bits: float,
    use_amp: bool,
    split_name: str = "val",
    log_every_batch: bool = False,
) -> Tuple[float, dict, float, float]:
    log = logging.getLogger("vae_cox_lite.eval")
    model.eval()
    l2_once = _non_bias_l2_sum(model, device)
    sum_recon = 0.0
    sum_kl = 0.0
    n_samples = 0
    log_h_chunks: List[torch.Tensor] = []
    t_chunks: List[torch.Tensor] = []
    e_chunks: List[torch.Tensor] = []
    risks: List[float] = []
    times_l: List[float] = []
    events_l: List[float] = []
    n_steps = len(loader)
    log.info(
        "[%s] Step: evaluation pass — device=%s amp=%s batches=%d",
        split_name,
        device,
        use_amp and device.type == "cuda",
        n_steps,
    )
    t0 = time.perf_counter()
    for batch_idx, (xm, xs, t, e) in enumerate(loader):
        xm = xm.to(device, non_blocking=True)
        xs = xs.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)
        bs = int(xm.shape[0])
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            recon, target, log_h, mu, logvar = model(xm, xs)
        recon_loss = F.smooth_l1_loss(recon, target)
        kl = kl_gaussian(mu, logvar, free_bits=free_bits)
        sum_recon += float(recon_loss.detach().float().cpu()) * bs
        sum_kl += float(kl.detach().float().cpu()) * bs
        n_samples += bs
        log_h_chunks.append(log_h.detach().float())
        t_chunks.append(t.detach().float())
        e_chunks.append(e.detach().float())
        risks.extend(log_h.detach().float().cpu().numpy().tolist())
        times_l.extend(t.detach().cpu().numpy().tolist())
        events_l.extend(e.detach().cpu().numpy().tolist())
        if log_every_batch:
            cox_b = cox_ph_loss(log_h, t, e)
            loss_b = (
                alpha * recon_loss
                + beta * cox_b
                + gamma_kl * kl
                + lambda_l2 * l2_once
            )
            log.info(
                "[%s] batch %d/%d loss=%.6f recon=%.6f cox=%.6f kl=%.6f batch_n=%d events=%d",
                split_name,
                batch_idx + 1,
                n_steps,
                float(loss_b.detach().cpu()),
                float(recon_loss.detach().cpu()),
                float(cox_b.detach().cpu()),
                float(kl.detach().cpu()),
                bs,
                int(e.sum().item()),
            )
    if n_samples == 0:
        tot = 0.0
        agg = {"recon": 0.0, "cox": 0.0, "kl": 0.0}
    else:
        mean_recon = sum_recon / n_samples
        mean_kl = sum_kl / n_samples
        log_h_all = torch.cat(log_h_chunks, dim=0)
        t_all = torch.cat(t_chunks, dim=0)
        e_all = torch.cat(e_chunks, dim=0)
        cox_full = cox_ph_loss(log_h_all, t_all, e_all)
        cox_f = float(cox_full.detach().cpu())
        tot = (
            alpha * mean_recon
            + beta * cox_f
            + gamma_kl * mean_kl
            + lambda_l2 * float(l2_once.detach().cpu())
        )
        agg = {"recon": mean_recon, "cox": cox_f, "kl": mean_kl}
    ci_clinical, ci_flip = concordance_index_safe(
        np.array(times_l), np.array(risks), np.array(events_l)
    )
    log.info(
        "[%s] Step: evaluation done in %.3fs — mean_loss=%.6f recon=%.6f cox=%.6f kl=%.6f "
        "(cox=full-split likelihood) C-index(risk)=%s C-index(flip_sign)=%s n_samples=%d",
        split_name,
        time.perf_counter() - t0,
        tot,
        agg["recon"],
        agg["cox"],
        agg["kl"],
        f"{ci_clinical:.4f}" if not np.isnan(ci_clinical) else "nan",
        f"{ci_flip:.4f}" if not np.isnan(ci_flip) else "nan",
        len(times_l),
    )
    return tot, agg, ci_clinical, ci_flip


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE+Cox lite (AESURV-style low-rank, efficient I/O)")
    p.add_argument(
        "--combined-npz",
        type=str,
        default=str(_REPO_ROOT / "FHS_methylation_with_snp_1milfeatures_combined_training.npz"),
        help="*_combined_training.npz (default: FHS 1M merge in repo root)",
    )
    p.add_argument(
        "--meta-parquet",
        type=str,
        default=str(_REPO_ROOT / "FHS_methylation_with_snp_1milfeatures.parquet"),
        help="Parquet aligned with NPZ iid (default: FHS 1M merged parquet in repo root)",
    )
    p.add_argument("--snp-columns-txt", type=str, default=None, help="Override SNP name file path")
    p.add_argument("--max-cpg", type=int, default=None, help="Use first N CpG columns only (debug speed)")
    p.add_argument("--max-snp", type=int, default=None, help="Use first N SNP columns only (debug speed)")
    p.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (stratified by event)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (L2 on weights)")
    p.add_argument("--lambda-l2", type=float, default=0.0, help="Extra manual L2 in loss (joint_loss)")
    p.add_argument("--meth-rank", type=int, default=128)
    p.add_argument("--snp-rank", type=int, default=64)
    p.add_argument("--branch-out", type=int, default=256)
    p.add_argument("--fused-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--alpha", type=float, default=1.0, help="Reconstruction weight")
    p.add_argument("--beta", type=float, default=5.0, help="Cox weight")
    p.add_argument("--gamma-kl", type=float, default=0.1, help="KL weight")
    p.add_argument("--free-bits", type=float, default=0.0, help="Per-dim KL floor (nats)")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hazard-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast")
    p.add_argument("--balance-events", action="store_true", help="WeightedRandomSampler favors events")
    p.add_argument("--compile", action="store_true", help="torch.compile(model) (PyTorch 2+)")
    p.add_argument(
        "--save",
        type=str,
        default=str(_REPO_ROOT / "vae_cox_lite_checkpoint.pt"),
        help="Path for state_dict checkpoint (default: vae_cox_lite_checkpoint.pt in repo root)",
    )
    p.add_argument(
        "--test-combined-npz",
        type=str,
        default=None,
        help="Optional second cohort NPZ (e.g. WHI) for external C-index only",
    )
    p.add_argument(
        "--test-meta-parquet",
        type=str,
        default=None,
        help="Parquet for test cohort (must align with test NPZ iid)",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=str(_REPO_ROOT / "vae_cox_lite_train.log"),
        help="Append UTF-8 log file (default: vae_cox_lite_train.log in repo root)",
    )
    p.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable file logging (console only)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Console log level (default: INFO)",
    )
    p.add_argument(
        "--log-file-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING"],
        help="File log level (default: DEBUG — full detail in file)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Console DEBUG (same as --log-level DEBUG)",
    )
    p.add_argument(
        "--log-every-train-batch",
        action="store_true",
        help="Log every training batch (loss + grad norm) to console and file",
    )
    p.add_argument(
        "--log-every-val-batch",
        action="store_true",
        help="Log every validation / eval batch",
    )
    p.add_argument(
        "--log-batch-interval",
        type=int,
        default=0,
        help="If >0, log a DEBUG line every N training batches (running mean loss)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Epoch patience for early stopping on validation metric (0 = disable)",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Required improvement in monitored metric to reset patience.",
    )
    p.add_argument(
        "--early-stop-metric",
        type=str,
        default="val_ci",
        choices=["val_ci", "val_loss"],
        help="Metric for early stopping: val_ci (maximize) or val_loss (minimize).",
    )
    p.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=10,
        help="Do not trigger early stopping before this many epochs.",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(_REPO_ROOT / "vae_cox_cache"),
        help="Directory for cached aligned tensors (FHS/WHI); speeds up subsequent runs.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk caching of aligned tensors.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    console_level = (
        logging.DEBUG if args.verbose else _parse_log_level(args.log_level)
    )
    file_level = _parse_log_level(args.log_file_level)
    log_file_path: Optional[Path] = None
    if not args.no_log_file:
        log_file_path = _resolve_data_path(args.log_file)
    setup_run_logging(
        log_file=log_file_path,
        console_level=console_level,
        file_level=file_level,
    )
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log = logging.getLogger("vae_cox_lite.train")
    wall0 = time.perf_counter()
    log.info(
        "========== VAE+Cox lite run start — %s ==========",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )
    log.info(
        "Logging: console=%s file=%s file_level=%s",
        logging.getLevelName(console_level),
        log_file_path.resolve() if log_file_path else "(disabled)",
        logging.getLevelName(file_level),
    )
    log.debug("Full CLI args: %r", vars(args))

    set_seed(args.seed)
    log.info("Step: set_seed(%d)", args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp
    if device.type == "cuda":
        log.info(
            "Step: device=CUDA name=%s capability=%s amp=%s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
            use_amp,
        )
    else:
        log.info("Step: device=CPU (CUDA not available) amp=%s", use_amp)

    snp_txt = _resolve_data_path(args.snp_columns_txt) if args.snp_columns_txt else None
    if snp_txt is not None:
        log.info("Step: SNP columns override: %s", snp_txt.resolve())

    cache_root = None if args.no_cache else _resolve_data_path(args.cache_dir)
    if cache_root is not None:
        log.info("Step: cache root: %s", cache_root.resolve())

    fhs_combined_path = _resolve_data_path(args.combined_npz)
    fhs_meta_path = _resolve_data_path(args.meta_parquet)
    X_meth, X_snp, time_arr, event, n_cpg, n_snp = load_bundle_with_cache(
        label="FHS",
        combined_npz=fhs_combined_path,
        meta_parquet=fhs_meta_path,
        snp_columns_txt=snp_txt,
        max_cpg=args.max_cpg,
        max_snp=args.max_snp,
        cache_root=cache_root,
        use_cache=not args.no_cache,
    )
    log.info(
        "Step: training tensor summary — N=%d meth_dim=%d snp_dim=%d events=%d/%d (%.2f%%)",
        X_meth.shape[0],
        n_cpg,
        n_snp,
        int(event.sum()),
        len(event),
        100.0 * float(event.mean()),
    )

    # Optionally pre-load WHI at startup and align common features so that
    # training and test always share the same CpG/SNP feature space.
    whi_X_meth = whi_X_snp = whi_time = whi_event = None  # type: ignore[assignment]
    whi_n_cpg = whi_n_snp = None  # type: ignore[assignment]
    if args.test_combined_npz and args.test_meta_parquet:
        log.info("Step: pre-loading WHI bundle for common feature alignment")
        whi_combined_path = _resolve_data_path(args.test_combined_npz)
        whi_meta_path = _resolve_data_path(args.test_meta_parquet)
        X_meth_whi_raw, X_snp_whi_raw, time_whi, event_whi, n_cpg_whi, n_snp_whi = load_bundle_with_cache(
            label="WHI_raw",
            combined_npz=whi_combined_path,
            meta_parquet=whi_meta_path,
            snp_columns_txt=None,
            max_cpg=args.max_cpg,
            max_snp=args.max_snp,
            cache_root=cache_root,
            use_cache=not args.no_cache,
        )
        # Read truncated feature name lists for both cohorts and align.
        meth_fhs_names, snp_fhs_names = _get_truncated_feature_names(
            fhs_combined_path, snp_txt, n_cpg, n_snp
        )
        meth_whi_names, snp_whi_names = _get_truncated_feature_names(
            whi_combined_path, None, n_cpg_whi, n_snp_whi
        )
        (
            X_meth_aligned_fhs,
            X_snp_aligned_fhs,
            X_meth_aligned_whi,
            X_snp_aligned_whi,
            n_cpg_common,
            n_snp_common,
        ) = _align_common_features(
            X_meth,
            X_snp,
            meth_fhs_names,
            snp_fhs_names,
            X_meth_whi_raw,
            X_snp_whi_raw,
            meth_whi_names,
            snp_whi_names,
        )
        X_meth = X_meth_aligned_fhs
        X_snp = X_snp_aligned_fhs
        whi_X_meth = X_meth_aligned_whi
        whi_X_snp = X_snp_aligned_whi
        whi_time = time_whi
        whi_event = event_whi
        n_cpg = whi_n_cpg = n_cpg_common
        n_snp = whi_n_snp = n_snp_common
        log.info(
            "Step: post-alignment training summary — N=%d meth_dim=%d snp_dim=%d events=%d/%d (%.2f%%)",
            X_meth.shape[0],
            n_cpg,
            n_snp,
            int(event.sum()),
            len(event),
            100.0 * float(event.mean()),
        )

    train_idx, val_idx = stratified_train_val_split(event, args.val_frac, args.seed)
    n_pos = int((event[train_idx] > 0.5).sum())
    n_neg = int(len(train_idx) - n_pos)
    log.info(
        "Step: stratified split — train_n=%d (events+=%d censored=%d) val_n=%d val_frac=%.4f",
        len(train_idx),
        n_pos,
        n_neg,
        len(val_idx),
        args.val_frac,
    )

    train_ds = SurvivalNPZDataset(X_meth, X_snp, time_arr, event, train_idx)
    val_ds = SurvivalNPZDataset(X_meth, X_snp, time_arr, event, val_idx)

    sampler = None
    if args.balance_events:
        w = np.where(event[train_idx] > 0.5, 2.0, 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(
            torch.from_numpy(w), num_samples=len(train_idx), replacement=True
        )
        log.info("Step: using WeightedRandomSampler for event balance on train")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw)
    log.info(
        "Step: DataLoaders — train_batches=%d val_batches=%d batch_size=%d num_workers=%d "
        "pin_memory=%s prefetch=%s",
        len(train_loader),
        len(val_loader),
        args.batch_size,
        args.num_workers,
        loader_kw["pin_memory"],
        loader_kw.get("prefetch_factor", "n/a"),
    )

    log.info(
        "Step: building VAECoxLiteFusion — meth_rank=%d snp_rank=%d branch_out=%d "
        "fused_dim=%d latent_dim=%d dropout=%.4f",
        args.meth_rank,
        args.snp_rank,
        args.branch_out,
        args.fused_dim,
        args.latent_dim,
        args.dropout,
    )
    model = VAECoxLiteFusion(
        meth_dim=n_cpg,
        snp_dim=n_snp,
        meth_rank=args.meth_rank,
        snp_rank=args.snp_rank,
        branch_out=args.branch_out,
        fused_dim=args.fused_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        hazard_dropout=args.hazard_dropout,
    ).to(device)
    n_params = count_trainable_params(model)
    log.info("Step: model trainable parameters = %d (%.2e)", n_params, float(n_params))

    if args.compile and hasattr(torch, "compile"):
        log.info("Step: torch.compile(model) enabled")
        model = torch.compile(model)  # type: ignore[assignment]

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    log.info(
        "Step: optimizer AdamW lr=%g weight_decay=%g loss_weights alpha=%g beta=%g gamma_kl=%g "
        "lambda_l2=%g free_bits=%g",
        args.lr,
        args.weight_decay,
        args.alpha,
        args.beta,
        args.gamma_kl,
        args.lambda_l2,
        args.free_bits,
    )

    best_val_ci = -1.0
    best_state = None
    # For generic early stopping we track both C-index and loss view of progress.
    best_val_loss = float("inf")
    patience_counter = 0
    log.info(
        "Step: starting training loop — epochs=%d early_stop_patience=%d metric=%s min_delta=%.3g min_epochs=%d",
        args.epochs,
        args.early_stop_patience,
        args.early_stop_metric,
        args.early_stop_min_delta,
        args.early_stop_min_epochs,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_b = 0
        t_ep = time.perf_counter()
        log.info("--- Epoch %d / %d start ---", epoch, args.epochs)
        for batch_idx, (xm, xs, t, e) in enumerate(train_loader):
            xm = xm.to(device, non_blocking=True)
            xs = xs.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                recon, target, log_h, mu, logvar = model(xm, xs)
                loss, parts = joint_loss(
                    model,
                    recon,
                    target,
                    log_h,
                    mu,
                    logvar,
                    t,
                    e,
                    args.alpha,
                    args.beta,
                    args.gamma_kl,
                    args.lambda_l2,
                    free_bits=args.free_bits,
                )
            loss.backward()
            sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().pow(2).sum().cpu())
            grad_norm_pre_clip = sq**0.5
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            loss_f = float(loss.detach().cpu())
            running += loss_f
            n_b += 1
            if args.log_every_train_batch:
                log.info(
                    "epoch %d train batch %d/%d loss=%.6f recon=%.6f cox=%.6f kl=%.6f "
                    "grad_norm_pre_clip=%.6f batch_n=%d events_in_batch=%d",
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    loss_f,
                    float(parts["recon"].cpu()),
                    float(parts["cox"].cpu()),
                    float(parts["kl"].cpu()),
                    grad_norm_pre_clip,
                    int(xm.shape[0]),
                    int(e.sum().item()),
                )
            elif args.log_batch_interval > 0 and (batch_idx + 1) % args.log_batch_interval == 0:
                log.debug(
                    "epoch %d train batch %d/%d running_mean_loss=%.6f",
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    running / n_b,
                )

        train_loss = running / max(n_b, 1)
        log.info(
            "Epoch %d: train phase done in %.3fs mean_train_loss=%.6f batches=%d",
            epoch,
            time.perf_counter() - t_ep,
            train_loss,
            n_b,
        )

        val_loss, val_parts, ci_clinical, ci_flip = eval_split(
            model,
            val_loader,
            device,
            args.alpha,
            args.beta,
            args.gamma_kl,
            args.lambda_l2,
            args.free_bits,
            use_amp,
            split_name=f"epoch{epoch}/val",
            log_every_batch=args.log_every_val_batch,
        )
        ci_report = (
            ci_clinical if not np.isnan(ci_clinical) else ci_flip
        )
        if not np.isnan(ci_clinical) and not np.isnan(ci_flip):
            if ci_flip > ci_clinical + 0.02:
                log.warning(
                    "C-index(flip_sign)=%.4f > C-index(risk)=%.4f — check Cox target/censoring "
                    "coding or time direction.",
                    ci_flip,
                    ci_clinical,
                )
        log.info(
            "Epoch %03d summary: train_loss=%.6f val_loss=%.6f val_recon=%.6f val_cox=%.6f "
            "val_kl=%.6f val_C-index=%s",
            epoch,
            train_loss,
            val_loss,
            val_parts["recon"],
            val_parts["cox"],
            val_parts["kl"],
            f"{ci_report:.4f}" if not np.isnan(ci_report) else "nan",
        )
        improved = False
        # Update bests depending on monitored metric
        if not np.isnan(ci_clinical) and args.early_stop_metric == "val_ci":
            if ci_clinical > best_val_ci + args.early_stop_min_delta:
                improved = True
                best_val_ci = ci_clinical
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log.info(
                    "Epoch %d: new best val C-index(risk)=%.6f (checkpoint state cached)",
                    epoch,
                    best_val_ci,
                )
        elif args.early_stop_metric == "val_loss":
            if val_loss < best_val_loss - args.early_stop_min_delta:
                improved = True
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log.info(
                    "Epoch %d: new best val_loss=%.6f (checkpoint state cached)",
                    epoch,
                    best_val_loss,
                )

        if args.early_stop_patience > 0 and args.early_stop_metric in {"val_ci", "val_loss"}:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                log.info(
                    "Epoch %d: no improvement on %s (patience %d / %d)",
                    epoch,
                    args.early_stop_metric,
                    patience_counter,
                    args.early_stop_patience,
                )
            if (
                epoch >= args.early_stop_min_epochs
                and patience_counter >= args.early_stop_patience
            ):
                log.info(
                    "Early stopping triggered at epoch %d after %d epochs without improvement.",
                    epoch,
                    patience_counter,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if args.early_stop_metric == "val_loss":
            log.info("Step: restored best weights — best val_loss=%.6f", best_val_loss)
        else:
            log.info("Step: restored best weights — val C-index(risk)=%.6f", best_val_ci)

    if args.save:
        path = _resolve_data_path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "meth_dim": n_cpg,
                "snp_dim": n_snp,
                "meth_rank": args.meth_rank,
                "snp_rank": args.snp_rank,
                "branch_out": args.branch_out,
                "fused_dim": args.fused_dim,
                "latent_dim": args.latent_dim,
                "args": vars(args),
            },
            path,
        )
        log.info("Step: saved checkpoint — %s", path.resolve())

    if args.test_combined_npz and args.test_meta_parquet and whi_X_meth is not None:
        log.info("Step: external test cohort evaluation (WHI, aligned common features)")
        test_idx = np.arange(len(whi_time))
        test_ds = SurvivalNPZDataset(whi_X_meth, whi_X_snp, whi_time, whi_event, test_idx)
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kw)
        _, _, ci_ext, ci_ext_flip = eval_split(
            model,
            test_loader,
            device,
            args.alpha,
            args.beta,
            args.gamma_kl,
            args.lambda_l2,
            args.free_bits,
            use_amp,
            split_name="external/WHI",
            log_every_batch=args.log_every_val_batch,
        )
        log.info(
            "Step: external test finished — C-index(risk)=%s C-index(flip_sign)=%s n=%d",
            f"{ci_ext:.4f}" if not np.isnan(ci_ext) else "nan",
            f"{ci_ext_flip:.4f}" if not np.isnan(ci_ext_flip) else "nan",
            len(test_idx),
        )

    log.info(
        "========== Run finished in %.1f s wall time ==========",
        time.perf_counter() - wall0,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
