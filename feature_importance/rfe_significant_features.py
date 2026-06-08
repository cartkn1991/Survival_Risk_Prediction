#!/usr/bin/env python3
"""Backward / forward RFE on risk-significant features (frozen DANN-Aux masking).

Inactive features → DANN scaler mean. Forward selection (recommended) adds
top-|grad| significant features until FHS val C-index matches the full model;
optional binary refinement finds the exact minimal count.

Outputs (``--out-dir``, default ``feature_importance/rfe/``):
  Backward: rfe_steps.csv, rfe_minimal_features.csv, rfe_curve.pdf
  Forward:  rfe_forward_steps.csv, rfe_forward_minimal_features.csv, rfe_forward_curve.pdf
  Combined: rfe_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dann_aesurv_bridge import DannLatentEncoder, InvariantPreprocessor, load_mini_dann_for_fusion
from train_aesurv_dann_latent_aux import AESurvHeadAux
from train_dann_survival import harrell_c_index
from train_vae_cox_lite import (
    _align_common_features,
    _get_truncated_feature_names,
    load_bundle_with_cache,
)

log = logging.getLogger("rfe_significant")


def _torch_load_compat(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_feature_names(fhs_npz: Path, n_cpg: int, n_snp: int, snp_txt: Optional[Path]) -> Tuple[List[str], List[str]]:
    meth, snp = _get_truncated_feature_names(fhs_npz, snp_txt, n_cpg, n_snp)
    return list(meth), list(snp)


def build_name_index(meth_names: List[str], snp_names: List[str]) -> Dict[str, int]:
    idx = {str(n): i for i, n in enumerate(meth_names)}
    n_cpg = len(meth_names)
    for j, n in enumerate(snp_names):
        idx[str(n)] = n_cpg + j
    return idx


def load_significant_indices(
    sig_csv: Path,
    name_to_idx: Dict[str, int],
    n_d_in: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (sig_indices, removal_order, addition_order, sig_df)."""
    df = pd.read_csv(sig_csv, low_memory=False)
    if "feature" not in df.columns:
        raise ValueError(f"sig csv missing 'feature' column: {sig_csv}")
    missing = []
    rows = []
    for _, row in df.iterrows():
        name = str(row["feature"])
        if name not in name_to_idx:
            missing.append(name)
            continue
        rows.append((name_to_idx[name], row))
    if missing:
        log.warning("  %d significant features not in aligned model (%d mapped)", len(missing), len(rows))
    if not rows:
        raise SystemExit("No significant features mapped to model indices.")
    sig_idx = np.array([r[0] for r in rows], dtype=np.int64)
    # removal order: smallest |mean_grad| first (weakest contributors)
    sort_key = []
    for i, row in rows:
        g = float(row.get("abs_mean_grad", abs(float(row.get("mean_grad", 0)))))
        sort_key.append((g, i))
    sort_key.sort(key=lambda t: t[0])
    removal_order = np.array([t[1] for t in sort_key], dtype=np.int64)
    mapped = pd.DataFrame([
        {
            "feature": r[1]["feature"],
            "kind": r[1].get("kind", ""),
            "abs_mean_grad": float(r[1].get("abs_mean_grad", np.nan)),
            "mean_grad": float(r[1].get("mean_grad", np.nan)),
            "col_idx": int(r[0]),
        }
        for r in rows
    ])
    mapped = mapped.sort_values("abs_mean_grad", ascending=True).reset_index(drop=True)
    addition_order = removal_order[::-1].copy()
    return sig_idx, removal_order, addition_order, mapped


def build_forward_mask(
    d_in: int,
    sig_idx: np.ndarray,
    addition_order: np.ndarray,
    n_active_sig: int,
) -> np.ndarray:
    """Non-significant features ON; top ``n_active_sig`` from addition_order ON; rest of sig pool OFF."""
    active = np.ones(d_in, dtype=bool)
    sig_set = set(int(i) for i in sig_idx)
    for i in sig_set:
        active[i] = False
    n_on = min(int(n_active_sig), len(addition_order))
    for i in addition_order[:n_on]:
        active[int(i)] = True
    return active


def evaluate_splits(
    model: FrozenRiskModel,
    X_meth_va: np.ndarray,
    X_snp_va: np.ndarray,
    t_va: np.ndarray,
    e_va: np.ndarray,
    X_meth_te: np.ndarray,
    X_snp_te: np.ndarray,
    t_te: np.ndarray,
    e_te: np.ndarray,
    active_mask: np.ndarray,
    n_cpg: int,
) -> Tuple[float, float]:
    c_va = evaluate_mask(model, X_meth_va, X_snp_va, t_va, e_va, active_mask, n_cpg)
    c_te = evaluate_mask(model, X_meth_te, X_snp_te, t_te, e_te, active_mask, n_cpg)
    return c_va, c_te


def _record_step(
    records: List[Dict],
    phase: str,
    step: int,
    n_active_sig: int,
    n_pool: int,
    c_va: float,
    c_te: float,
    cindex_baseline: float,
    cindex_tol: float,
) -> None:
    records.append({
        "phase": phase,
        "step": step,
        "n_active_sig": n_active_sig,
        "n_sig_pool": n_pool,
        "cindex_val": c_va,
        "cindex_whi": c_te,
        "meets_threshold": c_va >= cindex_baseline - cindex_tol,
    })


def apply_active_mask(
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    active_mask: np.ndarray,
    mean_full: np.ndarray,
    n_cpg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inactive columns → scaler mean (raw space)."""
    mmean = mean_full[:n_cpg].astype(np.float32)
    smean = mean_full[n_cpg:].astype(np.float32)
    Xm = np.asarray(X_meth, dtype=np.float32)
    Xs = np.asarray(X_snp, dtype=np.float32)
    am = active_mask[:n_cpg]
    a_s = active_mask[n_cpg:]
    if not np.all(am):
        Xm = Xm.copy()
        Xm[:, ~am] = mmean[~am]
    if not np.all(a_s):
        Xs = Xs.copy()
        Xs[:, ~a_s] = smean[~a_s]
    return Xm, Xs


class FrozenRiskModel:
    def __init__(self, bundle_dir: Path, preprocess_npz: Path, device: torch.device):
        self.device = device
        self.preproc = InvariantPreprocessor(preprocess_npz).to(device).eval()
        self.mean_full = self.preproc.mean_.detach().cpu().numpy().astype(np.float32)
        z = np.load(preprocess_npz, allow_pickle=False)
        self.meth_as_mvalues = bool(int(z["meth_as_mvalues"][0])) if "meth_as_mvalues" in z.files else True
        mini, _ = load_mini_dann_for_fusion(bundle_dir / "dann_encoder.pt", map_location=device)
        self.encoder = DannLatentEncoder(mini.to(device)).eval()
        head_ck = _torch_load_compat(bundle_dir / "aesurv_head.pt", map_location=device)
        cfg = head_ck["config"]
        self.head = AESurvHeadAux(
            in_dim=int(cfg["in_dim"]), enc_hidden=tuple(cfg["enc_hidden"]),
            dec_hidden=tuple(cfg["dec_hidden"]), z_dim=int(cfg["z_dim"]),
            cohort_hidden=int(cfg["cohort_hidden"]), dropout=float(cfg["dropout"]),
            n_cells=int(cfg["n_cells"]),
        ).to(device).eval()
        self.head.load_state_dict(head_ck["state_dict"])

    @torch.no_grad()
    def predict_log_h(
        self,
        X_meth: np.ndarray,
        X_snp: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        n = X_meth.shape[0]
        out = np.empty((n,), dtype=np.float32)
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            tm = torch.from_numpy(X_meth[s:e]).to(self.device)
            ts = torch.from_numpy(X_snp[s:e]).to(self.device)
            x_proj = self.preproc(tm, ts, self.meth_as_mvalues)
            mu = self.encoder(x_proj)
            _, _, log_h, _, _, _, _ = self.head(mu, sample_z=False)
            out[s:e] = log_h.squeeze(-1).cpu().numpy()
        return out


def evaluate_mask(
    model: FrozenRiskModel,
    X_meth: np.ndarray,
    X_snp: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    active_mask: np.ndarray,
    n_cpg: int,
) -> float:
    Xm, Xs = apply_active_mask(X_meth, X_snp, active_mask, model.mean_full, n_cpg)
    log_h = model.predict_log_h(Xm, Xs)
    return harrell_c_index(time, event, log_h)


def run_backward_rfe(
    model: FrozenRiskModel,
    X_meth_va: np.ndarray,
    X_snp_va: np.ndarray,
    t_va: np.ndarray,
    e_va: np.ndarray,
    X_meth_te: np.ndarray,
    X_snp_te: np.ndarray,
    t_te: np.ndarray,
    e_te: np.ndarray,
    n_cpg: int,
    d_in: int,
    sig_idx: np.ndarray,
    removal_order: np.ndarray,
    cindex_baseline: float,
    cindex_tol: float,
    step_size: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    active = np.ones(d_in, dtype=bool)
    sig_set: Set[int] = set(int(i) for i in sig_idx)
    remaining = [i for i in removal_order if i in sig_set]  # preserve order

    records = []
    # baseline full (all features)
    c_va = evaluate_mask(model, X_meth_va, X_snp_va, t_va, e_va, active, n_cpg)
    c_te = evaluate_mask(model, X_meth_te, X_snp_te, t_te, e_te, active, n_cpg)
    records.append({
        "step": 0, "n_active_total": int(active.sum()), "n_active_sig": len(sig_set),
        "n_removed_sig": 0, "cindex_val": c_va, "cindex_whi": c_te,
        "meets_threshold": c_va >= cindex_baseline - cindex_tol,
    })
    log.info("  step 0: n_active=%d  C_val=%.4f  C_whi=%.4f", active.sum(), c_va, c_te)

    step = 0
    while remaining:
        step += 1
        chunk = remaining[:step_size]
        remaining = remaining[len(chunk):]
        for idx in chunk:
            active[idx] = False
        c_va = evaluate_mask(model, X_meth_va, X_snp_va, t_va, e_va, active, n_cpg)
        c_te = evaluate_mask(model, X_meth_te, X_snp_te, t_te, e_te, active, n_cpg)
        n_sig_active = int(sum(1 for i in sig_set if active[i]))
        meets = c_va >= cindex_baseline - cindex_tol
        records.append({
            "step": step,
            "n_active_total": int(active.sum()),
            "n_active_sig": n_sig_active,
            "n_removed_sig": len(sig_set) - n_sig_active,
            "cindex_val": c_va,
            "cindex_whi": c_te,
            "meets_threshold": meets,
        })
        log.info(
            "  step %d: n_sig_active=%d  C_val=%.4f  meets=%s",
            step, n_sig_active, c_va, meets,
        )
        if not meets:
            # revert last chunk for minimal set = state before this step
            for idx in chunk:
                active[idx] = True
            break

    df_steps = pd.DataFrame(records)
    # minimal mask: last row that met threshold
    met = df_steps[df_steps["meets_threshold"]]
    if met.empty:
        minimal_mask = active.copy()
        log.warning("No subset met threshold; using full significant set.")
    else:
        last_met = met.iloc[-1]
        # reconstruct mask at that step
        n_remove = int(last_met["n_removed_sig"])
        minimal_mask = np.ones(d_in, dtype=bool)
        removed = set(removal_order[:n_remove])
        for idx in removed:
            if idx in sig_set:
                minimal_mask[idx] = False

    summary = {
        "cindex_baseline_full_val": float(cindex_baseline),
        "cindex_tol": float(cindex_tol),
        "n_significant_pool": int(len(sig_idx)),
        "n_minimal_sig_active": int(sum(1 for i in sig_set if minimal_mask[i])),
        "n_minimal_total_active": int(minimal_mask.sum()),
        "minimal_cindex_val": float(met.iloc[-1]["cindex_val"]) if not met.empty else float("nan"),
        "minimal_cindex_whi": float(met.iloc[-1]["cindex_whi"]) if not met.empty else float("nan"),
    }
    return df_steps, minimal_mask, summary


def run_forward_selection(
    model: FrozenRiskModel,
    X_meth_va: np.ndarray,
    X_snp_va: np.ndarray,
    t_va: np.ndarray,
    e_va: np.ndarray,
    X_meth_te: np.ndarray,
    X_snp_te: np.ndarray,
    t_te: np.ndarray,
    e_te: np.ndarray,
    n_cpg: int,
    d_in: int,
    sig_idx: np.ndarray,
    addition_order: np.ndarray,
    cindex_baseline: float,
    cindex_tol: float,
    step_size: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Coarse forward selection + binary refinement on the last bracket."""
    n_pool = len(sig_idx)
    records: List[Dict] = []
    step = 0
    n_active = 0
    mask = build_forward_mask(d_in, sig_idx, addition_order, 0)
    c_va, c_te = evaluate_splits(
        model, X_meth_va, X_snp_va, t_va, e_va, X_meth_te, X_snp_te, t_te, e_te, mask, n_cpg,
    )
    _record_step(records, "coarse", step, n_active, n_pool, c_va, c_te, cindex_baseline, cindex_tol)
    log.info("  forward coarse step %d: n_sig=%d  C_val=%.4f  meets=%s",
             step, n_active, c_va, c_va >= cindex_baseline - cindex_tol)

    n_lo_fail = 0
    n_hi_pass: Optional[int] = None
    remaining = n_pool

    while remaining > 0:
        chunk = min(step_size, remaining)
        step += 1
        n_active += chunk
        remaining -= chunk
        mask = build_forward_mask(d_in, sig_idx, addition_order, n_active)
        c_va, c_te = evaluate_splits(
            model, X_meth_va, X_snp_va, t_va, e_va, X_meth_te, X_snp_te, t_te, e_te, mask, n_cpg,
        )
        meets = c_va >= cindex_baseline - cindex_tol
        _record_step(records, "coarse", step, n_active, n_pool, c_va, c_te, cindex_baseline, cindex_tol)
        log.info("  forward coarse step %d: n_sig=%d  C_val=%.4f  meets=%s", step, n_active, c_va, meets)
        if meets:
            n_hi_pass = n_active
            n_lo_fail = n_active - chunk
            break
        n_lo_fail = n_active

    if n_hi_pass is None:
        log.warning("Forward selection: all %d significant features active but threshold not met.", n_pool)
        n_hi_pass = n_pool
        n_lo_fail = max(0, n_pool - step_size)

    # Binary refinement between last fail and first pass
    refine_step = 0
    lo, hi = int(n_lo_fail), int(n_hi_pass)
    best_n = hi
    best_c_va, best_c_te = evaluate_splits(
        model, X_meth_va, X_snp_va, t_va, e_va, X_meth_te, X_snp_te, t_te, e_te,
        build_forward_mask(d_in, sig_idx, addition_order, hi), n_cpg,
    )
    while hi - lo > 1:
        refine_step += 1
        mid = (lo + hi) // 2
        mask_mid = build_forward_mask(d_in, sig_idx, addition_order, mid)
        c_va, c_te = evaluate_splits(
            model, X_meth_va, X_snp_va, t_va, e_va, X_meth_te, X_snp_te, t_te, e_te, mask_mid, n_cpg,
        )
        meets = c_va >= cindex_baseline - cindex_tol
        _record_step(records, "refine", refine_step, mid, n_pool, c_va, c_te, cindex_baseline, cindex_tol)
        log.info("  forward refine %d: n_sig=%d  C_val=%.4f  meets=%s", refine_step, mid, c_va, meets)
        if meets:
            hi = mid
            best_n = mid
            best_c_va, best_c_te = c_va, c_te
        else:
            lo = mid

    minimal_mask = build_forward_mask(d_in, sig_idx, addition_order, best_n)
    df_steps = pd.DataFrame(records)
    summary = {
        "method": "forward_with_binary_refinement",
        "cindex_baseline_full_val": float(cindex_baseline),
        "cindex_tol": float(cindex_tol),
        "n_significant_pool": int(n_pool),
        "n_minimal_sig_active": int(best_n),
        "n_coarse_bracket_lo": int(n_lo_fail),
        "n_coarse_bracket_hi": int(n_hi_pass),
        "minimal_cindex_val": float(best_c_va),
        "minimal_cindex_whi": float(best_c_te),
        "n_refinement_steps": int(refine_step),
        "n_coarse_steps": int(step),
    }
    return df_steps, minimal_mask, summary


def save_minimal_feature_list(
    minimal_mask: np.ndarray,
    meth_names: List[str],
    snp_names: List[str],
    sig_df: pd.DataFrame,
    sig_idx: np.ndarray,
    out_csv: Path,
) -> None:
    """Write active features in the significant pool only (not all 1.37M columns)."""
    n_cpg = len(meth_names)
    rows = []
    for i in sig_idx:
        if not minimal_mask[int(i)]:
            continue
        if i < n_cpg:
            name, kind = meth_names[int(i)], "cpg"
        else:
            name, kind = snp_names[int(i) - n_cpg], "snp"
        rows.append({"feature": name, "kind": kind, "col_idx": int(i)})
    out = pd.DataFrame(rows)
    if not sig_df.empty and "feature" in sig_df.columns:
        gmap = sig_df.set_index("feature")[["abs_mean_grad", "mean_grad"]].to_dict("index")
        out["abs_mean_grad"] = out["feature"].map(lambda f: gmap.get(f, {}).get("abs_mean_grad", np.nan))
        out["mean_grad"] = out["feature"].map(lambda f: gmap.get(f, {}).get("mean_grad", np.nan))
    out.to_csv(out_csv, index=False)


def plot_curve(
    df_steps: pd.DataFrame,
    cindex_baseline: float,
    cindex_tol: float,
    out_path: Path,
    *,
    title: str = "RFE on risk-significant features (frozen DANN-Aux mask)",
    color: str = "#c0392b",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    coarse = df_steps[df_steps["phase"] == "coarse"] if "phase" in df_steps.columns else df_steps
    ax.plot(coarse["n_active_sig"], coarse["cindex_val"], "o-", label="Coarse (FHS val C)", color=color)
    if "phase" in df_steps.columns and (df_steps["phase"] == "refine").any():
        ref = df_steps[df_steps["phase"] == "refine"]
        ax.plot(ref["n_active_sig"], ref["cindex_val"], "s", label="Binary refine", color="#8e44ad", ms=6)
    ax.axhline(cindex_baseline, color="k", ls="--", lw=1, label=f"Full model baseline ({cindex_baseline:.3f})")
    ax.axhline(cindex_baseline - cindex_tol, color="gray", ls=":", lw=1,
               label=f"Threshold (baseline − {cindex_tol})")
    ax.set_xlabel("Number of active risk-significant features")
    ax.set_ylabel("Harrell C-index (FHS val)")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="RFE on risk-significant features (frozen model masking).")
    p.add_argument("--sig-csv", type=str, default="feature_importance/significance/significant_risk.csv")
    p.add_argument("--bundle-dir", type=str, default="models/aesurv_final")
    p.add_argument("--dann-preprocess-npz", type=str, default="runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz")
    p.add_argument("--fhs-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--fhs-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--whi-pq", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--snp-columns-txt", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="feature_importance/rfe")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="best",
                   choices=("backward", "forward", "best", "both"),
                   help="best=forward+binary refine (default); both=backward+forward.")
    p.add_argument("--step-size", type=int, default=500,
                   help="Features per coarse forward/backward step.")
    p.add_argument("--cindex-tol", type=float, default=0.005,
                   help="Max allowed drop from full-feature FHS val C-index.")
    p.add_argument("--baseline-cindex", type=float, default=None,
                   help="Override baseline C-index (else computed on val split).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    log.info("Loading FHS/WHI bundles...")
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_f, n_snp_f = load_bundle_with_cache(
        "FHS", Path(args.fhs_npz), Path(args.fhs_pq),
        Path(args.snp_columns_txt) if args.snp_columns_txt else None,
        None, None, Path("vae_cox_cache"), True,
    )
    X_meth_whi, X_snp_whi, t_whi, e_whi, n_cpg_w, n_snp_w = load_bundle_with_cache(
        "WHI_raw", Path(args.whi_npz), Path(args.whi_pq),
        None, None, None, Path("vae_cox_cache"), True,
    )
    snp_txt = Path(args.snp_columns_txt) if args.snp_columns_txt else None
    meth_f, snp_f = load_feature_names(Path(args.fhs_npz), n_cpg_f, n_snp_f, snp_txt)
    meth_w, snp_w = load_feature_names(Path(args.whi_npz), n_cpg_w, n_snp_w, None)
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, n_cpg, n_snp = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_f, snp_f,
        X_meth_whi, X_snp_whi, meth_w, snp_w,
    )
    d_in = n_cpg + n_snp
    meth_names, snp_names = meth_f[:n_cpg], snp_f[:n_snp]
    name_to_idx = build_name_index(meth_names, snp_names)

    preprocess = Path(args.dann_preprocess_npz)
    if not preprocess.exists():
        preprocess = Path(args.bundle_dir) / "dann_preprocess.npz"
    model = FrozenRiskModel(Path(args.bundle_dir), preprocess, device)

    # FHS train/val split (same as training)
    idx = np.arange(X_meth_fhs.shape[0])
    strat = e_fhs.astype(np.int32) if e_fhs.sum() >= 2 else None
    tr_idx, va_idx = train_test_split(
        idx, test_size=args.val_frac, random_state=args.seed, stratify=strat, shuffle=True,
    )
    log.info("FHS split: train=%d val=%d  WHI test=%d", len(tr_idx), len(va_idx), X_meth_whi.shape[0])

    X_meth_va, X_snp_va = X_meth_fhs[va_idx], X_snp_fhs[va_idx]
    t_va, e_va = t_fhs[va_idx], e_fhs[va_idx]

    active_full = np.ones(d_in, dtype=bool)
    if args.baseline_cindex is not None:
        c_baseline = float(args.baseline_cindex)
    else:
        c_baseline = evaluate_mask(model, X_meth_va, X_snp_va, t_va, e_va, active_full, n_cpg)
    log.info("Baseline full-feature FHS val C-index = %.4f", c_baseline)

    sig_idx, removal_order, addition_order, sig_df = load_significant_indices(
        Path(args.sig_csv), name_to_idx, d_in,
    )
    log.info("Mapped %d significant features for RFE pool", len(sig_idx))

    X_meth_te, X_snp_te = X_meth_whi, X_snp_whi
    combined_summary: Dict = {
        "cindex_baseline_full_val": float(c_baseline),
        "cindex_tol": float(args.cindex_tol),
        "n_significant_pool": int(len(sig_idx)),
        "mode": args.mode,
        "reference_metrics": str(Path("models/aesurv_final/run_metrics.json")),
    }

    if args.mode in ("backward", "both"):
        log.info("=== Backward RFE ===")
        df_bwd, mask_bwd, sum_bwd = run_backward_rfe(
            model, X_meth_va, X_snp_va, t_va, e_va,
            X_meth_te, X_snp_te, t_whi, e_whi,
            n_cpg, d_in, sig_idx, removal_order,
            c_baseline, args.cindex_tol, args.step_size,
        )
        sum_bwd["method"] = "backward"
        df_bwd.to_csv(out_dir / "rfe_steps.csv", index=False)
        save_minimal_feature_list(
            mask_bwd, meth_names, snp_names, sig_df, sig_idx, out_dir / "rfe_minimal_features.csv",
        )
        plot_curve(df_bwd, c_baseline, args.cindex_tol, out_dir / "rfe_curve.pdf",
                   title="Backward RFE (frozen DANN-Aux mask)")
        combined_summary["backward"] = sum_bwd

    if args.mode in ("forward", "best", "both"):
        log.info("=== Forward selection + binary refinement ===")
        df_fwd, mask_fwd, sum_fwd = run_forward_selection(
            model, X_meth_va, X_snp_va, t_va, e_va,
            X_meth_te, X_snp_te, t_whi, e_whi,
            n_cpg, d_in, sig_idx, addition_order,
            c_baseline, args.cindex_tol, args.step_size,
        )
        df_fwd.to_csv(out_dir / "rfe_forward_steps.csv", index=False)
        save_minimal_feature_list(
            mask_fwd, meth_names, snp_names, sig_df, sig_idx,
            out_dir / "rfe_forward_minimal_features.csv",
        )
        plot_curve(
            df_fwd, c_baseline, args.cindex_tol, out_dir / "rfe_forward_curve.pdf",
            title="Forward selection + binary refine (frozen DANN-Aux mask)",
            color="#27ae60",
        )
        sum_fwd["runtime_sec"] = time.time() - t0
        combined_summary["forward"] = sum_fwd
        combined_summary["recommended_method"] = "forward"
        (out_dir / "rfe_forward_summary.json").write_text(
            json.dumps(sum_fwd, indent=2), encoding="utf-8",
        )
        log.info("Forward minimal significant features: %d (C_val=%.4f)",
                 sum_fwd["n_minimal_sig_active"], sum_fwd["minimal_cindex_val"])

    combined_summary["runtime_sec"] = time.time() - t0
    (out_dir / "rfe_summary.json").write_text(
        json.dumps(combined_summary, indent=2), encoding="utf-8",
    )
    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
