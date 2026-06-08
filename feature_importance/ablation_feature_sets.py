#!/usr/bin/env python3
"""Ablation + random-null masking: prove risk-selected features carry prognostic signal.

Compares frozen AESurv C-index under different active feature sets:
  - full: all features
  - significant_only: only risk-selected SNPs+CpGs (rest → scaler mean)
  - nonsignificant_only: inverse mask
  - sig_cpg_only / sig_snp_only: modality subsets
  - clock_cpgs_only: union Horvath/Hannum/PhenoAge/GrimAgeV2 CpGs
  - random_null: matched-size random non-selected features (repeated)

Outputs: feature_importance/ablation/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_importance.rfe_significant_features import (
    FrozenRiskModel,
    build_name_index,
    evaluate_mask,
    load_feature_names,
    load_significant_indices,
)
from train_vae_cox_lite import _align_common_features, load_bundle_with_cache

log = logging.getLogger("ablation")


def _load_clock_cpg_indices(meth_names: List[str], clock_dir: Path) -> np.ndarray:
    clocks = ["clock_horvath.csv", "clock_hannum.csv", "clock_phenoage.csv", "clock_grimagev2.csv"]
    cpgs: Set[str] = set()
    for fn in clocks:
        p = clock_dir / fn
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "CpGmarker" if "CpGmarker" in df.columns else df.columns[0]
        cpgs.update(df[col].astype(str).str.strip())
    name_to_i = {str(n): i for i, n in enumerate(meth_names)}
    idx = [name_to_i[c] for c in cpgs if c in name_to_i]
    return np.asarray(sorted(set(idx)), dtype=np.int64)


def _mask_significant_only(d_in: int, sig_idx: np.ndarray) -> np.ndarray:
    active = np.zeros(d_in, dtype=bool)
    active[sig_idx] = True
    return active


def _mask_nonsignificant_only(d_in: int, sig_idx: np.ndarray) -> np.ndarray:
    active = np.ones(d_in, dtype=bool)
    active[sig_idx] = False
    return active


def _mask_modality_sig(d_in: int, n_cpg: int, sig_idx: np.ndarray, kind: str) -> np.ndarray:
    active = np.zeros(d_in, dtype=bool)
    for i in sig_idx:
        if kind == "cpg" and i < n_cpg:
            active[i] = True
        elif kind == "snp" and i >= n_cpg:
            active[i] = True
    return active


def _mask_clock_only(d_in: int, clock_idx: np.ndarray) -> np.ndarray:
    active = np.zeros(d_in, dtype=bool)
    active[clock_idx] = True
    return active


def _mask_random_null(
    d_in: int,
    n_cpg: int,
    sig_idx: np.ndarray,
    n_sig_cpg: int,
    n_sig_snp: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sig_set = set(int(i) for i in sig_idx)
    cpg_pool = np.array([i for i in range(n_cpg) if i not in sig_set], dtype=np.int64)
    snp_pool = np.array([i for i in range(n_cpg, d_in) if i not in sig_set], dtype=np.int64)
    active = np.zeros(d_in, dtype=bool)
    if len(cpg_pool) >= n_sig_cpg:
        active[rng.choice(cpg_pool, n_sig_cpg, replace=False)] = True
    if len(snp_pool) >= n_sig_snp:
        active[rng.choice(snp_pool, n_sig_snp, replace=False)] = True
    return active


def _eval_condition(
    model: FrozenRiskModel,
    name: str,
    mask: np.ndarray,
    X_meth_va, X_snp_va, t_va, e_va,
    X_meth_te, X_snp_te, t_te, e_te,
    n_cpg: int,
) -> Dict:
    c_va = evaluate_mask(model, X_meth_va, X_snp_va, t_va, e_va, mask, n_cpg)
    c_te = evaluate_mask(model, X_meth_te, X_snp_te, t_te, e_te, mask, n_cpg)
    return {
        "condition": name,
        "n_active": int(mask.sum()),
        "cindex_fhs_val": float(c_va),
        "cindex_whi": float(c_te),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sig-csv", default="feature_importance/significance/significant_risk.csv")
    p.add_argument("--bundle-dir", default="models/aesurv_final")
    p.add_argument("--dann-preprocess-npz", default="runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz")
    p.add_argument("--fhs-npz", default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--fhs-pq", default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-npz", default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--whi-pq", default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--clock-dir", default="feature_importance/annot/clock_lists")
    p.add_argument("--out-dir", default="feature_importance/ablation")
    p.add_argument("--n-null", type=int, default=20, help="Random-null replicates")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    log.info("Loading bundles...")
    X_meth_fhs, X_snp_fhs, t_fhs, e_fhs, n_cpg_f, n_snp_f = load_bundle_with_cache(
        "FHS", Path(args.fhs_npz), Path(args.fhs_pq), None, None, None, Path("vae_cox_cache"), True,
    )
    X_meth_whi, X_snp_whi, t_whi, e_whi, n_cpg_w, n_snp_w = load_bundle_with_cache(
        "WHI_raw", Path(args.whi_npz), Path(args.whi_pq), None, None, None, Path("vae_cox_cache"), True,
    )
    meth_f, snp_f = load_feature_names(Path(args.fhs_npz), n_cpg_f, n_snp_f, None)
    meth_w, snp_w = load_feature_names(Path(args.whi_npz), n_cpg_w, n_snp_w, None)
    X_meth_fhs, X_snp_fhs, X_meth_whi, X_snp_whi, n_cpg, n_snp = _align_common_features(
        X_meth_fhs, X_snp_fhs, meth_f, snp_f, X_meth_whi, X_snp_whi, meth_w, snp_w,
    )
    d_in = n_cpg + n_snp
    meth_names = meth_f[:n_cpg]

    preprocess = Path(args.dann_preprocess_npz)
    if not preprocess.exists():
        preprocess = Path(args.bundle_dir) / "dann_preprocess.npz"
    model = FrozenRiskModel(Path(args.bundle_dir), preprocess, device)

    idx = np.arange(X_meth_fhs.shape[0])
    strat = e_fhs.astype(np.int32) if e_fhs.sum() >= 2 else None
    tr_idx, va_idx = train_test_split(idx, test_size=args.val_frac, random_state=args.seed, stratify=strat)
    X_meth_va, X_snp_va = X_meth_fhs[va_idx], X_snp_fhs[va_idx]
    t_va, e_va = t_fhs[va_idx], e_fhs[va_idx]

    name_to_idx = build_name_index(meth_names, snp_f[:n_snp])
    sig_idx, _, _, sig_df = load_significant_indices(Path(args.sig_csv), name_to_idx, d_in)
    n_sig_cpg = int((sig_df["kind"] == "cpg").sum())
    n_sig_snp = int((sig_df["kind"] == "snp").sum())
    log.info("Significant pool: %d CpGs + %d SNPs", n_sig_cpg, n_sig_snp)

    clock_idx = _load_clock_cpg_indices(meth_names, Path(args.clock_dir))
    log.info("Clock CpGs mapped: %d", len(clock_idx))

    rows: List[Dict] = []
    fixed_conditions = [
        ("full", np.ones(d_in, dtype=bool)),
        ("significant_only", _mask_significant_only(d_in, sig_idx)),
        ("nonsignificant_only", _mask_nonsignificant_only(d_in, sig_idx)),
        ("sig_cpg_only", _mask_modality_sig(d_in, n_cpg, sig_idx, "cpg")),
        ("sig_snp_only", _mask_modality_sig(d_in, n_cpg, sig_idx, "snp")),
        ("clock_cpgs_only", _mask_clock_only(d_in, clock_idx)),
    ]
    for name, mask in fixed_conditions:
        log.info("Evaluating %s (n_active=%d)...", name, mask.sum())
        rows.append(_eval_condition(
            model, name, mask,
            X_meth_va, X_snp_va, t_va, e_va,
            X_meth_whi, X_snp_whi, t_whi, e_whi,
            n_cpg,
        ))

    null_rows = []
    for rep in range(args.n_null):
        mask = _mask_random_null(d_in, n_cpg, sig_idx, n_sig_cpg, n_sig_snp, rng)
        r = _eval_condition(
            model, f"random_null_{rep}", mask,
            X_meth_va, X_snp_va, t_va, e_va,
            X_meth_whi, X_snp_whi, t_whi, e_whi,
            n_cpg,
        )
        r["replicate"] = rep
        null_rows.append(r)
        log.info("  null %d: C_val=%.4f  C_whi=%.4f", rep, r["cindex_fhs_val"], r["cindex_whi"])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(out_dir / "ablation_random_null.csv", index=False)

    null_summary = {
        "n_replicates": args.n_null,
        "cindex_fhs_val_mean": float(null_df["cindex_fhs_val"].mean()),
        "cindex_fhs_val_std": float(null_df["cindex_fhs_val"].std()),
        "cindex_whi_mean": float(null_df["cindex_whi"].mean()),
        "cindex_whi_std": float(null_df["cindex_whi"].std()),
    }
    sig_row = df[df["condition"] == "significant_only"].iloc[0]
    full_row = df[df["condition"] == "full"].iloc[0]
    summary = {
        "n_significant_cpg": n_sig_cpg,
        "n_significant_snp": n_sig_snp,
        "n_clock_cpgs": int(len(clock_idx)),
        "full_cindex_fhs_val": float(full_row["cindex_fhs_val"]),
        "significant_only_cindex_fhs_val": float(sig_row["cindex_fhs_val"]),
        "significant_only_retains_fraction": float(
            sig_row["cindex_fhs_val"] / max(full_row["cindex_fhs_val"], 1e-6)
        ),
        "random_null": null_summary,
        "significant_beats_null_fhs_val": bool(
            sig_row["cindex_fhs_val"] > null_summary["cindex_fhs_val_mean"] + null_summary["cindex_fhs_val_std"]
        ),
        "conditions": rows,
    }
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_df = df.sort_values("cindex_fhs_val", ascending=True)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["cindex_fhs_val"], color="#3498db", alpha=0.85, label="FHS val")
    ax.barh(y, plot_df["cindex_whi"], color="#e67e22", alpha=0.55, height=0.4, label="WHI")
    ax.axvline(full_row["cindex_fhs_val"], color="#2c3e50", ls="--", lw=1, label="full (FHS val)")
    ax.axvline(null_summary["cindex_fhs_val_mean"], color="#95a5a6", ls=":", lw=1.2,
               label=f"random null mean (n={args.n_null})")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["condition"], fontsize=9)
    ax.set_xlabel("Harrell C-index")
    ax.set_title("Feature-set ablation (frozen AESurv)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_barplot.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "ablation_barplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
