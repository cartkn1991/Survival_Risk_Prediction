#!/usr/bin/env python3
"""Mean-gradient feature attribution for a joint epoch checkpoint (raw omics -> JL proj)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bio_relevance.joint_model_scores import joint_grad_model, joint_projected_features
from feature_importance_aux import back_project, ensure_W_npy, grad_proj_means, _save_top_csv, _plot_top_bar
from train_vae_cox_lite import _resolve_data_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--joint-analysis-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--alignment-dir", type=str, default="models/aesurv_final",
                   help="Directory with feature_alignment*.npy/json (same feature order as preprocess)")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-k-plot", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-rows", type=int, default=200_000)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    analysis = _resolve_data_path(args.joint_analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    align_dir = Path(args.alignment_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    manifest = json.loads((analysis / "model_manifest.json").read_text(encoding="utf-8"))
    pre_npz = Path(manifest["dann_preprocess_npz"])

    print("[1] Joint projected features (FHS+WHI) ...")
    proj, n_fhs, n_whi = joint_projected_features(analysis, device, batch_size=args.batch_size)
    proj_path = analysis / "scores" / "joint_proj_pooled.npz"
    proj_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(proj_path, x=proj, n_fhs=np.asarray(n_fhs), n_whi=np.asarray(n_whi))
    print(f"  saved {proj_path} shape={proj.shape}")

    print("[2] Gradient model ...")
    model = joint_grad_model(analysis, device)

    print("[3] <grad_proj> per cohort ...")
    g_age_fhs, g_risk_fhs = grad_proj_means(model, proj[:n_fhs], device, batch=args.batch_size)
    g_age_whi, g_risk_whi = grad_proj_means(model, proj[n_fhs:], device, batch=args.batch_size)
    g_age_pool = (n_fhs * g_age_fhs + n_whi * g_age_whi) / (n_fhs + n_whi)
    g_risk_pool = (n_fhs * g_risk_fhs + n_whi * g_risk_whi) / (n_fhs + n_whi)

    with np.load(pre_npz, allow_pickle=False, mmap_mode="r") as zpre:
        sigma = np.asarray(zpre["scaler_scale"], dtype=np.float32).copy()
        sigma = np.where(sigma < 1e-8, 1e-8, sigma).astype(np.float32)
        d_in = int(sigma.shape[0])

    print("[4] Back-project ...")
    W_npy = ensure_W_npy(pre_npz, out_dir / "dann_W.npy")
    g_raw_age_pool = back_project(g_age_pool, W_npy, sigma, chunk_rows=args.chunk_rows)
    g_raw_risk_pool = back_project(g_risk_pool, W_npy, sigma, chunk_rows=args.chunk_rows)
    g_raw_age_fhs = back_project(g_age_fhs, W_npy, sigma, chunk_rows=args.chunk_rows)
    g_raw_age_whi = back_project(g_age_whi, W_npy, sigma, chunk_rows=args.chunk_rows)
    g_raw_risk_fhs = back_project(g_risk_fhs, W_npy, sigma, chunk_rows=args.chunk_rows)
    g_raw_risk_whi = back_project(g_risk_whi, W_npy, sigma, chunk_rows=args.chunk_rows)

    align = json.loads((align_dir / "feature_alignment.json").read_text(encoding="utf-8"))
    n_cpg = int(align["n_cpg"])
    n_snp = int(align["n_snp"])
    cpg_names = np.load(align_dir / "feature_alignment_cpg.npy", allow_pickle=True).astype(object)
    snp_names = np.load(align_dir / "feature_alignment_snp.npy", allow_pickle=True).astype(object)

    for target, gpool, gfhs, gwhi in [
        ("age", g_raw_age_pool, g_raw_age_fhs, g_raw_age_whi),
        ("risk", g_raw_risk_pool, g_raw_risk_fhs, g_raw_risk_whi),
    ]:
        g_cpg = gpool[:n_cpg]
        g_snp = gpool[n_cpg:]
        _save_top_csv(g_cpg, cpg_names, "cpg", target, out_dir, top_k=args.top_k)
        _save_top_csv(g_snp, snp_names, "snp", target, out_dir, top_k=args.top_k)
        _plot_top_bar(g_cpg, cpg_names, g_snp, snp_names, target,
                      out_dir / f"top_{target}_bar.png", top_k=args.top_k_plot)
        np.savez(
            out_dir / f"{target}_attrib.npz",
            grad_raw_fhs=gfhs.astype(np.float32),
            grad_raw_whi=gwhi.astype(np.float32),
            grad_raw_pool=gpool.astype(np.float32),
            n_cpg=np.asarray(n_cpg, dtype=np.int64),
            n_snp=np.asarray(n_snp, dtype=np.int64),
        )
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
