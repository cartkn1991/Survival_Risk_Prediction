#!/usr/bin/env python3
"""Statistical cutoff for AESURV-DANN-Aux feature importance.

Replaces the ``top-K`` selection in ``feature_importance_aux.py`` with a
principled per-feature significance test.  For each of the 1.37 M (CpG, SNP)
inputs we compute:

    g_x_i[j]  = (W[j,:] @ g_proj_i) / sigma[j]                 per-sample raw gradient
    mu_x[j]   = (W[j,:] @ <g_proj>) / sigma[j]                 mean across samples
    var_x[j]  = (W[j,:] @ Cov_proj @ W[j,:].T) / sigma[j]^2    per-sample variance
    SE[j]     = sqrt(var_x[j] / N)
    z[j]      = mu_x[j] / SE[j]   =  sqrt(N) * (W[j,:] @ mu_p) / sqrt(W[j,:] @ Cov_p @ W[j,:].T)
    p[j]      = 2 * (1 - Phi(|z|))            two-sided
    q[j]      = Benjamini-Hochberg FDR on p

Two targets: predicted *age* and predicted *log-hazard* (risk).  A feature is
declared significant for that target when q < ``--fdr`` (default 0.05).

The back-projection is implemented as a *single* chunked streaming pass over
``dann_W.npy`` and yields both mean and variance for both targets at once.
``sigma`` cancels in z, so it is only carried through to report the per-feature
mean gradient on the same scale as the original ``feature_importance_aux.py``
output for direct comparison.

Outputs (under ``--out-dir``, default ``feature_importance/significance/``):

  significance_age.npz     mean_x, se_x, z, p, q for the 1.37 M features (age)
  significance_risk.npz    same, for log-hazard
  significant_age.csv      features with q_age < fdr, plus annotation flags
  significant_risk.csv     features with q_risk < fdr
  significant_either.csv   union (q_age < fdr OR q_risk < fdr)
  volcano_age.png          volcano (mean_x vs -log10(q)) for age
  volcano_risk.png         volcano for risk
  summary.json             tallies, thresholds, cohort counts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from dann_aesurv_bridge import DannLatentEncoder, load_mini_dann_for_fusion
from train_aesurv_dann_latent_aux import AESurvHeadAux


# --------------------------------------------------------------------------- #
# Per-sample projected gradients (mean + outer-products)                      #
# --------------------------------------------------------------------------- #
def _torch_load_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class FrozenAgeRisk(nn.Module):
    def __init__(self, encoder: DannLatentEncoder, head: AESurvHeadAux,
                 age_mu: float, age_sd: float):
        super().__init__()
        self.encoder = encoder; self.head = head
        self.age_mu = float(age_mu); self.age_sd = float(age_sd)
        for p in self.parameters():
            p.requires_grad = False
        self.encoder.eval(); self.head.eval()

    def forward(self, x_proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_dann = self.encoder(x_proj)
        _, _z, log_h, _mu, _logvar, age_z, _cell = self.head(mu_dann, sample_z=False)
        age_pred = age_z * self.age_sd + self.age_mu
        return age_pred, log_h


def per_sample_projected_grads(
    model: FrozenAgeRisk, proj: np.ndarray, device: torch.device, batch: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-sample (N, d_proj) gradient arrays for age and log-hazard."""
    n, d = proj.shape
    g_age = np.empty((n, d), dtype=np.float32)
    g_risk = np.empty((n, d), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        x = torch.from_numpy(proj[s:e]).float().to(device).requires_grad_(True)
        age_pred, log_h = model(x)
        ga = torch.autograd.grad(age_pred.sum(), x, retain_graph=True, create_graph=False)[0]
        gr = torch.autograd.grad(log_h.sum(), x, retain_graph=False, create_graph=False)[0]
        g_age[s:e] = ga.detach().cpu().numpy().astype(np.float32)
        g_risk[s:e] = gr.detach().cpu().numpy().astype(np.float32)
        del x, age_pred, log_h, ga, gr
    return g_age, g_risk


# --------------------------------------------------------------------------- #
# Streaming back-projection of (mean, covariance) -> per-feature (mean, var) #
# --------------------------------------------------------------------------- #
def back_project_mean_and_var(
    mean_p_age: np.ndarray, cov_p_age: np.ndarray,
    mean_p_risk: np.ndarray, cov_p_risk: np.ndarray,
    W_npy_path: Path, sigma: np.ndarray,
    n_samples: int,
    chunk_rows: int = 200_000,
) -> dict:
    """Stream a single pass over ``W`` to fill mean_x and var_x per feature
    (for both age and risk targets simultaneously).

    Returns dict with arrays ``mean_age``, ``var_age``, ``mean_risk``,
    ``var_risk``, each of shape ``(d_in,)``, dtype float32.

    Math (per feature j, per target t):
        mean_x_t[j] = (W[j,:] @ mu_p_t) / sigma[j]
        var_x_t[j]  = (W[j,:] @ Cov_p_t @ W[j,:].T) / sigma[j]^2
        Then SE_t[j] = sqrt(var_x_t[j] / N) and z_t[j] = mean_x_t[j] / SE_t[j].
    """
    d_in = int(sigma.shape[0])
    out = {
        "mean_age":  np.empty(d_in, dtype=np.float32),
        "var_age":   np.empty(d_in, dtype=np.float32),
        "mean_risk": np.empty(d_in, dtype=np.float32),
        "var_risk":  np.empty(d_in, dtype=np.float32),
    }
    W = np.load(W_npy_path, allow_pickle=False, mmap_mode="r")
    if W.shape[0] != d_in:
        raise SystemExit(f"W shape mismatch: W={W.shape} vs sigma={sigma.shape}")

    mu_a = np.asarray(mean_p_age, dtype=np.float32)
    mu_r = np.asarray(mean_p_risk, dtype=np.float32)
    # Precompute W @ Cov by chunks (need W_chunk @ Cov -> then *W_chunk).sum(1))
    # so we just compute everything per chunk.
    Cov_a = np.asarray(cov_p_age, dtype=np.float32)
    Cov_r = np.asarray(cov_p_risk, dtype=np.float32)
    sigma2 = (sigma.astype(np.float32) ** 2)

    t0 = time.time()
    n_chunks = (d_in + chunk_rows - 1) // chunk_rows
    for ci, s in enumerate(range(0, d_in, chunk_rows)):
        e = min(d_in, s + chunk_rows)
        Wc = np.ascontiguousarray(W[s:e], dtype=np.float32)  # (chunk, d_proj)
        sig_c = sigma[s:e]; sig2_c = sigma2[s:e]
        # Mean projection -> per-feature mean
        m_a = Wc @ mu_a; m_r = Wc @ mu_r
        out["mean_age"][s:e]  = (m_a / sig_c).astype(np.float32)
        out["mean_risk"][s:e] = (m_r / sig_c).astype(np.float32)
        # Variance: diag(W @ Cov @ W.T) per row
        WC_a = Wc @ Cov_a                                   # (chunk, d_proj)
        var_a = np.einsum("ij,ij->i", WC_a, Wc)
        WC_r = Wc @ Cov_r
        var_r = np.einsum("ij,ij->i", WC_r, Wc)
        out["var_age"][s:e]  = (var_a / sig2_c).astype(np.float32)
        out["var_risk"][s:e] = (var_r / sig2_c).astype(np.float32)
        dt = time.time() - t0
        print(f"  chunk {ci + 1}/{n_chunks} ({100 * e / d_in:5.1f}%)  "
              f"elapsed {dt:6.1f}s", flush=True)
    return out


# --------------------------------------------------------------------------- #
# z, p, q                                                                     #
# --------------------------------------------------------------------------- #
def compute_stats(mean_x: np.ndarray, var_x: np.ndarray, n_samples: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (se, z, p, q) -- two-sided + BH FDR over the entire feature axis."""
    se = np.sqrt(np.maximum(var_x, 0.0) / float(n_samples)).astype(np.float64)
    # Guard against near-zero variance: those features should not get z=inf
    # because numerator is also tiny; use a relative floor on se.
    floor = max(1e-30, np.nanmedian(se) * 1e-6)
    se_floored = np.where(se < floor, floor, se)
    z = (mean_x.astype(np.float64) / se_floored)
    # Two-sided normal p-value with log-space for tail accuracy
    p = 2.0 * stats.norm.sf(np.abs(z))
    p = np.clip(p, 1e-300, 1.0)
    # Benjamini-Hochberg FDR
    m = p.size
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    bh = p_sorted * m / ranks
    # Make BH monotone non-increasing from the right
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    q = np.empty(m, dtype=np.float64)
    q[order] = bh
    return se.astype(np.float32), z.astype(np.float32), p.astype(np.float32), q.astype(np.float32)


# --------------------------------------------------------------------------- #
# Robust outlier (effect-size) cutoff via MAD on the back-projected gradient. #
# --------------------------------------------------------------------------- #
def robust_effect_threshold(mean_x: np.ndarray, k_sigma: float = 5.0) -> Tuple[float, float, float]:
    """Treat |mean_x| as a half-normal-ish bulk distribution; flag features in
    the extreme tail using a MAD-based robust z-score.

    Returns ``(median_abs, sigma_mad, threshold)`` where ``threshold = median + k * sigma_mad``
    and ``sigma_mad = MAD(|mean_x|) / 0.6745`` is the Gaussian-consistent scale.
    """
    a = np.abs(mean_x).astype(np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    sigma = max(mad / 0.6745, 1e-300)
    thr = med + k_sigma * sigma
    return med, sigma, thr


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_volcano(mean_x: np.ndarray, q: np.ndarray, selected: np.ndarray,
                 kind_mask_cpg: np.ndarray, target: str, eff_thresh: float,
                 k_sigma: float, out_path: Path, max_points: int = 250_000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = mean_x.size
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        # Always keep all the *selected* features so they appear in the plot.
        sel_idx = np.flatnonzero(selected)
        idx = np.unique(np.concatenate([idx, sel_idx]))
    else:
        idx = np.arange(n)
    qsub  = q[idx]; msub = mean_x[idx]
    cpg_sub = kind_mask_cpg[idx]; sel_sub = selected[idx]
    logq = -np.log10(np.clip(qsub, 1e-300, 1.0))

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    # Background non-selected
    ns_cpg = ~sel_sub & cpg_sub
    ns_snp = ~sel_sub & ~cpg_sub
    ax.scatter(msub[ns_cpg], logq[ns_cpg], s=2, c="#bdc3c7", alpha=0.25,
               label=f"CpG (n.s., {int((~selected & kind_mask_cpg).sum()):,})", linewidths=0)
    ax.scatter(msub[ns_snp], logq[ns_snp], s=2, c="#dadfe1", alpha=0.25,
               label=f"SNP (n.s., {int((~selected & ~kind_mask_cpg).sum()):,})", linewidths=0)
    # Selected
    s_cpg = sel_sub & cpg_sub
    s_snp = sel_sub & ~cpg_sub
    ax.scatter(msub[s_cpg], logq[s_cpg], s=8, c="#2980b9", alpha=0.85,
               label=f"CpG selected ({int((selected & kind_mask_cpg).sum()):,})", linewidths=0)
    ax.scatter(msub[s_snp], logq[s_snp], s=8, c="#c0392b", alpha=0.85,
               label=f"SNP selected ({int((selected & ~kind_mask_cpg).sum()):,})", linewidths=0)
    ax.axvline(+eff_thresh, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.axvline(-eff_thresh, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel(f"mean d({target})/d(feature)   (back-projected to raw scale)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{BH\ FDR})$")
    n_sel = int(selected.sum())
    ax.set_title(f"Volcano: {target}  |  features = {n:,}  |  "
                 f"selected at |mean| > median + {k_sigma:.1f}*sigma_MAD "
                 f"(={eff_thresh:.2e}) & q<0.05  ->  {n_sel:,}", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=True, markerscale=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=str, default="models/aesurv_final")
    p.add_argument("--proj-fhs-npz", type=str, default="vae_cox_cache/jl_proj/proj_FHS_b24e34c31a.npz")
    p.add_argument("--proj-whi-npz", type=str, default="vae_cox_cache/jl_proj/proj_WHI_b159d5bf88.npz")
    p.add_argument("--W-npy", type=str, default="feature_importance/dann_W.npy",
                   help="Cached .npy of the JL projection matrix (extract via feature_importance_aux.py).")
    p.add_argument("--fdr", type=float, default=0.05,
                   help="Benjamini-Hochberg FDR cutoff (reported, but not the primary selector).")
    p.add_argument("--k-sigma", type=float, default=5.0,
                   help="Primary selection: |mean_grad| > median + k_sigma * MAD/0.6745 (~k_sigma Gaussian SDs).")
    p.add_argument("--out-dir", type=str, default="feature_importance/significance")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-rows", type=int, default=200_000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--joint-analysis-dir", type=str, default=None,
                   help="Use joint checkpoint projections/gradients instead of frozen bundle")
    p.add_argument("--joint-proj-npz", type=str, default=None,
                   help="Pooled joint projections (default: <joint-analysis-dir>/scores/joint_proj_pooled.npz)")
    p.add_argument("--alignment-dir", type=str, default="models/aesurv_final")
    args = p.parse_args()

    bundle = Path(args.bundle_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device={device}  bundle={bundle}  W={args.W_npy}  FDR<{args.fdr}")

    joint_dir = Path(args.joint_analysis_dir) if args.joint_analysis_dir else None
    if joint_dir is not None:
        from bio_relevance.joint_model_scores import joint_grad_model
        print("\n[1/5] Loading joint gradient model ...")
        model = joint_grad_model(joint_dir, device)
        print("\n[2/5] Loading joint projections + per-sample gradients ...")
        proj_npz = Path(args.joint_proj_npz) if args.joint_proj_npz else joint_dir / "scores" / "joint_proj_pooled.npz"
        z = np.load(proj_npz, allow_pickle=False)
        proj_all = z["x"].astype(np.float32)
        bundle = Path(args.alignment_dir)
        print(f"  joint pooled proj {proj_all.shape}")
    else:
        print("\n[1/5] Loading frozen DANN encoder + AESURV head ...")
        mini, _cfg = load_mini_dann_for_fusion(bundle / "dann_encoder.pt", map_location=device)
        encoder = DannLatentEncoder(mini.to(device)).eval()
        head_ck = _torch_load_compat(bundle / "aesurv_head.pt", map_location=device)
        cfg = head_ck["config"]
        head = AESurvHeadAux(
            in_dim=int(cfg["in_dim"]), enc_hidden=tuple(cfg["enc_hidden"]),
            dec_hidden=tuple(cfg["dec_hidden"]), z_dim=int(cfg["z_dim"]),
            cohort_hidden=int(cfg["cohort_hidden"]), dropout=float(cfg["dropout"]),
            n_cells=int(cfg["n_cells"]),
        ).to(device).eval()
        head.load_state_dict(head_ck["state_dict"])
        aux_norm = cfg["aux_norm"]
        age_mu = float(aux_norm["age_mu"]); age_sd = float(aux_norm["age_sd"])
        model = FrozenAgeRisk(encoder, head, age_mu, age_sd).to(device).eval()
        print("\n[2/5] Loading projections + computing per-sample projected gradients ...")
        proj_fhs = np.load(args.proj_fhs_npz, allow_pickle=False)["x"].astype(np.float32)
        proj_whi = np.load(args.proj_whi_npz, allow_pickle=False)["x"].astype(np.float32)
        proj_all = np.concatenate([proj_fhs, proj_whi], axis=0)
        print(f"  FHS proj {proj_fhs.shape}  +  WHI proj {proj_whi.shape}  ->  pooled {proj_all.shape}")
    g_age, g_risk = per_sample_projected_grads(model, proj_all, device, batch=args.batch_size)
    N, D = g_age.shape
    print(f"  per-sample gradient tensors: g_age {g_age.shape}, g_risk {g_risk.shape}")

    # mean + (sample) covariance in projected space
    mu_a = g_age.mean(axis=0).astype(np.float64)
    mu_r = g_risk.mean(axis=0).astype(np.float64)
    Ga_c = g_age.astype(np.float64) - mu_a
    Gr_c = g_risk.astype(np.float64) - mu_r
    Cov_a = (Ga_c.T @ Ga_c) / max(1, N - 1)
    Cov_r = (Gr_c.T @ Gr_c) / max(1, N - 1)
    print(f"  ||mu_a||={np.linalg.norm(mu_a):.3e}  trace(Cov_a)={np.trace(Cov_a):.3e}")
    print(f"  ||mu_r||={np.linalg.norm(mu_r):.3e}  trace(Cov_r)={np.trace(Cov_r):.3e}")

    # ---- 3.  Sigma + W
    print("\n[3/5] Reading sigma ...")
    if joint_dir is not None:
        import json as _json
        manifest = _json.loads((joint_dir / "model_manifest.json").read_text(encoding="utf-8"))
        pre_npz = Path(manifest["dann_preprocess_npz"])
    else:
        pre_npz = bundle / "dann_preprocess.npz"
    with np.load(pre_npz, allow_pickle=False, mmap_mode="r") as zpre:
        sigma = np.asarray(zpre["scaler_scale"], dtype=np.float32).copy()
        sigma = np.where(sigma < 1e-8, 1e-8, sigma).astype(np.float32)
        d_in = int(sigma.shape[0])
    W_npy = Path(args.W_npy)
    if not W_npy.exists():
        raise SystemExit(f"W cache missing: {W_npy}.  Run feature_importance_aux.py first to extract it.")
    print(f"  d_in={d_in}  W cache={W_npy} ({W_npy.stat().st_size / 1e9:.2f} GB)")

    # ---- 4.  Streamed back-projection of (mean, variance) per target
    print(f"\n[4/5] Streaming back-projection (chunk_rows={args.chunk_rows}) ...")
    bp = back_project_mean_and_var(
        mu_a, Cov_a, mu_r, Cov_r,
        W_npy_path=W_npy, sigma=sigma, n_samples=N,
        chunk_rows=args.chunk_rows,
    )

    # ---- 5.  z, p, q  + selection + plots
    print("\n[5/5] Computing z/p/FDR and writing outputs ...")
    se_a, z_a, p_a, q_a = compute_stats(bp["mean_age"], bp["var_age"], N)
    se_r, z_r, p_r, q_r = compute_stats(bp["mean_risk"], bp["var_risk"], N)

    align_root = Path(args.alignment_dir) if joint_dir is not None else bundle
    align = json.loads((align_root / "feature_alignment.json").read_text(encoding="utf-8"))
    n_cpg = int(align["n_cpg"]); n_snp = int(align["n_snp"])
    if n_cpg + n_snp != d_in:
        raise SystemExit(f"Feature alignment {n_cpg}+{n_snp}={n_cpg + n_snp} != d_in {d_in}")
    cpg_names = np.load(align_root / "feature_alignment_cpg.npy", allow_pickle=True).astype(object)
    snp_names = np.load(align_root / "feature_alignment_snp.npy", allow_pickle=True).astype(object)
    feature_names = np.concatenate([cpg_names, snp_names])
    feature_kind = np.array(["cpg"] * n_cpg + ["snp"] * n_snp)
    is_cpg = np.arange(d_in) < n_cpg

    # Robust effect-size thresholds (PRIMARY selector for downstream analyses)
    med_a, sig_a, thr_a = robust_effect_threshold(bp["mean_age"],  k_sigma=args.k_sigma)
    med_r, sig_r, thr_r = robust_effect_threshold(bp["mean_risk"], k_sigma=args.k_sigma)
    sel_age  = (np.abs(bp["mean_age"])  > thr_a) & (q_a < args.fdr)
    sel_risk = (np.abs(bp["mean_risk"]) > thr_r) & (q_r < args.fdr)
    print(f"  age  threshold |mean_grad| > median + {args.k_sigma}*sigma_MAD = "
          f"{med_a:.3e} + {args.k_sigma}*{sig_a:.3e} = {thr_a:.3e}")
    print(f"  risk threshold |mean_grad| > median + {args.k_sigma}*sigma_MAD = "
          f"{med_r:.3e} + {args.k_sigma}*{sig_r:.3e} = {thr_r:.3e}")

    np.savez_compressed(
        out_dir / "significance_age.npz",
        mean_x=bp["mean_age"], se=se_a, z=z_a, p=p_a, q=q_a,
        selected=sel_age,
        n_cpg=np.asarray(n_cpg, np.int64), n_snp=np.asarray(n_snp, np.int64),
        n_samples=np.asarray(N, np.int64), fdr=np.asarray(args.fdr, np.float32),
        k_sigma=np.asarray(args.k_sigma, np.float32),
        threshold=np.asarray(thr_a, np.float32),
        median_abs=np.asarray(med_a, np.float32),
        sigma_mad=np.asarray(sig_a, np.float32),
    )
    np.savez_compressed(
        out_dir / "significance_risk.npz",
        mean_x=bp["mean_risk"], se=se_r, z=z_r, p=p_r, q=q_r,
        selected=sel_risk,
        n_cpg=np.asarray(n_cpg, np.int64), n_snp=np.asarray(n_snp, np.int64),
        n_samples=np.asarray(N, np.int64), fdr=np.asarray(args.fdr, np.float32),
        k_sigma=np.asarray(args.k_sigma, np.float32),
        threshold=np.asarray(thr_r, np.float32),
        median_abs=np.asarray(med_r, np.float32),
        sigma_mad=np.asarray(sig_r, np.float32),
    )
    print(f"  wrote significance_*.npz")

    def _save_csv(mask: np.ndarray, mean_x: np.ndarray, se: np.ndarray, z: np.ndarray,
                  p: np.ndarray, q: np.ndarray, path: Path) -> int:
        df = pd.DataFrame({
            "feature":   feature_names[mask].astype(str),
            "kind":      feature_kind[mask],
            "mean_grad": mean_x[mask].astype(np.float64),
            "abs_mean_grad": np.abs(mean_x[mask]).astype(np.float64),
            "se":        se[mask].astype(np.float64),
            "z":         z[mask].astype(np.float64),
            "p":         p[mask].astype(np.float64),
            "q":         q[mask].astype(np.float64),
            "direction": np.where(mean_x[mask] > 0, "accelerator", "decelerator"),
        })
        df = df.sort_values("abs_mean_grad", ascending=False, kind="mergesort")
        df.to_csv(path, index=False)
        return int(mask.sum())

    n_sig_age  = _save_csv(sel_age,  bp["mean_age"],  se_a, z_a, p_a, q_a, out_dir / "significant_age.csv")
    n_sig_risk = _save_csv(sel_risk, bp["mean_risk"], se_r, z_r, p_r, q_r, out_dir / "significant_risk.csv")

    # Joint table on the union of the two selectors
    mask_either = sel_age | sel_risk
    df_either = pd.DataFrame({
        "feature":          feature_names[mask_either].astype(str),
        "kind":             feature_kind[mask_either],
        "mean_grad_age":    bp["mean_age"][mask_either].astype(np.float64),
        "z_age":            z_a[mask_either].astype(np.float64),
        "q_age":            q_a[mask_either].astype(np.float64),
        "mean_grad_risk":   bp["mean_risk"][mask_either].astype(np.float64),
        "z_risk":           z_r[mask_either].astype(np.float64),
        "q_risk":           q_r[mask_either].astype(np.float64),
        "sel_age":          sel_age[mask_either],
        "sel_risk":         sel_risk[mask_either],
        "sel_both":         sel_age[mask_either] & sel_risk[mask_either],
    })
    df_either["max_abs"] = np.maximum(np.abs(df_either["mean_grad_age"]),
                                       np.abs(df_either["mean_grad_risk"]))
    df_either = df_either.sort_values("max_abs", ascending=False, kind="mergesort")
    df_either.to_csv(out_dir / "significant_either.csv", index=False)

    # Also report counts at several reference thresholds for transparency.
    def _tally(mean_x, q, z, label):
        a = np.abs(mean_x)
        tallies = {
            "n_q05":           int((q < 0.05).sum()),
            "n_q01":           int((q < 0.01).sum()),
            "n_q1e6":          int((q < 1e-6).sum()),
            "n_bonferroni_5pc": int((q < 0.05 / max(1, d_in)).sum()),
            "n_top1pct_abs":   int((a > np.quantile(a, 0.99)).sum()),
            "n_top0p5pct_abs": int((a > np.quantile(a, 0.995)).sum()),
            "n_top0p1pct_abs": int((a > np.quantile(a, 0.999)).sum()),
        }
        return tallies

    summary = {
        "n_samples":  N,
        "n_features": d_in,
        "n_cpg":      n_cpg,
        "n_snp":      n_snp,
        "fdr":        args.fdr,
        "k_sigma":    args.k_sigma,
        "age_threshold":   {"median_abs": med_a, "sigma_mad": sig_a, "threshold": thr_a},
        "risk_threshold":  {"median_abs": med_r, "sigma_mad": sig_r, "threshold": thr_r},
        "n_selected_age":       int(sel_age.sum()),
        "n_selected_age_cpg":   int((sel_age & is_cpg).sum()),
        "n_selected_age_snp":   int((sel_age & ~is_cpg).sum()),
        "n_selected_risk":      int(sel_risk.sum()),
        "n_selected_risk_cpg":  int((sel_risk & is_cpg).sum()),
        "n_selected_risk_snp":  int((sel_risk & ~is_cpg).sum()),
        "n_selected_either":    int(mask_either.sum()),
        "n_selected_both":      int((sel_age & sel_risk).sum()),
        "n_selected_both_cpg":  int((sel_age & sel_risk & is_cpg).sum()),
        "n_selected_both_snp":  int((sel_age & sel_risk & ~is_cpg).sum()),
        "tallies_age":  _tally(bp["mean_age"],  q_a, z_a, "age"),
        "tallies_risk": _tally(bp["mean_risk"], q_r, z_r, "risk"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                          encoding="utf-8")
    print(f"  AGE  selected: {n_sig_age:,}  "
          f"({int((sel_age & is_cpg).sum()):,} CpG / {int((sel_age & ~is_cpg).sum()):,} SNP)")
    print(f"  RISK selected: {n_sig_risk:,}  "
          f"({int((sel_risk & is_cpg).sum()):,} CpG / {int((sel_risk & ~is_cpg).sum()):,} SNP)")
    print(f"  EITHER selected: {int(mask_either.sum()):,}  "
          f"  BOTH selected: {int((sel_age & sel_risk).sum()):,}")

    # Volcanoes
    plot_volcano(bp["mean_age"],  q_a, sel_age,  is_cpg, "age",  thr_a,
                 args.k_sigma, out_dir / "volcano_age.png")
    plot_volcano(bp["mean_risk"], q_r, sel_risk, is_cpg, "risk", thr_r,
                 args.k_sigma, out_dir / "volcano_risk.png")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
