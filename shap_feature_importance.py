#!/usr/bin/env python3
"""SHAP feature importance for AESURV-DANN-Aux risk prediction.

Standard SHAP / KernelSHAP is infeasible on a 1.37 M-d input, but the
preprocessing layer of the model is *purely linear*:

    x_proj  =  (x_raw - mu_scaler) / sigma_scaler  @  W

so any path-based attribution (Integrated Gradients, GradientSHAP) propagates
through it via the chain rule.  Setting the baseline at the dataset mean
(``baseline_raw = mu_scaler``, so ``baseline_proj = 0``) gives:

    x_proj_alpha = alpha * x_proj                          (interpolated input)
    IG_proj_k    = int_0^1  d f / d x_proj_k (x_proj_alpha)  d alpha
    IG_raw_j     = (1 / sigma_j) * (W[j,:] @ IG_proj)            (linear chain rule)
    phi_raw_j    = (x_raw_j - mu_j) * IG_raw_j  =  z_j * (W[j,:] @ IG_proj)

Sum_j phi_raw_j = f(x) - f(baseline) by the IG completeness axiom.

We compute ``IG_proj`` for every sample using a trapezoidal numerical integral
over ``--n-steps`` interpolation points (default 50), then evaluate
``phi_raw_j`` only on a focused candidate set ``J`` of features (default: the
union of AESURV-significant CpGs + SNPs from
``feature_importance/significance/significant_either.csv``).  This avoids the
20 GB cost of materialising per-sample SHAP across all 1.37 M features while
still yielding global importances for every feature the model itself flagged.

Outputs (under ``--out-dir``, default ``feature_importance/shap/``):

  shap_proj_importance.csv         per-projected-dim importance (2048 rows)
  shap_top_cpg_risk.csv            CpGs passing **within-modality** mean|SHAP| quantile
                                   (default: ≥ 90th percentile among candidates) plus optional
                                   stability filters; sorted by mean_abs
  shap_top_snp_risk.csv            same for SNPs
  shap_top_cpg_age.csv             optional, if ``--age-target`` is set
  shap_top_snp_age.csv             same
  shap_per_sample_<target>.npz     per-sample phi for candidate features
                                   (samples x K), z (samples x K), feature names
  shap_vs_gradient_<target>.png    scatter of SHAP global importance vs mean-gradient importance
  shap_bar_<target>.png            quadrant bar chart (``--bar-quadrant-k`` per corner, from candidates)
  shap_summary.json                counts, baseline, top-10 features, completeness diagnostic
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dann_aesurv_bridge import DannLatentEncoder, load_mini_dann_for_fusion
from train_aesurv_dann_latent_aux import AESurvHeadAux

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def threshold_shap_table(
    df: pd.DataFrame,
    quantile: float,
    *,
    min_mean_over_mean_abs: Optional[float],
    min_mean_abs_over_std: Optional[float],
    max_max_abs_over_mean_abs: Optional[float],
) -> Tuple[pd.DataFrame, float]:
    """Rows with mean_abs >= quantile(mean_abs); optional stability filters. Sorted by mean_abs desc."""
    if len(df) == 0:
        return df.copy(), float("nan")
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise SystemExit(f"quantile must be in (0,1); got {quantile}")
    q_thr = float(df["mean_abs"].quantile(q))
    sel = (df["mean_abs"] >= q_thr).to_numpy()
    if min_mean_over_mean_abs is not None:
        rat = (df["mean"].abs() / df["mean_abs"].clip(lower=1e-12)).to_numpy(dtype=float)
        sel &= rat >= float(min_mean_over_mean_abs)
    if min_mean_abs_over_std is not None:
        sn = (df["mean_abs"] / df["std"].clip(lower=1e-12)).to_numpy(dtype=float)
        sel &= sn >= float(min_mean_abs_over_std)
    if max_max_abs_over_mean_abs is not None:
        mxr = (df["max_abs"] / df["mean_abs"].clip(lower=1e-12)).to_numpy(dtype=float)
        sel &= mxr <= float(max_max_abs_over_mean_abs)
    out = df.loc[sel].sort_values("mean_abs", ascending=False)
    return out, q_thr


def _torch_load_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _read_txt_list(p: Path) -> List[str]:
    with open(p, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n").rstrip("\r") for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
class FrozenAgeRiskNet(nn.Module):
    """``(age_pred_years, log_hazard)`` from the projected JL input."""
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
        return age_z * self.age_sd + self.age_mu, log_h


def integrated_gradients_proj(model: FrozenAgeRiskNet, proj: np.ndarray,
                              target: str, device: torch.device, *,
                              baseline_proj: np.ndarray, n_steps: int = 50,
                              batch_size: int = 64,
                              ) -> np.ndarray:
    """Trapezoidal-rule integrated gradient at the *projected* layer for every
    sample in ``proj``.  Returns array of shape ``(N, d_proj)``."""
    n, d = proj.shape
    bl = torch.from_numpy(baseline_proj.astype(np.float32)).to(device)
    out = np.empty((n, d), dtype=np.float32)
    # trapezoidal alpha grid: alpha_k = k / n_steps, k = 0..n_steps -> n_steps+1 points
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=torch.float32)
    # trapezoidal weights: 0.5 / n_steps at endpoints, 1.0 / n_steps in the middle
    w_alpha = torch.full_like(alphas, 1.0 / n_steps); w_alpha[0] *= 0.5; w_alpha[-1] *= 0.5

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        x = torch.from_numpy(proj[s:e]).float().to(device)
        delta = (x - bl[None, :])           # (B, d)
        ig_sum = torch.zeros_like(x)
        for k, alpha in enumerate(alphas):
            x_alpha = bl[None, :] + alpha * delta
            x_alpha = x_alpha.detach().requires_grad_(True)
            age_pred, log_h = model(x_alpha)
            scalar = log_h if target == "risk" else age_pred
            g = torch.autograd.grad(scalar.sum(), x_alpha, retain_graph=False)[0]
            ig_sum = ig_sum + w_alpha[k] * g
        out[s:e] = ig_sum.detach().cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# Extract standardized raw values  z = (x_raw - mu) / sigma  for a focused J  #
# --------------------------------------------------------------------------- #
def _beta_to_m(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """log2(beta / (1 - beta)) -- matches dann_aesurv_bridge.beta_to_m_torch."""
    b = np.clip(beta, eps, 1.0 - eps)
    return np.log2(b / (1.0 - b))


def extract_standardized_focused(
    candidate_cpg_names: List[str], candidate_snp_names: List[str],
    fhs_bundle: Path, whi_bundle: Path,
    fhs_cpg_txt: Path, whi_cpg_txt: Path,
    fhs_snp_txt: Path, whi_snp_txt: Path,
    cpg_aligned_names: List[str], snp_aligned_names: List[str],
    scaler_mean: np.ndarray, scaler_scale: np.ndarray,
    meth_as_mvalues: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(z_cpg, names_cpg_kept, z_snp, names_snp_kept, aligned_idx_cpg, aligned_idx_snp)``.

    z_cpg shape: (n_samples_pooled, n_cpg_resolved)
    z_snp shape: (n_samples_pooled, n_snp_resolved)
    Samples are pooled in FHS-then-WHI order, matching the projection layout.
    """
    print(f"  reading bundle column lists ...")
    fhs_cpg = _read_txt_list(fhs_cpg_txt); whi_cpg = _read_txt_list(whi_cpg_txt)
    fhs_snp = _read_txt_list(fhs_snp_txt); whi_snp = _read_txt_list(whi_snp_txt)
    fhs_cpg_map = {n: i for i, n in enumerate(fhs_cpg)}
    whi_cpg_map = {n: i for i, n in enumerate(whi_cpg)}
    fhs_snp_map = {n: i for i, n in enumerate(fhs_snp)}
    whi_snp_map = {n: i for i, n in enumerate(whi_snp)}

    align_cpg_idx = {n: i for i, n in enumerate(cpg_aligned_names)}
    align_snp_idx = {n: i for i, n in enumerate(snp_aligned_names)}
    n_cpg = len(cpg_aligned_names); n_snp = len(snp_aligned_names)

    # Resolve candidate CpGs / SNPs that exist in BOTH cohort bundles + aligned set
    def _resolve(cands: List[str], fmap: Dict[str, int], wmap: Dict[str, int],
                 align_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        names: List[str] = []
        fcol: List[int] = []; wcol: List[int] = []; aidx: List[int] = []
        for c in cands:
            if c in fmap and c in wmap and c in align_idx:
                names.append(c); fcol.append(fmap[c]); wcol.append(wmap[c]); aidx.append(align_idx[c])
        return (np.asarray(names, dtype=object),
                np.asarray(fcol, dtype=np.int64),
                np.asarray(wcol, dtype=np.int64),
                np.asarray(aidx, dtype=np.int64))

    names_cpg, fcol_cpg, wcol_cpg, aidx_cpg = _resolve(candidate_cpg_names, fhs_cpg_map, whi_cpg_map, align_cpg_idx)
    names_snp, fcol_snp, wcol_snp, aidx_snp = _resolve(candidate_snp_names, fhs_snp_map, whi_snp_map, align_snp_idx)
    print(f"  resolved {len(names_cpg)} / {len(candidate_cpg_names)} CpGs and "
          f"{len(names_snp)} / {len(candidate_snp_names)} SNPs in both cohort bundles")

    # Load mmap'd bundles and slice columns
    print(f"  reading column slices from cohort bundles (mmap) ...")
    t0 = time.time()
    fz = np.load(fhs_bundle, allow_pickle=False, mmap_mode="r")
    wz = np.load(whi_bundle, allow_pickle=False, mmap_mode="r")
    # Sort columns for sequential access then unscramble
    def _slice(arr, cols):
        order = np.argsort(cols); inv = np.argsort(order)
        return arr[:, cols[order]].astype(np.float32)[:, inv]

    X_cpg_fhs = _slice(fz["X_meth"], fcol_cpg)
    X_snp_fhs = _slice(fz["X_snp"],  fcol_snp)
    X_cpg_whi = _slice(wz["X_meth"], wcol_cpg)
    X_snp_whi = _slice(wz["X_snp"],  wcol_snp)
    print(f"  X_cpg_fhs {X_cpg_fhs.shape}  X_snp_fhs {X_snp_fhs.shape}   "
          f"X_cpg_whi {X_cpg_whi.shape}  X_snp_whi {X_snp_whi.shape}   "
          f"in {time.time() - t0:.1f}s")

    # Pool FHS-then-WHI
    X_cpg = np.concatenate([X_cpg_fhs, X_cpg_whi], axis=0).astype(np.float32)
    X_snp = np.concatenate([X_snp_fhs, X_snp_whi], axis=0).astype(np.float32)

    # Methylation: bundle stores beta-values; the DANN scaler was fit on M-values
    # (log2(beta/(1-beta))). Apply the same transform here so z is comparable.
    if meth_as_mvalues:
        X_cpg = _beta_to_m(X_cpg).astype(np.float32)
        print(f"  applied beta -> M-value transform to {X_cpg.shape[1]} CpG columns")

    # Standardize using the scaler's mean / scale (broadcasting per column)
    mu_cpg = scaler_mean[aidx_cpg].astype(np.float32)
    sd_cpg = scaler_scale[aidx_cpg].astype(np.float32)
    mu_snp = scaler_mean[n_cpg + aidx_snp].astype(np.float32)   # SNP columns live after CpG columns
    sd_snp = scaler_scale[n_cpg + aidx_snp].astype(np.float32)
    z_cpg = (X_cpg - mu_cpg) / np.maximum(sd_cpg, 1e-8)
    z_snp = (X_snp - mu_snp) / np.maximum(sd_snp, 1e-8)

    return z_cpg, names_cpg, z_snp, names_snp, aidx_cpg, aidx_snp


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=str, default="models/aesurv_final")
    p.add_argument("--proj-fhs-npz", type=str, default="vae_cox_cache/jl_proj/proj_FHS_b24e34c31a.npz")
    p.add_argument("--proj-whi-npz", type=str, default="vae_cox_cache/jl_proj/proj_WHI_b159d5bf88.npz")
    p.add_argument("--fhs-npz", type=str,
                   default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--whi-npz", type=str,
                   default="vae_cox_cache/bundles/WHI_raw_WHI_methylation_with_snp_merged_1milfeatures_combined_training_WHI_methylation_with_snp_merged_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-cpg-txt", type=str, default="FHS_methylation_with_snp_1milfeatures_cpg_columns.txt")
    p.add_argument("--whi-cpg-txt", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_cpg_columns.txt")
    p.add_argument("--fhs-snp-txt", type=str, default="FHS_methylation_with_snp_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--whi-snp-txt", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--candidate-csv", type=str,
                   default="feature_importance/significance/significant_either.csv",
                   help="CSV with column 'feature' giving the candidate raw features to score "
                        "(typically the union of AESURV-significant CpGs+SNPs).")
    p.add_argument("--targets", type=str, default="risk",
                   help="Comma-separated list of targets to compute SHAP for: 'risk' and/or 'age'.")
    p.add_argument("--n-steps", type=int, default=50, help="Trapezoidal integration steps for IG.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--baseline", type=str, default="zero",
                   choices=["zero", "mean"],
                   help="'zero'  -> baseline_proj = 0  (equiv. to baseline_raw = scaler.mean_).  "
                        "'mean'  -> baseline_proj = mean over pooled samples.")
    p.add_argument("--top-k", type=int, default=200,
                   help="With --legacy-head-topk: number of rows for shap_top_*.csv (default 200).")
    p.add_argument("--bar-quadrant-k", type=int, default=15,
                   help="Bar plot: max features per quadrant (by signed mean SHAP).")
    p.add_argument(
        "--select-quantile-cpg",
        type=float,
        default=0.9,
        help="Keep CpGs with mean_abs >= this quantile of CpG candidate mean_abs (default 0.9 = top 10%%).",
    )
    p.add_argument(
        "--select-quantile-snp",
        type=float,
        default=0.9,
        help="Same for SNPs (default 0.9).",
    )
    p.add_argument(
        "--legacy-head-topk",
        action="store_true",
        help="If set, shap_top_*.csv = first --top-k rows by mean_abs (legacy); ignores quantile filters.",
    )
    p.add_argument(
        "--top-export-max",
        type=int,
        default=0,
        help="If >0, cap rows written to shap_top_*.csv after selection (0 = no cap).",
    )
    p.add_argument(
        "--min-mean-over-mean-abs",
        type=float,
        default=None,
        help="Optional: require |mean SHAP| / mean_abs >= this (e.g. 0.05) for directional consistency.",
    )
    p.add_argument(
        "--min-mean-abs-over-std",
        type=float,
        default=None,
        help="Optional: require mean_abs / std >= this SNR across samples.",
    )
    p.add_argument(
        "--max-max-abs-over-mean-abs",
        type=float,
        default=None,
        help="Optional: exclude if max_abs/mean_abs > this (suppress single-sample spikes).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="feature_importance/shap")
    p.add_argument("--joint-analysis-dir", type=str, default=None)
    p.add_argument("--joint-proj-npz", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.bundle_dir)
    device = torch.device(args.device)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in ("risk", "age"):
            raise SystemExit(f"Unknown target {t}; must be 'risk' or 'age'.")
    print(f"device={device}  bundle={bundle}  n_steps={args.n_steps}  targets={targets}")

    joint_dir = Path(args.joint_analysis_dir) if args.joint_analysis_dir else None
    if joint_dir is not None:
        from bio_relevance.joint_model_scores import joint_grad_model
        import json as _json
        print("\n[1/5] Loading joint gradient model ...")
        model = joint_grad_model(joint_dir, device)
        print("\n[2/5] Loading joint projections + scaler ...")
        proj_npz = Path(args.joint_proj_npz) if args.joint_proj_npz else joint_dir / "scores" / "joint_proj_pooled.npz"
        proj_all = np.load(proj_npz, allow_pickle=False)["x"].astype(np.float32)
        n_fhs = int(np.load(proj_npz, allow_pickle=False)["n_fhs"])
        n_whi = int(np.load(proj_npz, allow_pickle=False)["n_whi"])
        manifest = _json.loads((joint_dir / "model_manifest.json").read_text(encoding="utf-8"))
        pre_npz = Path(manifest["dann_preprocess_npz"])
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
        age_mu = float(cfg["aux_norm"]["age_mu"]); age_sd = float(cfg["aux_norm"]["age_sd"])
        model = FrozenAgeRiskNet(encoder, head, age_mu, age_sd).to(device).eval()
        print("\n[2/5] Loading projections + scaler ...")
        proj_fhs = np.load(args.proj_fhs_npz, allow_pickle=False)["x"].astype(np.float32)
        proj_whi = np.load(args.proj_whi_npz, allow_pickle=False)["x"].astype(np.float32)
        proj_all = np.concatenate([proj_fhs, proj_whi], axis=0)
        n_fhs, n_whi = proj_fhs.shape[0], proj_whi.shape[0]
        pre_npz = bundle / "dann_preprocess.npz"
    print(f"  FHS n={n_fhs}  WHI n={n_whi}  -> pooled {proj_all.shape}")
    with np.load(pre_npz, allow_pickle=False, mmap_mode="r") as zpre:
        scaler_mean = np.asarray(zpre["scaler_mean"], dtype=np.float32).copy()
        scaler_scale = np.asarray(zpre["scaler_scale"], dtype=np.float32).copy()
        meth_as_mvalues = bool(int(zpre["meth_as_mvalues"][0])) if "meth_as_mvalues" in zpre.files else True
    print(f"  scaler mean/scale: shape={scaler_mean.shape}  d_in={scaler_mean.size}  "
          f"meth_as_mvalues={meth_as_mvalues}")

    if args.baseline == "zero":
        baseline_proj = np.zeros(proj_all.shape[1], dtype=np.float32)
        print(f"  baseline_proj = zero  (equivalent raw baseline = scaler.mean_)")
    else:
        baseline_proj = proj_all.mean(axis=0).astype(np.float32)
        print(f"  baseline_proj = mean(proj)  ||baseline||={np.linalg.norm(baseline_proj):.3e}")

    # f(baseline) for completeness diagnostic
    with torch.no_grad():
        bl_t = torch.from_numpy(baseline_proj).float().to(device).unsqueeze(0)
        age_b, log_h_b = model(bl_t)
        f_bl = {"risk": float(log_h_b.item()), "age": float(age_b.item())}

    # ---- 3.  Integrated gradients per sample, per target (projected space)
    print(f"\n[3/5] Computing IG_proj for {proj_all.shape[0]} samples x "
          f"{proj_all.shape[1]} dims  (n_steps={args.n_steps}) ...")
    ig_proj: Dict[str, np.ndarray] = {}
    completeness: Dict[str, Tuple[float, float]] = {}
    for t in targets:
        t0 = time.time()
        ig_proj[t] = integrated_gradients_proj(
            model, proj_all, target=t, device=device,
            baseline_proj=baseline_proj, n_steps=args.n_steps,
            batch_size=args.batch_size,
        )
        # Quick completeness check at the projected layer
        phi_proj = (proj_all - baseline_proj[None, :]) * ig_proj[t]
        with torch.no_grad():
            x_t = torch.from_numpy(proj_all).float().to(device)
            age_x, log_h_x = model(x_t)
            f_x = (log_h_x if t == "risk" else age_x).cpu().numpy()
        completeness_err = float(np.abs(phi_proj.sum(axis=1) - (f_x - f_bl[t])).mean())
        max_err = float(np.abs(phi_proj.sum(axis=1) - (f_x - f_bl[t])).max())
        completeness[t] = (completeness_err, max_err)
        print(f"  {t}: IG_proj computed in {time.time() - t0:.1f}s  "
              f"||IG||~{np.linalg.norm(ig_proj[t].mean(0)):.3e}  "
              f"completeness  mean|sum(phi)-(f(x)-f(bl))| = {completeness_err:.3e}  max={max_err:.3e}")

    # Per-projected-dim importance (CSV)
    proj_imp = pd.DataFrame({"proj_dim": np.arange(proj_all.shape[1], dtype=np.int32)})
    for t in targets:
        phi_proj = (proj_all - baseline_proj[None, :]) * ig_proj[t]
        proj_imp[f"mean_abs_phi_{t}"] = np.abs(phi_proj).mean(axis=0)
        proj_imp[f"mean_phi_{t}"]    = phi_proj.mean(axis=0)
        proj_imp[f"std_phi_{t}"]     = phi_proj.std(axis=0)
    proj_imp = proj_imp.sort_values(f"mean_abs_phi_{targets[0]}", ascending=False)
    proj_imp.to_csv(out_dir / "shap_proj_importance.csv", index=False)
    print(f"  wrote {out_dir / 'shap_proj_importance.csv'}")

    # ---- 4.  Standardize raw features for the candidate set
    print(f"\n[4/5] Standardizing raw features for candidate set ...")
    cand = pd.read_csv(args.candidate_csv)
    if "kind" not in cand.columns:
        raise SystemExit(f"--candidate-csv must have a 'kind' column with cpg/snp; got {cand.columns.tolist()}")
    cand_cpg = cand[cand["kind"] == "cpg"]["feature"].astype(str).tolist()
    cand_snp = cand[cand["kind"] == "snp"]["feature"].astype(str).tolist()
    print(f"  candidate features: {len(cand_cpg)} CpGs + {len(cand_snp)} SNPs "
          f"(total {len(cand_cpg) + len(cand_snp)})")

    cpg_aligned = np.load(bundle / "feature_alignment_cpg.npy", allow_pickle=True).astype(object).tolist()
    snp_aligned = np.load(bundle / "feature_alignment_snp.npy", allow_pickle=True).astype(object).tolist()

    z_cpg, names_cpg, z_snp, names_snp, aidx_cpg, aidx_snp = extract_standardized_focused(
        cand_cpg, cand_snp,
        fhs_bundle=Path(args.fhs_npz), whi_bundle=Path(args.whi_npz),
        fhs_cpg_txt=Path(args.fhs_cpg_txt), whi_cpg_txt=Path(args.whi_cpg_txt),
        fhs_snp_txt=Path(args.fhs_snp_txt), whi_snp_txt=Path(args.whi_snp_txt),
        cpg_aligned_names=cpg_aligned, snp_aligned_names=snp_aligned,
        scaler_mean=scaler_mean, scaler_scale=scaler_scale,
        meth_as_mvalues=meth_as_mvalues,
    )
    if z_cpg.shape[0] != proj_all.shape[0] or z_snp.shape[0] != proj_all.shape[0]:
        raise SystemExit(f"row mismatch: z_cpg {z_cpg.shape}, z_snp {z_snp.shape}, proj {proj_all.shape}")

    # Some genotype encodings may have negative sentinel values for missing;
    # replace those with 0 (mu_scaler already subtracted in the model's
    # pipeline; standardised value 0 corresponds to the cohort mean).
    z_cpg = np.where(np.isfinite(z_cpg), z_cpg, 0.0).astype(np.float32)
    z_snp = np.where(np.isfinite(z_snp), z_snp, 0.0).astype(np.float32)

    # Read the W rows for our candidate features (we only need those rows)
    n_cpg_aligned = len(cpg_aligned)
    print(f"  loading W rows for {len(names_cpg) + len(names_snp)} candidate features ...")
    W_npy = Path("feature_importance/dann_W.npy")
    if not W_npy.exists():
        raise SystemExit(f"Missing W cache at {W_npy}; run feature_importance_aux.py first.")
    W = np.load(W_npy, allow_pickle=False, mmap_mode="r")
    rows_cpg = aidx_cpg.astype(np.int64)
    rows_snp = (n_cpg_aligned + aidx_snp).astype(np.int64)
    # Sorted access to mmap is much faster
    sort_c = np.argsort(rows_cpg); inv_c = np.argsort(sort_c)
    sort_s = np.argsort(rows_snp); inv_s = np.argsort(sort_s)
    W_cpg = np.ascontiguousarray(W[rows_cpg[sort_c]], dtype=np.float32)[inv_c]
    W_snp = np.ascontiguousarray(W[rows_snp[sort_s]], dtype=np.float32)[inv_s]
    print(f"  W_cpg={W_cpg.shape}  W_snp={W_snp.shape}")

    # ---- 5.  SHAP per sample for candidate features, per target
    print(f"\n[5/5] Computing SHAP for candidate features and writing outputs ...")
    summary: Dict[str, object] = {
        "n_samples": int(proj_all.shape[0]),
        "n_fhs": int(n_fhs), "n_whi": int(n_whi),
        "baseline": args.baseline, "n_steps": args.n_steps,
        "n_candidate_cpg": int(len(names_cpg)), "n_candidate_snp": int(len(names_snp)),
        "f_baseline": f_bl,
    }
    grad_dir = Path("feature_importance")

    for t in targets:
        ig_t = ig_proj[t]
        bp_cpg = ig_t @ W_cpg.T                       # (N, K_cpg)
        bp_snp = ig_t @ W_snp.T                       # (N, K_snp)
        phi_cpg = z_cpg * bp_cpg                      # SHAP per sample per CpG
        phi_snp = z_snp * bp_snp                      # SHAP per sample per SNP

        # Per-feature aggregates
        def _agg(phi, names):
            return pd.DataFrame({
                "feature":  names,
                "mean_abs": np.abs(phi).mean(axis=0),
                "mean":     phi.mean(axis=0),
                "std":      phi.std(axis=0),
                "max_abs":  np.abs(phi).max(axis=0),
                "direction": np.where(phi.mean(axis=0) > 0, "accelerator", "decelerator"),
            })

        df_cpg = _agg(phi_cpg, names_cpg).sort_values("mean_abs", ascending=False)
        df_snp = _agg(phi_snp, names_snp).sort_values("mean_abs", ascending=False)
        df_cpg["kind"] = "cpg"
        df_snp["kind"] = "snp"

        # Thresholded "top" lists (replace legacy fixed head-K by default)
        if args.legacy_head_topk:
            top_cpg = df_cpg.head(args.top_k).copy()
            top_snp = df_snp.head(args.top_k).copy()
            thr_c = thr_s = float("nan")
        else:
            top_cpg, thr_c = threshold_shap_table(
                df_cpg,
                args.select_quantile_cpg,
                min_mean_over_mean_abs=args.min_mean_over_mean_abs,
                min_mean_abs_over_std=args.min_mean_abs_over_std,
                max_max_abs_over_mean_abs=args.max_max_abs_over_mean_abs,
            )
            top_snp, thr_s = threshold_shap_table(
                df_snp,
                args.select_quantile_snp,
                min_mean_over_mean_abs=args.min_mean_over_mean_abs,
                min_mean_abs_over_std=args.min_mean_abs_over_std,
                max_max_abs_over_mean_abs=args.max_max_abs_over_mean_abs,
            )
            top_cpg = top_cpg.copy()
            top_snp = top_snp.copy()
            top_cpg["mean_abs_quantile_threshold"] = thr_c
            top_snp["mean_abs_quantile_threshold"] = thr_s
            if len(top_cpg) == 0:
                print(f"  WARNING {t}: no CpGs passed quantile/filters; writing top 50 by mean_abs fallback.")
                top_cpg = df_cpg.head(50).copy()
                top_cpg["mean_abs_quantile_threshold"] = float("nan")
            if len(top_snp) == 0:
                print(f"  WARNING {t}: no SNPs passed quantile/filters; writing top 50 by mean_abs fallback.")
                top_snp = df_snp.head(50).copy()
                top_snp["mean_abs_quantile_threshold"] = float("nan")

        if args.top_export_max > 0:
            top_cpg = top_cpg.head(int(args.top_export_max)).copy()
            top_snp = top_snp.head(int(args.top_export_max)).copy()

        top_cpg.to_csv(out_dir / f"shap_top_cpg_{t}.csv", index=False)
        top_snp.to_csv(out_dir / f"shap_top_snp_{t}.csv", index=False)
        # Also store the global per-candidate-feature table (no clipping)
        df_cpg.to_csv(out_dir / f"shap_all_cpg_{t}.csv", index=False)
        df_snp.to_csv(out_dir / f"shap_all_snp_{t}.csv", index=False)

        # Persist per-sample phi for downstream analyses (e.g. K-M splits by SHAP-defined groups)
        np.savez_compressed(
            out_dir / f"shap_per_sample_{t}.npz",
            phi_cpg=phi_cpg.astype(np.float32), names_cpg=names_cpg.astype(str),
            phi_snp=phi_snp.astype(np.float32), names_snp=names_snp.astype(str),
            z_cpg=z_cpg.astype(np.float32), z_snp=z_snp.astype(np.float32),
            n_fhs=np.asarray(n_fhs, np.int64), n_whi=np.asarray(n_whi, np.int64),
        )

        # Compare with mean-gradient importance, if available
        attrib_path = grad_dir / f"{t}_attrib.npz"
        if attrib_path.exists():
            zg = np.load(attrib_path, allow_pickle=False)
            g_raw = zg["grad_raw_pool"]
            n_cpg_aligned_local = int(zg["n_cpg"])
            g_cpg_full = g_raw[:n_cpg_aligned_local]
            g_snp_full = g_raw[n_cpg_aligned_local:]
            # Map candidate features back to their aligned-index for comparison
            g_cpg = g_cpg_full[aidx_cpg]
            g_snp = g_snp_full[aidx_snp]
            df_cpg["mean_grad"] = g_cpg
            df_snp["mean_grad"] = g_snp
            # Plot scatter: |mean_grad| (existing) vs |SHAP mean| (new)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            for ax, df, lbl in [(axes[0], df_cpg, "CpG"), (axes[1], df_snp, "SNP")]:
                ax.scatter(np.abs(df["mean_grad"]), df["mean_abs"], s=8,
                           c="#3498db" if lbl == "CpG" else "#c0392b", alpha=0.45, linewidths=0)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel(r"|mean $\partial$" + t + r"/$\partial$x|  (gradient method)")
                ax.set_ylabel(r"mean |SHAP|  (this method)")
                ax.set_title(f"{lbl} importance: gradient vs SHAP  (n={len(df):,})")
                ax.grid(True, which="both", alpha=0.3)
                # Spearman / Pearson
                from scipy.stats import spearmanr, pearsonr
                rho, _ = spearmanr(np.abs(df["mean_grad"]).fillna(0), df["mean_abs"])
                ax.text(0.04, 0.95, f"Spearman {rho:.3f}", transform=ax.transAxes,
                        fontsize=9, va="top")
            fig.tight_layout()
            fig.savefig(out_dir / f"shap_vs_gradient_{t}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out_dir / f'shap_vs_gradient_{t}.png'}")

        # Top-K bar plot (from full candidate tables for quadrant diversity)
        bqk = max(1, int(args.bar_quadrant_k))
        top_cpg_pos = df_cpg.sort_values("mean", ascending=False).head(bqk)
        top_cpg_neg = df_cpg.sort_values("mean", ascending=True).head(bqk)
        top_snp_pos = df_snp.sort_values("mean", ascending=False).head(bqk)
        top_snp_neg = df_snp.sort_values("mean", ascending=True).head(bqk)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4))

        def _bar(ax, df, c, title):
            y = np.arange(len(df))
            ax.barh(y, df["mean"], color=c, edgecolor="black", linewidth=0.4)
            ax.set_yticks(y); ax.set_yticklabels(df["feature"], fontsize=8)
            ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.6)
            ax.set_title(title, fontsize=10); ax.set_xlabel(f"mean SHAP on {t}")
            ax.grid(True, axis="x", alpha=0.3)

        _bar(axes[0, 0], top_cpg_pos, "#c0392b", f"CpGs that ACCELERATE {t}")
        _bar(axes[0, 1], top_cpg_neg, "#27ae60", f"CpGs that DECELERATE {t}")
        _bar(axes[1, 0], top_snp_pos, "#c0392b", f"SNPs that ACCELERATE {t}")
        _bar(axes[1, 1], top_snp_neg, "#27ae60", f"SNPs that DECELERATE {t}")
        fig.suptitle(
            f"Candidate features by mean SHAP ({proj_all.shape[0]} samples) — target = {t}  "
            f"(bar quadrants: top {bqk} by signed mean; see shap_top_* for quantile selection)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"shap_bar_{t}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_dir / f'shap_bar_{t}.png'}")

        # Summary entries
        summary[f"target_{t}"] = {
            "selection": {
                "legacy_head_topk": bool(args.legacy_head_topk),
                "select_quantile_cpg": float(args.select_quantile_cpg),
                "select_quantile_snp": float(args.select_quantile_snp),
                "mean_abs_thr_cpg": float(thr_c) if thr_c == thr_c else None,
                "mean_abs_thr_snp": float(thr_s) if thr_s == thr_s else None,
                "n_shap_top_cpg": int(len(top_cpg)),
                "n_shap_top_snp": int(len(top_snp)),
                "min_mean_over_mean_abs": args.min_mean_over_mean_abs,
                "min_mean_abs_over_std": args.min_mean_abs_over_std,
                "max_max_abs_over_mean_abs": args.max_max_abs_over_mean_abs,
                "top_export_max": int(args.top_export_max) if args.top_export_max > 0 else None,
            },
            "top10_cpg_by_abs": top_cpg.head(10)[["feature", "mean", "mean_abs", "direction"]]
            .to_dict(orient="records"),
            "top10_snp_by_abs": top_snp.head(10)[["feature", "mean", "mean_abs", "direction"]]
            .to_dict(orient="records"),
            "completeness_mean_err": completeness[t][0],
            "completeness_max_err": completeness[t][1],
        }

    (out_dir / "shap_summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                                encoding="utf-8")
    print(f"\nDone.  Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
