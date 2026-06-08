#!/usr/bin/env python3
"""Cell-fraction vs prediction correlations + CpG probe context (context35) enrichment.

Outputs ``feature_importance/bio_relevance/cell_chromatin/``:

  cell_vs_risk_correlations.csv    Pearson r: Houseman cells vs log_h, age_pred, residual log_h
  context35_enrichment.json        top SHAP CpGs vs random manifest CpGs: context35 distribution
  context35_compare.png            bar chart of mean context35

Residual log_h is residualised on predicted age + cohort (WHI chronological age parquet
is not read here to avoid pyarrow thrift limits on some installs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bio_relevance.model_scores import pooled_predictions


def load_cells_aligned(
    fhs_cell_pq: Path, whi_cell_pq: Path, cell_cols: List[str],
    fhs_meta_pq: Path, whi_id_col: str,
) -> np.ndarray:
    """Return cells pooled FHS-then-WHI (same order as JL projections / bundles)."""
    df_fhs = pd.read_parquet(fhs_meta_pq, columns=["Share_ID", "age"])
    df_cell_f = pd.read_parquet(fhs_cell_pq)
    df_cell_f["Share_ID"] = df_cell_f["Share_ID"].astype(str)
    df_fhs["Share_ID"] = df_fhs["Share_ID"].astype(str)
    jf = df_fhs.merge(df_cell_f, on="Share_ID", how="inner")
    cells_f = jf[cell_cols].to_numpy(dtype=np.float64)

    df_cell_w = pd.read_parquet(whi_cell_pq)
    idc = "sample_ID" if "sample_ID" in df_cell_w.columns else whi_id_col
    df_cell_w[idc] = df_cell_w[idc].astype(str)
    cells_w = df_cell_w[cell_cols].to_numpy(dtype=np.float64)
    return np.vstack([cells_f, cells_w])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/cell_chromatin")
    p.add_argument("--bundle-dir", type=str, default="models/aesurv_final")
    p.add_argument("--proj-fhs-npz", type=str, default="vae_cox_cache/jl_proj/proj_FHS_b24e34c31a.npz")
    p.add_argument("--proj-whi-npz", type=str, default="vae_cox_cache/jl_proj/proj_WHI_b159d5bf88.npz")
    p.add_argument("--fhs-meta-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--fhs-cell-pq", type=str, default="FHS_cell_composition.parquet")
    p.add_argument("--whi-cell-pq", type=str, default="WHI_cell_composition.parquet")
    p.add_argument("--whi-id-col", type=str, default="sample_ID",
                   help="WHI cell parquet id column (default sample_ID).")
    p.add_argument("--cell-cols", type=str, default="B,NK,CD4T,CD8T,Mono,Neutro")
    p.add_argument("--annot-csv", type=str, default="Annotation.csv")
    p.add_argument("--shap-top-cpg", type=str, default="feature_importance/shap/shap_top_cpg_risk.csv")
    p.add_argument("--n-random-cpg", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--joint-analysis-dir", type=str, default=None)
    args = p.parse_args()

    import torch
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cell_cols = [c.strip() for c in args.cell_cols.split(",") if c.strip()]

    joint_dir = Path(args.joint_analysis_dir) if args.joint_analysis_dir else None
    log_h, age_p, n_fhs, n_whi = pooled_predictions(
        Path(args.bundle_dir), Path(args.proj_fhs_npz), Path(args.proj_whi_npz), device,
        joint_analysis_dir=joint_dir,
    )
    cells = load_cells_aligned(
        Path(args.fhs_cell_pq), Path(args.whi_cell_pq), cell_cols,
        Path(args.fhs_meta_pq), args.whi_id_col,
    )
    n = min(len(log_h), cells.shape[0])
    log_h = log_h[:n]
    age_p = age_p[:n]
    cells = cells[:n, :]
    print(f"  aligned n={n}  (FHS block {n_fhs}, WHI {n_whi})")

    # Residualise log-hazard on predicted age + cohort
    age_line = age_p.astype(np.float64)
    cohort = np.concatenate([np.zeros(n_fhs, dtype=np.float64), np.ones(n - n_fhs, dtype=np.float64)])[:n]
    Xc = np.column_stack([np.ones(n), age_line, cohort])
    coef, *_ = np.linalg.lstsq(Xc, log_h.astype(np.float64), rcond=None)
    resid = log_h.astype(np.float64) - Xc @ coef

    rows = []
    for i, cname in enumerate(cell_cols):
        r_h, p_h = stats.pearsonr(cells[:, i], log_h[: cells.shape[0]])
        r_a, p_a = stats.pearsonr(cells[:, i], age_p[: cells.shape[0]])
        r_r, p_r = stats.pearsonr(cells[:, i], resid)
        rows.append({
            "cell": cname,
            "r_cell_log_h": float(r_h), "p_cell_log_h": float(p_h),
            "r_cell_age_pred": float(r_a), "p_cell_age_pred": float(p_a),
            "r_cell_risk_resid": float(r_r), "p_cell_risk_resid": float(p_r),
        })
    pd.DataFrame(rows).to_csv(out / "cell_vs_risk_correlations.csv", index=False)
    print(f"  wrote {out / 'cell_vs_risk_correlations.csv'}")

    # context35 for top SHAP CpGs vs random probes
    top_cpgs = pd.read_csv(args.shap_top_cpg, nrows=300)["feature"].astype(str).tolist()
    ann = pd.read_csv(args.annot_csv, usecols=["probeID", "context35"], low_memory=False)
    ann["probeID"] = ann["probeID"].astype(str)
    sub = ann[ann["probeID"].isin(top_cpgs)].drop_duplicates("probeID")
    rng = np.random.default_rng(42)
    pool = ann[~ann["probeID"].isin(top_cpgs)]["probeID"].to_numpy()
    pick = rng.choice(pool, size=min(args.n_random_cpg, len(pool)), replace=False)
    rnd = ann[ann["probeID"].isin(pick)]
    def summarize(df: pd.DataFrame) -> Dict[str, float]:
        v = df["context35"].astype(float)
        return {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()), "std": float(v.std())}
    summ = {"top_shap_cpg": summarize(sub), "random_manifest_cpg": summarize(rnd)}
    (out / "context35_enrichment.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [summ["top_shap_cpg"]["mean"], summ["random_manifest_cpg"]["mean"]],
           yerr=[summ["top_shap_cpg"]["std"] / np.sqrt(max(1, summ["top_shap_cpg"]["n"])),
                 summ["random_manifest_cpg"]["std"] / np.sqrt(max(1, summ["random_manifest_cpg"]["n"]))],
           capsize=4, color=["#c0392b", "#7f8c8d"], edgecolor="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Top SHAP CpGs", "Random CpGs"])
    ax.set_ylabel("mean context35 (local CpG density in ±35bp)")
    ax.set_title("Probe context: model-important vs random")
    fig.tight_layout()
    fig.savefig(out / "context35_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'context35_enrichment.json'} and context35_compare.png")


if __name__ == "__main__":
    main()
