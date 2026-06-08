#!/usr/bin/env python3
"""Mortality EWAS on risk-selected CpGs vs matched random-null CpGs (FHS).

Per CpG: linear screen event ~ beta_meth + age_z + sex; optional Cox on top hits.

Outputs (feature_importance/gwas/cpg_ewas/):
  - cpg_mortality_selected_results.csv
  - cpg_mortality_null_replicates.csv
  - cpg_mortality_cox_top.csv
  - cpg_mortality_ewas_summary.json
  - cpg_mortality_qq_selected.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from gwas.gwas_common import (
    batch_linear_gwas,
    bh_fdr,
    genomic_lambda,
    read_age_sex,
    read_txt_list,
    write_json,
)

try:
    from lifelines import CoxPHFitter
except ImportError:
    CoxPHFitter = None


def _map_cpg_indices(cpg_names: list[str], all_names: list[str]) -> tuple[np.ndarray, list[str]]:
    name_to_i = {s: i for i, s in enumerate(all_names)}
    cols, resolved = [], []
    for nm in cpg_names:
        i = name_to_i.get(str(nm))
        if i is not None:
            cols.append(i)
            resolved.append(str(nm))
    return np.asarray(cols, dtype=np.int64), resolved


def _cox_one(meth: np.ndarray, time: np.ndarray, event: np.ndarray, age: np.ndarray, sex: np.ndarray) -> dict:
    df = pd.DataFrame({
        "time": time, "event": event, "age": age, "sex": sex, "cpg": meth,
    }).dropna()
    out = {"n": len(df), "n_events": int(df["event"].sum())}
    if CoxPHFitter is None or len(df) < 50 or df["event"].sum() < 8 or df["cpg"].nunique() < 2:
        out["error"] = "skip"
        return out
    cph = CoxPHFitter(penalizer=0.01)
    try:
        formula = "cpg + age + sex" if df["sex"].std() > 1e-8 else "cpg + age"
        cph.fit(df, duration_col="time", event_col="event", formula=formula)
        se = float(cph.standard_errors_["cpg"])
        z = float(cph.params_["cpg"] / se) if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        out.update(
            hr_per_unit=float(np.exp(cph.params_["cpg"])),
            p_cox=p,
            coef=float(cph.params_["cpg"]),
        )
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sig-csv", default="feature_importance/significance/significant_risk.csv")
    p.add_argument("--fhs-npz",
                   default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-meta-pq", default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", default="Share_ID")
    p.add_argument("--cpg-columns-txt", default="FHS_methylation_with_snp_1milfeatures_cpg_columns.txt")
    p.add_argument("--out-dir", default="feature_importance/gwas/cpg_ewas")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-null", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cox-top-n", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sig = pd.read_csv(args.sig_csv)
    sig_cpg = sig.loc[sig["kind"] == "cpg", "feature"].astype(str).tolist()
    all_names = read_txt_list(Path(args.cpg_columns_txt))
    col_idx, sig_cpg = _map_cpg_indices(sig_cpg, all_names)
    print(f"[cpg-ewas] Selected CpGs mapped: {len(sig_cpg)} / {len(sig)}")

    z = np.load(args.fhs_npz, mmap_mode="r")
    X_meth = z["X_meth"]
    time = z["time"].astype(np.float64)
    event = z["event"].astype(np.float32)
    n = X_meth.shape[0]
    age, sex = read_age_sex(Path(args.fhs_meta_pq), args.fhs_id_col)
    age_z = (age - age.mean()) / (age.std() + 1e-6)
    cov = np.column_stack([np.ones(n), age_z.astype(np.float64), sex.astype(np.float64)])

    name_to_i = {s: i for i, s in enumerate(all_names)}
    sig_set = set(sig_cpg)
    pool = np.array([i for i in range(len(all_names)) if all_names[i] not in sig_set], dtype=np.int64)

    print("[cpg-ewas] Linear mortality screen on selected CpGs...")
    chunks = []
    for start in range(0, len(col_idx), args.batch_size):
        end = min(len(col_idx), start + args.batch_size)
        sl = col_idx[start:end]
        names = sig_cpg[start:end]
        G = X_meth[:, sl].astype(np.float32)
        chunks.append(batch_linear_gwas(event, G, cov, names))
        if end == len(col_idx) or start % (args.batch_size * 5) == 0:
            print(f"  [{end}/{len(col_idx)}]")

    sel_df = pd.concat(chunks, ignore_index=True)
    sel_df = sel_df.rename(columns={"snp": "cpg"})
    sel_df["q"] = bh_fdr(sel_df["p"].to_numpy())
    sel_df = sel_df.sort_values("p").reset_index(drop=True)
    sel_df.to_csv(out_dir / "cpg_mortality_selected_results.csv", index=False)
    lam_sel = genomic_lambda(sel_df["p"].to_numpy())
    n_p05_sel = int((sel_df["p"] < 0.05).sum())

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    p_obs = np.sort(sel_df["p"].to_numpy())
    p_exp = (np.arange(1, len(p_obs) + 1) - 0.5) / len(p_obs)
    ax.scatter(-np.log10(p_exp), -np.log10(p_obs.clip(1e-300)), s=4, c="#2c3e50", linewidths=0)
    mx = max(-np.log10(p_exp.min()), -np.log10(p_obs.min().clip(1e-300)))
    ax.plot([0, mx], [0, mx], color="#c0392b", lw=1)
    ax.set_title(f"Selected CpGs vs mortality (FHS)  lambda={lam_sel:.3f}", fontsize=11)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    fig.tight_layout()
    fig.savefig(out_dir / "cpg_mortality_qq_selected.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    null_stats = []
    n_pick = len(sig_cpg)
    print(f"[cpg-ewas] Random null: {args.n_null} x {n_pick} CpGs...")
    for rep in range(args.n_null):
        pick = rng.choice(pool, size=n_pick, replace=False)
        G_null = X_meth[:, pick].astype(np.float32)
        null_names = [all_names[i] for i in pick]
        ndf = batch_linear_gwas(event, G_null, cov, null_names)
        null_stats.append({
            "replicate": rep,
            "n_p_lt_0.05": int((ndf["p"] < 0.05).sum()),
            "min_p": float(ndf["p"].min()),
            "lambda_gc": genomic_lambda(ndf["p"].to_numpy()),
        })
        if (rep + 1) % 10 == 0:
            print(f"  null {rep + 1}/{args.n_null}")

    null_df = pd.DataFrame(null_stats)
    null_df.to_csv(out_dir / "cpg_mortality_null_replicates.csv", index=False)

    obs_min_p = float(sel_df["p"].min())
    perm_p_min = float((null_df["min_p"] <= obs_min_p).mean())
    perm_p_n05 = float((null_df["n_p_lt_0.05"] >= n_p05_sel).mean())

    cox_rows = []
    if args.cox_top_n > 0 and CoxPHFitter is not None:
        for _, row in sel_df.head(args.cox_top_n).iterrows():
            cpg = row["cpg"]
            j = name_to_i[cpg]
            r = _cox_one(
                X_meth[:, j].astype(np.float64), time, event,
                age.astype(np.float64), sex.astype(np.float64),
            )
            r["cpg"] = cpg
            r["p_linear"] = float(row["p"])
            cox_rows.append(r)
        pd.DataFrame(cox_rows).to_csv(out_dir / "cpg_mortality_cox_top.csv", index=False)

    summary = {
        "n_selected_cpgs": len(sig_cpg),
        "n_events": int(event.sum()),
        "n_samples": int(n),
        "lambda_gc_selected": lam_sel,
        "n_p_lt_0.05_selected": n_p05_sel,
        "min_p_selected": obs_min_p,
        "null_replicates": args.n_null,
        "null_n_p05_mean": float(null_df["n_p_lt_0.05"].mean()),
        "null_n_p05_std": float(null_df["n_p_lt_0.05"].std()),
        "null_min_p_mean": float(null_df["min_p"].mean()),
        "permutation_p_enrichment_n05": perm_p_n05,
        "permutation_p_best_hit": perm_p_min,
        "top10": sel_df.head(10).to_dict(orient="records"),
    }
    write_json(out_dir / "cpg_mortality_ewas_summary.json", summary)
    print(f"[cpg-ewas] selected p<0.05: {n_p05_sel}; null mean p<0.05: {null_df['n_p_lt_0.05'].mean():.1f}")
    print(f"  perm p (enrichment @0.05): {perm_p_n05:.4f}; perm p (best hit): {perm_p_min:.4f}")
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
