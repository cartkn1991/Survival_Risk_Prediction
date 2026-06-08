#!/usr/bin/env python3
"""Per-SNP Cox PH models for mortality in FHS or WHI.

For each SNP in a configurable list, fits:

    Surv(time, event) ~ SNP_additive + age [+ sex_female if variable]

Outputs (under --out-dir):
  - {cohort}_snp_cox_results.csv   HR, 95% CI, p, n, n_events per SNP
  - {cohort}_snp_cox_summary.json
  - snp_cox_replication.csv        (when --merge-fhs-csv provided with WHI run)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from gwas.gwas_common import fixed_effect_meta, load_genotype_columns, read_age_sex, resolve_snp_columns, write_json

try:
    from lifelines import CoxPHFitter
except ImportError:
    CoxPHFitter = None  # type: ignore


def _load_snp_list(args: argparse.Namespace) -> List[str]:
    if args.snp_list:
        return [s.strip() for s in args.snp_list if s.strip()]
    if args.snp_csv:
        df = pd.read_csv(args.snp_csv)
        col = args.snp_col
        if col not in df.columns:
            for alt in ("snp", "feature", "feature_id", "SNP"):
                if alt in df.columns:
                    col = alt
                    break
            else:
                raise SystemExit(f"Could not find SNP column in {args.snp_csv}")
        snps = df[col].astype(str).tolist()
        if args.top_n > 0:
            snps = snps[: args.top_n]
        return snps
    raise SystemExit("Provide --snp-list or --snp-csv")


def _fit_one_cox(
    snp: str,
    g: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    penalizer: float,
    *,
    min_n: int = 50,
    min_events: int = 8,
) -> dict:
    g = np.where((g >= 0) & (g <= 2), g, np.nan)
    df = pd.DataFrame({
        "time": time.astype(np.float64),
        "event": event.astype(np.int32),
        "age": age.astype(np.float64),
        "sex": sex.astype(np.float64),
        "snp": g.astype(np.float64),
    }).dropna()
    out = {"snp": snp, "n": int(len(df)), "n_events": int(df["event"].sum())}
    if CoxPHFitter is None:
        out["error"] = "lifelines not installed"
        return out
    if len(df) < min_n or df["event"].sum() < min_events:
        out["error"] = "insufficient sample or events"
        return out
    if df["snp"].nunique() < 2:
        out["error"] = "SNP has no variation"
        return out

    cov_cols = ["age"]
    if df["sex"].std() > 1e-8:
        cov_cols.append("sex")
    formula = "snp + " + " + ".join(cov_cols)

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df, duration_col="time", event_col="event", formula=formula)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    hr = float(np.exp(cph.params_["snp"]))
    se = float(cph.standard_errors_["snp"])
    lo = float(np.exp(cph.params_["snp"] - 1.96 * se))
    hi = float(np.exp(cph.params_["snp"] + 1.96 * se))
    z = float(cph.params_["snp"] / se) if se > 0 else np.nan
    from scipy import stats
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan

    out.update({
        "formula": formula,
        "hr_per_allele": hr,
        "hr_lo95": lo,
        "hr_hi95": hi,
        "coef_log_hr": float(cph.params_["snp"]),
        "se": se,
        "z": z,
        "p": p,
        "concordance": float(cph.concordance_index_),
        "geno_mean": float(df["snp"].mean()),
        "geno_0": int((df["snp"] == 0).sum()),
        "geno_1": int((df["snp"] == 1).sum()),
        "geno_2": int((df["snp"] == 2).sum()),
    })
    return out


def _merge_replication(fhs_df: pd.DataFrame, whi_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    fhs = fhs_df.rename(columns={
        c: f"fhs_{c}" for c in fhs_df.columns if c != "snp"
    })
    whi = whi_df.rename(columns={
        c: f"whi_{c}" for c in whi_df.columns if c != "snp"
    })
    merged = fhs.merge(whi, on="snp", how="outer")
    rows = []
    for _, r in merged.iterrows():
        same_sign = np.nan
        replicated = False
        if pd.notna(r.get("fhs_coef_log_hr")) and pd.notna(r.get("whi_coef_log_hr")):
            same_sign = bool(np.sign(r["fhs_coef_log_hr"]) == np.sign(r["whi_coef_log_hr"]))
            replicated = bool(
                same_sign
                and pd.notna(r.get("fhs_p")) and r["fhs_p"] < 0.05
                and pd.notna(r.get("whi_p")) and r["whi_p"] < 0.05
            )
        meta_b = meta_se = meta_p = np.nan
        if (
            pd.notna(r.get("fhs_coef_log_hr")) and pd.notna(r.get("fhs_se"))
            and pd.notna(r.get("whi_coef_log_hr")) and pd.notna(r.get("whi_se"))
        ):
            meta_b, meta_se, meta_p = fixed_effect_meta(
                float(r["fhs_coef_log_hr"]), float(r["fhs_se"]),
                float(r["whi_coef_log_hr"]), float(r["whi_se"]),
            )
        row = r.to_dict()
        row["same_sign"] = same_sign
        row["replicated"] = replicated
        row["meta_coef_log_hr"] = meta_b
        row["meta_se"] = meta_se
        row["meta_p"] = meta_p
        row["meta_hr"] = float(np.exp(meta_b)) if np.isfinite(meta_b) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("meta_p", kind="mergesort")
    out.to_csv(out_dir / "snp_cox_replication.csv", index=False)
    write_json(out_dir / "snp_cox_replication_summary.json", {
        "n_snps": int(len(out)),
        "n_replicated_strict": int(out["replicated"].sum()) if "replicated" in out.columns else 0,
        "n_same_sign": int(out["same_sign"].sum()) if "same_sign" in out.columns else 0,
        "rows": out[[
            "snp", "fhs_hr_per_allele", "fhs_p", "whi_hr_per_allele", "whi_p",
            "same_sign", "replicated", "meta_hr", "meta_p",
        ]].to_dict(orient="records") if not out.empty else [],
    })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Per-SNP Cox PH mortality models (FHS or WHI)")
    p.add_argument("--cohort", type=str, default="FHS", choices=["FHS", "WHI"])
    p.add_argument("--snp-csv", type=str, default="",
                   help="CSV with SNP IDs (default column: snp)")
    p.add_argument("--snp-col", type=str, default="snp")
    p.add_argument("--snp-list", nargs="*", default=[],
                   help="Explicit SNP IDs on command line")
    p.add_argument("--top-n", type=int, default=0, help="Limit to first N SNPs from CSV")
    p.add_argument("--fhs-npz", type=str,
                   default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--whi-npz", type=str,
                   default="vae_cox_cache/bundles/WHI_raw_WHI_methylation_with_snp_merged_1milfeatures_combined_training_WHI_methylation_with_snp_merged_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-snp-txt", type=str,
                   default="FHS_methylation_with_snp_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--whi-snp-txt", type=str,
                   default="WHI_methylation_with_snp_merged_1milfeatures_snp_genotypes.snp_columns.txt")
    p.add_argument("--fhs-meta-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-meta-pq", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--whi-id-col", type=str, default="sample_ID")
    p.add_argument("--out-dir", type=str, default="feature_importance/gwas")
    p.add_argument("--penalizer", type=float, default=0.01)
    p.add_argument("--min-n", type=int, default=0, help="Min samples (0 = cohort default)")
    p.add_argument("--min-events", type=int, default=0, help="Min events (0 = cohort default)")
    p.add_argument("--merge-fhs-csv", type=str, default="",
                   help="FHS Cox results CSV to merge after WHI run (replication table)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = args.cohort.upper()

    if cohort == "FHS":
        npz_path = Path(args.fhs_npz)
        snp_txt = Path(args.fhs_snp_txt)
        meta_pq = Path(args.fhs_meta_pq)
        id_col = args.fhs_id_col
        min_n = args.min_n or 50
        min_events = args.min_events or 8
    else:
        npz_path = Path(args.whi_npz)
        snp_txt = Path(args.whi_snp_txt)
        meta_pq = Path(args.whi_meta_pq)
        id_col = args.whi_id_col
        min_n = args.min_n or 30
        min_events = args.min_events or 5

    snp_names = _load_snp_list(args)
    print(f"[cox-{cohort}] SNPs to test: {len(snp_names)}")

    z = np.load(npz_path, mmap_mode="r")
    time = z["time"].astype(np.float64)
    event = z["event"].astype(np.int32)
    age, sex = read_age_sex(meta_pq, id_col)
    print(f"[cox-{cohort}] n={len(time)}  events={int(event.sum())}")

    cols, resolved = resolve_snp_columns(snp_names, snp_txt)
    missing = set(snp_names) - set(resolved.astype(str))
    if missing:
        print(f"  WARNING: {len(missing)} SNPs not found in genotype columns: {sorted(missing)[:5]}")
    if resolved.size == 0:
        raise SystemExit("No SNPs resolved to genotype columns.")

    G = load_genotype_columns(npz_path, cols)
    results = []
    for j, snp in enumerate(resolved):
        row = _fit_one_cox(
            str(snp), G[:, j], time, event, age, sex, args.penalizer,
            min_n=min_n, min_events=min_events,
        )
        row["cohort"] = cohort
        results.append(row)
        err = row.get("error", "")
        hr = row.get("hr_per_allele", "NA")
        pv = row.get("p", "NA")
        print(f"  [{j+1}/{len(resolved)}] {snp}  HR={hr}  p={pv}" + (f"  ({err})" if err else ""))

    df = pd.DataFrame(results)
    tag = cohort.lower()
    out_csv = out_dir / f"{tag}_snp_cox_results.csv"
    df.to_csv(out_csv, index=False)

    ok = df[df["error"].isna()] if "error" in df.columns else df
    summary = {
        "cohort": cohort,
        "n_requested": len(snp_names),
        "n_resolved": int(len(resolved)),
        "n_missing": len(missing),
        "n_fitted": int(len(ok)),
        "n_events_total": int(event.sum()),
        "n_significant_p05": int((ok["p"] < 0.05).sum()) if "p" in ok.columns else 0,
        "top_hits": ok.nsmallest(10, "p")[
            ["snp", "hr_per_allele", "hr_lo95", "hr_hi95", "p", "n_events"]
        ].to_dict(orient="records") if "p" in ok.columns and not ok.empty else [],
    }
    write_json(out_dir / f"{tag}_snp_cox_summary.json", summary)
    print(f"[cox-{cohort}] Wrote {out_csv}  fitted={summary['n_fitted']}/{summary['n_resolved']}")

    merge_path = args.merge_fhs_csv or (out_dir / "fhs_snp_cox_results.csv")
    if cohort == "WHI" and Path(merge_path).exists():
        fhs_df = pd.read_csv(merge_path)
        rep = _merge_replication(fhs_df, df, out_dir)
        print(f"[cox-replication] Wrote snp_cox_replication.csv  strict_replicated={int(rep['replicated'].sum())}")


if __name__ == "__main__":
    main()
