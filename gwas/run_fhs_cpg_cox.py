#!/usr/bin/env python3
"""Per-CpG Cox PH models for mortality in FHS or WHI.

For each CpG in a configurable list, fits:

    Surv(time, event) ~ methylation + age [+ sex_female if variable]

Outputs (under --out-dir):
  - {cohort}_cpg_cox_results.csv   HR per methylation unit, 95% CI, p, n, n_events
  - {cohort}_cpg_cox_summary.json
  - cpg_cox_replication.csv        (when --merge-fhs-csv provided with WHI run)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from gwas.gwas_common import (
    fixed_effect_meta,
    load_methylation_columns,
    read_age_sex,
    resolve_cpg_columns,
    write_json,
)

try:
    from lifelines import CoxPHFitter
except ImportError:
    CoxPHFitter = None  # type: ignore


def _load_cpg_list(args: argparse.Namespace) -> List[str]:
    if args.cpg_list:
        return [s.strip() for s in args.cpg_list if s.strip()]
    if args.cpg_csv:
        df = pd.read_csv(args.cpg_csv)
        col = args.cpg_col
        if col not in df.columns:
            for alt in ("cpg", "feature", "feature_id", "CpG"):
                if alt in df.columns:
                    col = alt
                    break
            else:
                raise SystemExit(f"Could not find CpG column in {args.cpg_csv}")
        cpgs = df[col].astype(str).tolist()
        if args.top_n > 0:
            cpgs = cpgs[: args.top_n]
        return cpgs
    raise SystemExit("Provide --cpg-list or --cpg-csv")


def _fit_one_cox(
    cpg: str,
    meth: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    penalizer: float,
    *,
    min_n: int = 50,
    min_events: int = 8,
    min_std: float = 0.01,
) -> dict:
    meth = np.where(np.isfinite(meth), meth, np.nan)
    df = pd.DataFrame({
        "time": time.astype(np.float64),
        "event": event.astype(np.int32),
        "age": age.astype(np.float64),
        "sex": sex.astype(np.float64),
        "cpg": meth.astype(np.float64),
    }).dropna()
    out = {"cpg": cpg, "n": int(len(df)), "n_events": int(df["event"].sum())}
    if CoxPHFitter is None:
        out["error"] = "lifelines not installed"
        return out
    if len(df) < min_n or df["event"].sum() < min_events:
        out["error"] = "insufficient sample or events"
        return out
    if df["cpg"].nunique() < 2 or float(df["cpg"].std()) < min_std:
        out["error"] = "CpG has no variation"
        return out

    cov_cols = ["age"]
    if df["sex"].std() > 1e-8:
        cov_cols.append("sex")
    formula = "cpg + " + " + ".join(cov_cols)

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df, duration_col="time", event_col="event", formula=formula)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    hr = float(np.exp(cph.params_["cpg"]))
    se = float(cph.standard_errors_["cpg"])
    lo = float(np.exp(cph.params_["cpg"] - 1.96 * se))
    hi = float(np.exp(cph.params_["cpg"] + 1.96 * se))
    z = float(cph.params_["cpg"] / se) if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan

    out.update({
        "formula": formula,
        "hr_per_unit": hr,
        "hr_lo95": lo,
        "hr_hi95": hi,
        "coef_log_hr": float(cph.params_["cpg"]),
        "se": se,
        "z": z,
        "p": p,
        "concordance": float(cph.concordance_index_),
        "meth_mean": float(df["cpg"].mean()),
        "meth_std": float(df["cpg"].std()),
    })
    return out


def _merge_replication(fhs_df: pd.DataFrame, whi_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    fhs = fhs_df.rename(columns={c: f"fhs_{c}" for c in fhs_df.columns if c != "cpg"})
    whi = whi_df.rename(columns={c: f"whi_{c}" for c in whi_df.columns if c != "cpg"})
    merged = fhs.merge(whi, on="cpg", how="outer")
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
    out.to_csv(out_dir / "cpg_cox_replication.csv", index=False)
    write_json(out_dir / "cpg_cox_replication_summary.json", {
        "n_cpgs": int(len(out)),
        "n_replicated_strict": int(out["replicated"].sum()) if "replicated" in out.columns else 0,
        "n_same_sign": int(out["same_sign"].sum()) if "same_sign" in out.columns else 0,
        "rows": out[[
            "cpg", "fhs_hr_per_unit", "fhs_p", "whi_hr_per_unit", "whi_p",
            "same_sign", "replicated", "meta_hr", "meta_p",
        ]].to_dict(orient="records") if not out.empty else [],
    })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Per-CpG Cox PH mortality models (FHS or WHI)")
    p.add_argument("--cohort", type=str, default="FHS", choices=["FHS", "WHI"])
    p.add_argument("--cpg-csv", type=str, default="", help="CSV with CpG IDs (default column: cpg)")
    p.add_argument("--cpg-col", type=str, default="cpg")
    p.add_argument("--cpg-list", nargs="*", default=[], help="Explicit CpG IDs on command line")
    p.add_argument("--top-n", type=int, default=0, help="Limit to first N CpGs from CSV")
    p.add_argument("--fhs-npz", type=str,
                   default="vae_cox_cache/bundles/FHS_FHS_methylation_with_snp_1milfeatures_combined_training_FHS_methylation_with_snp_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--whi-npz", type=str,
                   default="vae_cox_cache/bundles/WHI_raw_WHI_methylation_with_snp_merged_1milfeatures_combined_training_WHI_methylation_with_snp_merged_1milfeatures_cpgall_snpall.npz")
    p.add_argument("--fhs-cpg-txt", type=str,
                   default="FHS_methylation_with_snp_1milfeatures_cpg_columns.txt")
    p.add_argument("--whi-cpg-txt", type=str,
                   default="WHI_methylation_with_snp_merged_1milfeatures_cpg_columns.txt")
    p.add_argument("--fhs-meta-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-meta-pq", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument("--fhs-id-col", type=str, default="Share_ID")
    p.add_argument("--whi-id-col", type=str, default="sample_ID")
    p.add_argument("--out-dir", type=str, default="feature_importance/miniewas/whi_replication")
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
        cpg_txt = Path(args.fhs_cpg_txt)
        meta_pq = Path(args.fhs_meta_pq)
        id_col = args.fhs_id_col
        min_n = args.min_n or 50
        min_events = args.min_events or 8
    else:
        npz_path = Path(args.whi_npz)
        cpg_txt = Path(args.whi_cpg_txt)
        meta_pq = Path(args.whi_meta_pq)
        id_col = args.whi_id_col
        min_n = args.min_n or 30
        min_events = args.min_events or 5

    cpg_names = _load_cpg_list(args)
    print(f"[cox-{cohort}] CpGs to test: {len(cpg_names)}")

    z = np.load(npz_path, mmap_mode="r")
    time = z["time"].astype(np.float64)
    event = z["event"].astype(np.int32)
    age, sex = read_age_sex(meta_pq, id_col)
    print(f"[cox-{cohort}] n={len(time)}  events={int(event.sum())}")

    cols, resolved = resolve_cpg_columns(cpg_names, cpg_txt)
    missing = set(cpg_names) - set(resolved.astype(str))
    if missing:
        print(f"  WARNING: {len(missing)} CpGs not found in methylation columns: {sorted(missing)[:5]}")
    if resolved.size == 0:
        raise SystemExit("No CpGs resolved to methylation columns.")

    G = load_methylation_columns(npz_path, cols)
    results = []
    n_resolved = len(resolved)
    for j, cpg in enumerate(resolved):
        row = _fit_one_cox(
            str(cpg), G[:, j], time, event, age, sex, args.penalizer,
            min_n=min_n, min_events=min_events,
        )
        row["cohort"] = cohort
        results.append(row)
        if (j + 1) % 50 == 0 or j + 1 == n_resolved:
            err = row.get("error", "")
            hr = row.get("hr_per_unit", "NA")
            pv = row.get("p", "NA")
            print(f"  [{j+1}/{n_resolved}] {cpg}  HR={hr}  p={pv}" + (f"  ({err})" if err else ""))

    df = pd.DataFrame(results)
    tag = cohort.lower()
    out_csv = out_dir / f"{tag}_cpg_cox_results.csv"
    df.to_csv(out_csv, index=False)

    ok = df[df["error"].isna()] if "error" in df.columns else df
    summary = {
        "cohort": cohort,
        "n_requested": len(cpg_names),
        "n_resolved": int(len(resolved)),
        "n_missing": len(missing),
        "n_fitted": int(len(ok)),
        "n_events_total": int(event.sum()),
        "n_significant_p05": int((ok["p"] < 0.05).sum()) if "p" in ok.columns else 0,
        "top_hits": ok.nsmallest(10, "p")[
            ["cpg", "hr_per_unit", "hr_lo95", "hr_hi95", "p", "n_events"]
        ].to_dict(orient="records") if "p" in ok.columns and not ok.empty else [],
    }
    write_json(out_dir / f"{tag}_cpg_cox_summary.json", summary)
    print(f"[cox-{cohort}] Wrote {out_csv}  fitted={summary['n_fitted']}/{summary['n_resolved']}")

    merge_path = args.merge_fhs_csv or (out_dir / "fhs_cpg_cox_results.csv")
    if cohort == "WHI" and Path(merge_path).exists():
        fhs_df = pd.read_csv(merge_path)
        rep = _merge_replication(fhs_df, df, out_dir)
        print(f"[cox-replication] Wrote cpg_cox_replication.csv  replicated={int(rep['replicated'].sum())}")


if __name__ == "__main__":
    main()
