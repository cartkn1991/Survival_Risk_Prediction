#!/usr/bin/env python3
"""Cross-tab top log_h-residual vs mortality GWAS hits with LD-window matching.

Reads:
  - feature_importance/gwas/discovery_fhs_results.csv   (log_h residual GWAS)
  - feature_importance/gwas/mortality_fhs_results.csv   (mortality GWAS)

Outputs (under --out-dir):
  - cross_compare_leads_logh.csv        clumped log_h leads
  - cross_compare_leads_mortality.csv   clumped mortality leads
  - cross_compare_union.csv             union table with both phenotypes
  - cross_compare_pairs.csv             log_h <-> mortality window matches
  - cross_compare_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from gwas.gwas_common import clump_leads, write_json


def _lookup(df: pd.DataFrame, snp: str) -> Dict[str, float | str]:
    row = df.loc[df["snp"] == snp]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "beta": float(r["beta"]),
        "se": float(r["se"]),
        "p": float(r["p"]),
        "maf": float(r["maf"]),
        "chrom": int(r["chrom"]),
        "pos": int(r["pos"]),
    }


def _window_match(
    leads_a: pd.DataFrame,
    leads_b: pd.DataFrame,
    *,
    window_bp: int,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, a in leads_a.iterrows():
        ch_a, pos_a = int(a["chrom"]), int(a["pos"])
        snp_a = str(a["snp"])
        for _, b in leads_b.iterrows():
            ch_b, pos_b = int(b["chrom"]), int(b["pos"])
            snp_b = str(b["snp"])
            if ch_a != ch_b:
                continue
            dist = abs(pos_a - pos_b)
            if snp_a == snp_b:
                match_type = "exact"
            elif dist <= window_bp:
                match_type = f"window_{window_bp // 1000}kb"
            else:
                continue
            same_sign = np.sign(a["beta"]) == np.sign(b["beta"])
            rows.append({
                f"snp_{label_a}": snp_a,
                f"beta_{label_a}": float(a["beta"]),
                f"p_{label_a}": float(a["p"]),
                f"snp_{label_b}": snp_b,
                f"beta_{label_b}": float(b["beta"]),
                f"p_{label_b}": float(b["p"]),
                "chrom": ch_a,
                "pos_a": pos_a,
                "pos_b": pos_b,
                "dist_bp": dist,
                "match_type": match_type,
                "same_sign": bool(same_sign),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["match_type", "p_logh", "p_mortality"], kind="mergesort")
    return out.drop_duplicates(subset=[f"snp_{label_a}", f"snp_{label_b}"], keep="first").reset_index(drop=True)


def _build_union(
    logh_leads: pd.DataFrame,
    mort_leads: pd.DataFrame,
    logh_full: pd.DataFrame,
    mort_full: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    snps: set[str] = set(logh_leads["snp"].astype(str)) | set(mort_leads["snp"].astype(str))
    if not pairs.empty:
        snps |= set(pairs["snp_logh"].astype(str)) | set(pairs["snp_mortality"].astype(str))

    rows = []
    for snp in sorted(snps):
        lh = _lookup(logh_full, snp)
        mo = _lookup(mort_full, snp)
        chrom = lh.get("chrom", mo.get("chrom", 26))
        pos = lh.get("pos", mo.get("pos", -1))
        in_logh = snp in set(logh_leads["snp"].astype(str))
        in_mort = snp in set(mort_leads["snp"].astype(str))
        source = []
        if in_logh:
            source.append("logh_lead")
        if in_mort:
            source.append("mortality_lead")
        if not source:
            source.append("window_partner")

        same_sign = np.nan
        if lh and mo:
            same_sign = bool(np.sign(lh["beta"]) == np.sign(mo["beta"]))

        rows.append({
            "snp": snp,
            "chrom": chrom,
            "pos": pos,
            "source": ";".join(source),
            "logh_beta": lh.get("beta", np.nan),
            "logh_p": lh.get("p", np.nan),
            "logh_maf": lh.get("maf", np.nan),
            "mortality_beta": mo.get("beta", np.nan),
            "mortality_p": mo.get("p", np.nan),
            "mortality_maf": mo.get("maf", np.nan),
            "same_sign_both": same_sign,
        })

    df = pd.DataFrame(rows)
    df["min_p"] = df[["logh_p", "mortality_p"]].min(axis=1)
    return df.sort_values(["min_p", "chrom", "pos"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-tab log_h vs mortality GWAS leads")
    p.add_argument("--logh-csv", default="feature_importance/gwas/discovery_fhs_results.csv")
    p.add_argument("--mortality-csv", default="feature_importance/gwas/mortality_fhs_results.csv")
    p.add_argument("--out-dir", default="feature_importance/gwas")
    p.add_argument("--logh-p", type=float, default=1e-5, help="Clumping threshold for log_h leads")
    p.add_argument("--mortality-p", type=float, default=1e-5, help="Clumping threshold for mortality leads")
    p.add_argument("--clump-bp", type=int, default=1_000_000)
    p.add_argument("--ld-window-bp", type=int, default=250_000,
                   help="Physical distance for LD-window matching between lead lists")
    p.add_argument("--max-leads", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[cross] Loading GWAS result tables ...")
    logh = pd.read_csv(args.logh_csv)
    mort = pd.read_csv(args.mortality_csv)

    logh_leads = clump_leads(logh, p_max=args.logh_p, window_bp=args.clump_bp, max_leads=args.max_leads)
    mort_leads = clump_leads(mort, p_max=args.mortality_p, window_bp=args.clump_bp, max_leads=args.max_leads)
    logh_leads.to_csv(out_dir / "cross_compare_leads_logh.csv", index=False)
    mort_leads.to_csv(out_dir / "cross_compare_leads_mortality.csv", index=False)
    print(f"  log_h leads: {len(logh_leads)}  mortality leads: {len(mort_leads)}")

    exact = set(logh_leads["snp"].astype(str)) & set(mort_leads["snp"].astype(str))
    pairs = _window_match(
        logh_leads, mort_leads,
        window_bp=args.ld_window_bp,
        label_a="logh", label_b="mortality",
    )
    pairs.to_csv(out_dir / "cross_compare_pairs.csv", index=False)

    union = _build_union(logh_leads, mort_leads, logh, mort, pairs)
    union.to_csv(out_dir / "cross_compare_union.csv", index=False)

    n_window = int((pairs["match_type"] != "exact").sum()) if not pairs.empty else 0
    n_same_sign = int(pairs["same_sign"].sum()) if not pairs.empty else 0
    summary = {
        "n_logh_leads": int(len(logh_leads)),
        "n_mortality_leads": int(len(mort_leads)),
        "n_exact_overlap": len(exact),
        "exact_snps": sorted(exact),
        "n_window_pairs": int(len(pairs)),
        "n_window_non_exact": n_window,
        "n_pairs_same_sign": n_same_sign,
        "ld_window_bp": args.ld_window_bp,
        "logh_p_threshold": args.logh_p,
        "mortality_p_threshold": args.mortality_p,
        "top_convergent": union.loc[
            union["logh_p"].notna() & union["mortality_p"].notna()
        ].nsmallest(10, "min_p")[["snp", "logh_beta", "logh_p", "mortality_beta", "mortality_p", "same_sign_both"]].to_dict(orient="records"),
    }
    write_json(out_dir / "cross_compare_summary.json", summary)

    print(f"[cross] exact overlap: {len(exact)}")
    print(f"[cross] window pairs ({args.ld_window_bp // 1000} kb): {len(pairs)}")
    print(f"  wrote {out_dir / 'cross_compare_union.csv'}")
    print(f"  wrote {out_dir / 'cross_compare_pairs.csv'}")
    print(f"  wrote {out_dir / 'cross_compare_summary.json'}")


if __name__ == "__main__":
    main()
