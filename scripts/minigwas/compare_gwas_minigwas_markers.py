#!/usr/bin/env python3
"""Compare miniGWAS suggestive SNPs against internal GWAS tables.

Outputs:
  - minigwas_gwas_merged_tiered.csv
  - discovered_markers_strict.csv
  - discovered_markers_moderate.csv
  - discovered_markers_exploratory.csv
  - discovered_markers_summary_tiered.json
  - discovered_markers_README.md
  - discovered_markers_concordance_tiered.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sign(x: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _normalize_snp(s: object) -> str:
    if s is None:
        return ""
    return str(s).strip().upper()


def _class_from_beta(beta: float) -> str:
    if not np.isfinite(beta):
        return "unknown"
    return "accelerated_marker" if beta > 0 else "decelerated_marker"


def _pick_support_source(r: pd.Series) -> str:
    # Tier-aware support summary: use moderate-level evidence (p<0.05).
    mort_sig = bool(r.get("mortality_sig_005", False))
    logh_sig = bool(r.get("logh_sig_005", False))
    mort_conc = bool(r.get("concordant_mortality", False))
    logh_conc = bool(r.get("concordant_logh", False))
    if mort_sig and mort_conc and logh_sig and logh_conc:
        return "both_concordant"
    if mort_sig and mort_conc:
        return "mortality_concordant"
    if logh_sig and logh_conc:
        return "logh_concordant"
    if mort_sig and logh_sig:
        return "both_sig_mixed_sign"
    if mort_sig:
        return "mortality_sig_only_discordant"
    if logh_sig:
        return "logh_sig_only_discordant"
    return "none"


def _plot_forest_tiered(markers_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, max(5.0, 0.28 * max(9, len(markers_df)))))
    if markers_df.empty:
        ax.text(0.5, 0.5, "No tiered discovered markers", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    dfp = markers_df.copy()
    dfp["log_or_acc"] = np.log(np.clip(dfp["or_acc"].astype(float), 1e-12, None))
    dfp = dfp.sort_values("log_or_acc", ascending=True).reset_index(drop=True)
    y = np.arange(len(dfp))
    colors = np.where(dfp["marker_class"].eq("accelerated_marker"), "#c0392b", "#1f77b4")
    ax.scatter(dfp["log_or_acc"], y, c=colors, s=38, alpha=0.95, linewidths=0)
    ax.axvline(0.0, color="#2c3e50", lw=1.0, ls="--")

    # Add support annotations at right margin
    x_max = float(np.nanmax(dfp["log_or_acc"])) if len(dfp) else 1.0
    x_min = float(np.nanmin(dfp["log_or_acc"])) if len(dfp) else -1.0
    span = max(1e-6, x_max - x_min)
    x_annot = x_max + 0.1 * span
    for i, r in dfp.iterrows():
        mort = "M+" if bool(r.get("mortality_sig_005", False)) else "M-"
        logh = "L+" if bool(r.get("logh_sig_005", False)) else "L-"
        conc_m = "Cm+" if bool(r.get("concordant_mortality", False)) else "Cm-"
        conc_l = "Cl+" if bool(r.get("concordant_logh", False)) else "Cl-"
        tier = str(r.get("marker_tier", ""))
        ax.text(x_annot, i, f"{tier} | {mort} {logh} | {conc_m} {conc_l}", va="center", fontsize=9, color="#2c3e50")

    ax.set_yticks(y)
    ax.set_yticklabels(dfp["snp"].astype(str), fontsize=9)
    ax.set_xlabel("miniGWAS effect size: log(OR_acc)")
    ax.set_title("Tiered discovered markers: miniGWAS effect and GWAS concordance", fontsize=14)
    ax.grid(True, axis="x", alpha=0.28)
    ax.set_xlim(x_min - 0.08 * span, x_annot + 0.45 * span)
    ax.invert_yaxis()

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#c0392b", label="accelerated_marker"),
        plt.Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="decelerated_marker"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_readme(path: Path, summary: dict) -> None:
    txt = f"""# Discovered markers (tiered)

This table compares miniGWAS outlier SNP signals against two internal GWAS phenotypes:
- mortality GWAS
- log_h residual GWAS

Strict tier:
1) miniGWAS suggestive (`p < 0.05`)
2) at least one internal GWAS support (`p < 1e-5` in mortality or log_h)
3) same effect direction between miniGWAS and the supporting GWAS (`sign(beta)` concordant)

Moderate tier:
1) miniGWAS suggestive (`p < 0.05`)
2) at least one internal GWAS support (`p < 0.05` in mortality or log_h)
3) same effect direction between miniGWAS and the supporting GWAS

Exploratory tier:
1) miniGWAS suggestive (`p < 0.05`)
2) at least one internal GWAS support (`p < 0.10` in mortality or log_h)
3) same effect direction between miniGWAS and the supporting GWAS

Marker class:
- `accelerated_marker`: miniGWAS `beta > 0` (`OR_acc > 1`)
- `decelerated_marker`: miniGWAS `beta < 0` (`OR_acc < 1`)

Counts:
- suggestive miniGWAS rows analyzed: {summary['n_suggestive_rows']}
- strict discovered markers: {summary['tiers']['strict']['n_total']}
- moderate discovered markers: {summary['tiers']['moderate']['n_total']}
- exploratory discovered markers: {summary['tiers']['exploratory']['n_total']}

Interpretation:
A strict marker indicates that a model-selected SNP is not only predictive in miniGWAS outlier contrasts,
but also supported by independent internal GWAS phenotype association with direction consistency.
"""
    path.write_text(txt, encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=None, help="miniGWAS output directory")
    p.add_argument("--mortality-csv", type=Path, default=None)
    p.add_argument("--logh-csv", type=Path, default=None)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    root = args.out_dir or Path(__file__).resolve().parent
    suggestive_csv = root / "minigwas_report_suggestive_p_lt_0.05.csv"
    minigwas_full_csv = root / "minigwas_results.csv"
    mortality_csv = args.mortality_csv or repo_root / "feature_importance" / "gwas" / "mortality_fhs_results.csv"
    logh_csv = args.logh_csv or repo_root / "feature_importance" / "gwas" / "discovery_fhs_results.csv"

    out_merged = root / "minigwas_gwas_merged_tiered.csv"
    out_strict = root / "discovered_markers_strict.csv"
    out_moderate = root / "discovered_markers_moderate.csv"
    out_exploratory = root / "discovered_markers_exploratory.csv"
    out_summary = root / "discovered_markers_summary_tiered.json"
    out_readme = root / "discovered_markers_README.md"
    out_plot = root / "discovered_markers_concordance_tiered.png"

    suggestive = pd.read_csv(suggestive_csv, low_memory=False)
    mini_full = pd.read_csv(minigwas_full_csv, low_memory=False)
    mort = pd.read_csv(mortality_csv, low_memory=False)
    logh = pd.read_csv(logh_csv, low_memory=False)

    # Harmonize SNP key
    for df in (suggestive, mini_full, mort, logh):
        df["snp_key"] = df["snp"].map(_normalize_snp)

    # Keep suggestive universe (545 rows) and enrich from full miniGWAS columns.
    mini_cols = [
        "snp_key", "n_used", "maf", "or_acc", "beta", "se", "z", "p", "q",
        "sig_age", "gene_ensembl", "chrom", "pos",
    ]
    mini_add = mini_full[mini_cols].drop_duplicates("snp_key", keep="first")
    merged = suggestive[["snp", "snp_key"]].merge(mini_add, on="snp_key", how="left", suffixes=("", "_mini"))

    # Merge mortality/logh GWAS
    mort_add = mort[["snp_key", "beta", "se", "p", "maf", "chrom", "pos"]].rename(
        columns={
            "beta": "mortality_beta", "se": "mortality_se", "p": "mortality_p",
            "maf": "mortality_maf", "chrom": "mortality_chrom", "pos": "mortality_pos",
        }
    )
    logh_add = logh[["snp_key", "beta", "se", "p", "maf", "chrom", "pos"]].rename(
        columns={
            "beta": "logh_beta", "se": "logh_se", "p": "logh_p",
            "maf": "logh_maf", "chrom": "logh_chrom", "pos": "logh_pos",
        }
    )
    merged = merged.merge(mort_add.drop_duplicates("snp_key", keep="first"), on="snp_key", how="left")
    merged = merged.merge(logh_add.drop_duplicates("snp_key", keep="first"), on="snp_key", how="left")

    # Classify
    p_cut_strict = 1e-5
    p_cut_moderate = 0.05
    p_cut_exploratory = 0.10
    merged["mini_beta_sign"] = merged["beta"].astype(float).map(_sign)
    merged["mortality_beta_sign"] = merged["mortality_beta"].astype(float).map(_sign)
    merged["logh_beta_sign"] = merged["logh_beta"].astype(float).map(_sign)
    merged["concordant_mortality"] = (
        merged["mini_beta_sign"].ne(0)
        & merged["mortality_beta_sign"].ne(0)
        & merged["mini_beta_sign"].eq(merged["mortality_beta_sign"])
    )
    merged["concordant_logh"] = (
        merged["mini_beta_sign"].ne(0)
        & merged["logh_beta_sign"].ne(0)
        & merged["mini_beta_sign"].eq(merged["logh_beta_sign"])
    )
    merged["mortality_sig_1e5"] = merged["mortality_p"].astype(float) < p_cut_strict
    merged["logh_sig_1e5"] = merged["logh_p"].astype(float) < p_cut_strict
    merged["mortality_sig_005"] = merged["mortality_p"].astype(float) < p_cut_moderate
    merged["logh_sig_005"] = merged["logh_p"].astype(float) < p_cut_moderate
    merged["mortality_sig_010"] = merged["mortality_p"].astype(float) < p_cut_exploratory
    merged["logh_sig_010"] = merged["logh_p"].astype(float) < p_cut_exploratory
    merged["mini_suggestive"] = merged["p"].astype(float) < 0.05
    merged["support_source"] = merged.apply(_pick_support_source, axis=1)
    merged["strict_discovered"] = merged["mini_suggestive"] & (
        (merged["mortality_sig_1e5"] & merged["concordant_mortality"])
        | (merged["logh_sig_1e5"] & merged["concordant_logh"])
    )
    merged["moderate_discovered"] = merged["mini_suggestive"] & (
        (merged["mortality_sig_005"] & merged["concordant_mortality"])
        | (merged["logh_sig_005"] & merged["concordant_logh"])
    )
    merged["exploratory_discovered"] = merged["mini_suggestive"] & (
        (merged["mortality_sig_010"] & merged["concordant_mortality"])
        | (merged["logh_sig_010"] & merged["concordant_logh"])
    )
    merged["marker_class"] = merged["beta"].astype(float).map(_class_from_beta)
    merged["marker_tier"] = np.where(
        merged["strict_discovered"], "strict",
        np.where(merged["moderate_discovered"], "moderate",
                 np.where(merged["exploratory_discovered"], "exploratory", "none"))
    )

    # Cross-GWAS status for rows where both tables are significant
    merged["both_gwas_sig_1e5"] = merged["mortality_sig_1e5"] & merged["logh_sig_1e5"]
    merged["both_gwas_same_sign"] = (
        merged["both_gwas_sig_1e5"]
        & merged["mortality_beta_sign"].ne(0)
        & merged["logh_beta_sign"].ne(0)
        & merged["mortality_beta_sign"].eq(merged["logh_beta_sign"])
    )

    # Persist outputs
    merged = merged.sort_values(
        ["strict_discovered", "moderate_discovered", "exploratory_discovered", "p"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    merged.to_csv(out_merged, index=False)

    strict = merged[merged["strict_discovered"]].copy().reset_index(drop=True)
    moderate = merged[merged["moderate_discovered"]].copy().reset_index(drop=True)
    exploratory = merged[merged["exploratory_discovered"]].copy().reset_index(drop=True)
    strict.to_csv(out_strict, index=False)
    moderate.to_csv(out_moderate, index=False)
    exploratory.to_csv(out_exploratory, index=False)

    def _tier_stats(df_t: pd.DataFrame) -> dict:
        if df_t.empty:
            return {"n_total": 0, "n_accelerated": 0, "n_decelerated": 0}
        return {
            "n_total": int(len(df_t)),
            "n_accelerated": int((df_t["marker_class"] == "accelerated_marker").sum()),
            "n_decelerated": int((df_t["marker_class"] == "decelerated_marker").sum()),
        }

    summary = {
        "n_suggestive_rows": int(len(merged)),
        "tiers": {
            "strict": _tier_stats(strict),
            "moderate": _tier_stats(moderate),
            "exploratory": _tier_stats(exploratory),
        },
        "support_source_counts_strict": strict["support_source"].value_counts(dropna=False).to_dict(),
        "support_source_counts_moderate": moderate["support_source"].value_counts(dropna=False).to_dict(),
        "support_source_counts_exploratory": exploratory["support_source"].value_counts(dropna=False).to_dict(),
        "direction_concordance": {
            "strict_mortality_concordant": int((strict["mortality_sig_1e5"] & strict["concordant_mortality"]).sum()) if len(strict) else 0,
            "strict_logh_concordant": int((strict["logh_sig_1e5"] & strict["concordant_logh"]).sum()) if len(strict) else 0,
            "moderate_mortality_concordant": int((moderate["mortality_sig_005"] & moderate["concordant_mortality"]).sum()) if len(moderate) else 0,
            "moderate_logh_concordant": int((moderate["logh_sig_005"] & moderate["concordant_logh"]).sum()) if len(moderate) else 0,
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _build_readme(out_readme, summary)
    tiered_for_plot = merged[merged["marker_tier"] != "none"].copy()
    _plot_forest_tiered(tiered_for_plot, out_plot)

    print(f"wrote {out_merged}")
    print(f"wrote {out_strict}")
    print(f"wrote {out_moderate}")
    print(f"wrote {out_exploratory}")
    print(f"wrote {out_summary}")
    print(f"wrote {out_readme}")
    print(f"wrote {out_plot}")
    print(f"strict discovered markers: {len(strict)}")
    print(f"moderate discovered markers: {len(moderate)}")
    print(f"exploratory discovered markers: {len(exploratory)}")


if __name__ == "__main__":
    main()

