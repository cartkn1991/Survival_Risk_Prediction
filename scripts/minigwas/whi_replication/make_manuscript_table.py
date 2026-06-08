#!/usr/bin/env python3
"""Create manuscript-ready WHI replication tables for moderate markers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_hr(hr: float, lo: float, hi: float) -> str:
    if not (np.isfinite(hr) and np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"{hr:.2f} ({lo:.2f}-{hi:.2f})"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    root = Path(__file__).resolve().parent
    minigwas_dir = root.parent

    rep = pd.read_csv(root / "snp_cox_replication.csv", low_memory=False)
    moderate = pd.read_csv(minigwas_dir / "discovered_markers_moderate.csv", low_memory=False)
    gene_cache = pd.read_csv(repo_root / "feature_importance" / "annot" / "snp_gene_cache.csv", low_memory=False)
    gene_cache["snp"] = gene_cache["feature_id"].astype(str)

    keep_cols = [
        "snp", "marker_class", "support_source", "or_acc", "p", "q", "gene_ensembl",
    ]
    mod_sub = moderate[keep_cols].copy()
    merged = rep.merge(mod_sub, on="snp", how="left")

    # Fill gene labels from global cache if missing in moderate table.
    merged = merged.merge(gene_cache[["snp", "gene_ensembl"]].rename(columns={"gene_ensembl": "gene_ensembl_cache"}), on="snp", how="left")
    g1 = merged["gene_ensembl"].astype(str).replace({"nan": "", "None": ""})
    g2 = merged["gene_ensembl_cache"].astype(str).replace({"nan": "", "None": ""})
    merged["gene_symbol"] = np.where(g1.str.len() > 0, g1, g2)
    merged["gene_symbol"] = merged["gene_symbol"].replace({"": "NA"})

    merged["miniGWAS_OR_acc"] = merged["or_acc"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "")
    merged["miniGWAS_p"] = merged["p"].map(_fmt_p)
    merged["FHS_HR_95CI"] = merged.apply(lambda r: _fmt_hr(r["fhs_hr_per_allele"], r["fhs_hr_lo95"], r["fhs_hr_hi95"]), axis=1)
    merged["FHS_p"] = merged["fhs_p"].map(_fmt_p)
    merged["WHI_HR_95CI"] = merged.apply(lambda r: _fmt_hr(r["whi_hr_per_allele"], r["whi_hr_lo95"], r["whi_hr_hi95"]), axis=1)
    merged["WHI_p"] = merged["whi_p"].map(_fmt_p)
    merged["Direction_FHS_vs_WHI"] = np.where(merged["same_sign"].fillna(False), "Concordant", "Discordant")
    merged["Replicated_nominal"] = np.where(merged["replicated"].fillna(False), "Yes", "No")

    out_cols = [
        "snp", "gene_symbol", "marker_class", "support_source",
        "miniGWAS_OR_acc", "miniGWAS_p",
        "FHS_HR_95CI", "FHS_p",
        "WHI_HR_95CI", "WHI_p",
        "Direction_FHS_vs_WHI", "Replicated_nominal",
    ]
    out = merged[out_cols].copy()
    out = out.sort_values(
        by=["Replicated_nominal", "WHI_p"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    out_csv = root / "manuscript_table_whi_replication_moderate_markers.csv"
    out_md = root / "manuscript_table_whi_replication_moderate_markers.md"
    out_rep2_csv = root / "manuscript_table_whi_replicated_only.csv"

    out.to_csv(out_csv, index=False)
    out[out["Replicated_nominal"] == "Yes"].to_csv(out_rep2_csv, index=False)

    # Markdown table (top section with replicated first, then all).
    md_lines = []
    md_lines.append("# WHI replication of moderate candidate markers\n")
    md_lines.append("Replication criterion: FHS p<0.05, WHI p<0.05, and concordant effect direction.\n")
    rep2 = out[out["Replicated_nominal"] == "Yes"].copy()
    md_lines.append(f"Replicated markers: {len(rep2)} / {len(out)}\n")
    md_lines.append("## Replicated markers (nominal)\n")
    md_lines.append(rep2.to_markdown(index=False))
    md_lines.append("\n## Full moderate marker table (97)\n")
    md_lines.append(out.to_markdown(index=False))
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_rep2_csv}")
    print(f"wrote {out_md}")
    print(f"replicated_nominal={len(rep2)} total={len(out)}")


if __name__ == "__main__":
    main()

