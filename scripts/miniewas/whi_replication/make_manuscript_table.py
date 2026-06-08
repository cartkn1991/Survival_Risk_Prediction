#!/usr/bin/env python3
"""Create manuscript-ready WHI replication tables for moderate CpG markers."""
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
    root = Path(__file__).resolve().parent
    mini_dir = root.parent

    rep = pd.read_csv(root / "cpg_cox_replication.csv", low_memory=False)
    moderate = pd.read_csv(mini_dir / "discovered_cpg_markers_moderate.csv", low_memory=False)

    keep_cols = [
        "cpg", "marker_class", "or_acc", "p", "q", "gene", "chrom", "pos",
        "mortality_p", "logh_p", "strict_discovered", "moderate_discovered",
    ]
    mod_sub = moderate[[c for c in keep_cols if c in moderate.columns]].copy()
    merged = rep.merge(mod_sub, on="cpg", how="left")

    merged["gene_symbol"] = merged["gene"].astype(str).replace({"nan": "", "None": ""}).replace({"": "NA"})
    merged["miniEWAS_OR_acc"] = merged["or_acc"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "")
    merged["miniEWAS_p"] = merged["p"].map(_fmt_p)
    merged["FHS_HR_95CI"] = merged.apply(
        lambda r: _fmt_hr(r["fhs_hr_per_unit"], r["fhs_hr_lo95"], r["fhs_hr_hi95"]), axis=1
    )
    merged["FHS_p"] = merged["fhs_p"].map(_fmt_p)
    merged["WHI_HR_95CI"] = merged.apply(
        lambda r: _fmt_hr(r["whi_hr_per_unit"], r["whi_hr_lo95"], r["whi_hr_hi95"]), axis=1
    )
    merged["WHI_p"] = merged["whi_p"].map(_fmt_p)
    same = merged["same_sign"].astype("boolean").fillna(False)
    merged["Direction_FHS_vs_WHI"] = np.where(same, "Concordant", "Discordant")
    rep_flag = merged["replicated"].astype("boolean").fillna(False)
    merged["Replicated_nominal"] = np.where(rep_flag, "Yes", "No")

    out_cols = [
        "cpg", "gene_symbol", "chrom", "pos", "marker_class",
        "miniEWAS_OR_acc", "miniEWAS_p",
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

    out_csv = root / "manuscript_table_whi_replication_moderate_cpgs.csv"
    out_md = root / "manuscript_table_whi_replication_moderate_cpgs.md"
    out_rep2_csv = root / "manuscript_table_whi_replicated_cpgs_only.csv"

    out.to_csv(out_csv, index=False)
    out[out["Replicated_nominal"] == "Yes"].to_csv(out_rep2_csv, index=False)

    md_lines = []
    md_lines.append("# WHI replication of moderate candidate CpG markers\n")
    md_lines.append(
        "Replication criterion: FHS p<0.05, WHI p<0.05, and concordant Cox effect direction "
        "(per 1-unit methylation beta).\n"
    )
    rep2 = out[out["Replicated_nominal"] == "Yes"].copy()
    md_lines.append(f"Replicated CpGs: {len(rep2)} / {len(out)}\n")
    md_lines.append("## Replicated CpGs (nominal)\n")
    md_lines.append(rep2.to_markdown(index=False) if not rep2.empty else "_None_\n")
    md_lines.append(f"\n## Full moderate marker table ({len(out)})\n")
    md_lines.append(out.head(50).to_markdown(index=False))
    if len(out) > 50:
        md_lines.append(f"\n_(showing top 50 of {len(out)}; see CSV for full table)_\n")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_rep2_csv}")
    print(f"wrote {out_md}")
    print(f"replicated_nominal={len(rep2)} total={len(out)}")


if __name__ == "__main__":
    main()
