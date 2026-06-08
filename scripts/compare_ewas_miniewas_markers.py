#!/usr/bin/env python3
"""Cross-compare full EWAS vs mini-EWAS suggestive CpGs (tiered)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sign(x: float) -> int:
    if not np.isfinite(x):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def main() -> None:
    root = Path(__file__).resolve().parent
    ewas_dir = root / "ewas"
    mini_dir = root / "miniewas"

    mini = pd.read_csv(mini_dir / "miniewas_results.csv")
    suggestive = mini[mini["p"] < 0.05].copy()
    mort = pd.read_csv(ewas_dir / "mortality_fhs_ewas_results.csv")
    disc = pd.read_csv(ewas_dir / "discovery_fhs_ewas_results.csv")

    merged = suggestive.merge(
        mort[["cpg", "beta", "p"]].rename(columns={"beta": "mortality_beta", "p": "mortality_p"}),
        on="cpg", how="left",
    ).merge(
        disc[["cpg", "beta", "p"]].rename(columns={"beta": "logh_beta", "p": "logh_p"}),
        on="cpg", how="left",
    )

    merged["mini_beta_sign"] = merged["beta"].map(_sign)
    merged["mortality_beta_sign"] = merged["mortality_beta"].map(_sign)
    merged["logh_beta_sign"] = merged["logh_beta"].map(_sign)
    merged["concordant_mortality"] = merged["mini_beta_sign"].eq(merged["mortality_beta_sign"]) & merged["mini_beta_sign"].ne(0)
    merged["concordant_logh"] = merged["mini_beta_sign"].eq(merged["logh_beta_sign"]) & merged["mini_beta_sign"].ne(0)
    merged["mortality_sig_1e5"] = merged["mortality_p"] < 1e-5
    merged["logh_sig_1e5"] = merged["logh_p"] < 1e-5
    merged["mortality_sig_005"] = merged["mortality_p"] < 0.05
    merged["logh_sig_005"] = merged["logh_p"] < 0.05
    merged["marker_class"] = np.where(merged["beta"] > 0, "accelerated_marker", "decelerated_marker")
    merged["strict_discovered"] = (
        (merged["mortality_sig_1e5"] & merged["concordant_mortality"])
        | (merged["logh_sig_1e5"] & merged["concordant_logh"])
    )
    merged["moderate_discovered"] = (
        (merged["mortality_sig_005"] & merged["concordant_mortality"])
        | (merged["logh_sig_005"] & merged["concordant_logh"])
    )

    out_merged = mini_dir / "miniewas_ewas_merged_tiered.csv"
    out_strict = mini_dir / "discovered_cpg_markers_strict.csv"
    out_moderate = mini_dir / "discovered_cpg_markers_moderate.csv"
    out_summary = mini_dir / "discovered_cpg_markers_summary.json"

    merged.to_csv(out_merged, index=False)
    merged[merged["strict_discovered"]].to_csv(out_strict, index=False)
    merged[merged["moderate_discovered"]].to_csv(out_moderate, index=False)

    summary = {
        "n_mini_suggestive_p_lt_0_05": int(len(suggestive)),
        "n_strict_discovered": int(merged["strict_discovered"].sum()),
        "n_moderate_discovered": int(merged["moderate_discovered"].sum()),
        "n_accelerated_moderate": int((merged["moderate_discovered"] & (merged["marker_class"] == "accelerated_marker")).sum()),
        "n_decelerated_moderate": int((merged["moderate_discovered"] & (merged["marker_class"] == "decelerated_marker")).sum()),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_merged}")
    print(f"strict={summary['n_strict_discovered']} moderate={summary['n_moderate_discovered']}")


if __name__ == "__main__":
    main()
