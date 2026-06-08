#!/usr/bin/env python3
"""Annotate SHAP top-hit CSVs with gene symbols and known associations.

Reuses the heavy-lifting helpers from ``annotate_top_features.py``:

  - CpGs            -> gene symbol (EPIC manifest Annotation.csv) + flags for
                       membership in the 9 epigenetic clocks (Horvath / Hannum /
                       PhenoAge / GrimAgeV2 / DunedinPACE / Zhang2019 / Lin /
                       Weidner / Horvath_SkinBlood).
  - SNPs            -> nearest gene via Ensembl REST (cached) + GWAS catalog
                       rs IDs / hit counts within +-5 kb of the position.

Inputs  : feature_importance/shap/shap_top_cpg_<target>.csv,
          feature_importance/shap/shap_top_snp_<target>.csv  (target in {risk, age})
Outputs : feature_importance/shap/annot/<target>_top_cpg_annotated.csv
          feature_importance/shap/annot/<target>_top_snp_annotated.csv
          feature_importance/shap/annot/snp_gene_cache.csv         (live-extended)
          feature_importance/shap/annot/snp_gwas_cache.csv         (live-extended)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from annotate_top_features import (
    annotate_cpgs,
    annotate_snps,
    load_clock_cpgs,
    load_cpg_to_gene_local,
    fetch_snp_genes,
    fetch_gwas_nearby,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shap-dir", type=str, default="feature_importance/shap")
    p.add_argument("--annot-csv", type=str, default="Annotation.csv")
    p.add_argument("--snp-gene-existing-csv", type=str,
                   default="analysis_out/gene_pathway_pipeline/resources/auto_snp_to_gene_ensembl.csv")
    p.add_argument("--clock-dir", type=str,
                   default="feature_importance/annot/clock_lists",
                   help="Reuse already-downloaded clock CpG CSVs to avoid re-downloading.")
    p.add_argument("--targets", type=str, default="risk,age")
    p.add_argument("--no-ensembl", action="store_true")
    p.add_argument("--no-gwas", action="store_true")
    p.add_argument("--ensembl-sleep", type=float, default=0.05)
    p.add_argument("--gwas-radius-bp", type=int, default=5000)
    p.add_argument("--gwas-sleep", type=float, default=0.05)
    p.add_argument("--out-dir", type=str, default="feature_importance/shap/annot")
    args = p.parse_args()

    shap_dir = Path(args.shap_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    print("[1/4] Loading epigenetic clock CpG lists ...")
    clock_sets = load_clock_cpgs(Path(args.clock_dir))

    print("\n[2/4] Loading CpG -> gene mapping ...")
    cpg_gene = load_cpg_to_gene_local(Path(args.annot_csv))

    print("\n[3/4] Loading SNP -> gene mapping (existing + Ensembl REST) ...")
    existing: Dict[str, str] = {}
    if Path(args.snp_gene_existing_csv).exists():
        df_ex = pd.read_csv(args.snp_gene_existing_csv)
        ex = (df_ex.groupby("feature_id")["gene_symbol"]
                       .apply(lambda s: ";".join(sorted({str(x) for x in s if x})))
                       .to_dict())
        existing = ex
        print(f"  loaded existing SNP map: {len(existing)} SNPs")

    # Union of SHAP top SNPs across both targets
    top_snp_ids: Set[str] = set()
    for t in targets:
        snp_csv = shap_dir / f"shap_top_snp_{t}.csv"
        if snp_csv.exists():
            top_snp_ids.update(pd.read_csv(snp_csv)["feature"].astype(str).tolist())
    print(f"  union of SHAP top SNPs across {targets}: {len(top_snp_ids)}")

    # Existing SNP gene cache (from prior annotate runs)
    snp_cache_path = out_dir / "snp_gene_cache.csv"
    snp_cache: Dict[str, str] = {}
    # Bootstrap from the gradient-method cache if it exists, to save Ensembl traffic
    other_cache = Path("feature_importance/annot/snp_gene_cache.csv")
    if other_cache.exists():
        df_c = pd.read_csv(other_cache)
        snp_cache.update(dict(zip(df_c["feature_id"].astype(str),
                                   df_c["gene_ensembl"].astype(str))))
    if snp_cache_path.exists():
        df_c = pd.read_csv(snp_cache_path)
        snp_cache.update(dict(zip(df_c["feature_id"].astype(str),
                                   df_c["gene_ensembl"].astype(str))))

    new_snp_gene: Dict[str, str] = {}
    if args.no_ensembl:
        new_snp_gene = snp_cache
        print("  --no-ensembl set: only using cached + existing SNP mappings.")
    else:
        already = set(existing) | set(snp_cache)
        to_query = sorted(top_snp_ids - already)
        if to_query:
            print(f"  {len(to_query)} SHAP-top SNPs need Ensembl REST lookup")
        new_snp_gene = fetch_snp_genes(
            sorted(top_snp_ids), snp_cache, snp_cache_path,
            sleep_sec=args.ensembl_sleep,
        )

    gwas_lookup: Dict[str, Dict[str, object]] = {}
    if not args.no_gwas:
        print("\n[3b/4] GWAS catalog nearby SNP lookup ...")
        # Reuse the gradient-method cache as a starting point if present
        gwas_seed = Path("feature_importance/annot/snp_gwas_cache.csv")
        if gwas_seed.exists() and not (out_dir / "snp_gwas_cache.csv").exists():
            print(f"  bootstrapping GWAS cache from {gwas_seed}")
            df = pd.read_csv(gwas_seed)
            df.to_csv(out_dir / "snp_gwas_cache.csv", index=False)
        gwas_lookup = fetch_gwas_nearby(
            sorted(top_snp_ids), out_dir / "snp_gwas_cache.csv",
            radius_bp=args.gwas_radius_bp, sleep_sec=args.gwas_sleep,
        )

    print("\n[4/4] Writing annotated SHAP top-feature CSVs ...")
    for t in targets:
        cpg_csv = shap_dir / f"shap_top_cpg_{t}.csv"
        if cpg_csv.exists():
            annotate_cpgs(cpg_csv, cpg_gene, clock_sets,
                          out_dir / f"{t}_top_cpg_annotated.csv")
        snp_csv = shap_dir / f"shap_top_snp_{t}.csv"
        if snp_csv.exists():
            annotate_snps(snp_csv, existing, new_snp_gene, gwas_lookup,
                          out_dir / f"{t}_top_snp_annotated.csv")

    print(f"\nDone. Annotated SHAP tables in {out_dir}/")


if __name__ == "__main__":
    main()
