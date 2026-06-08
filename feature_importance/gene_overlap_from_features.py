#!/usr/bin/env python3
"""Map significant-risk and SHAP CpGs/SNPs to HGNC-like genes; count unique and overlap.

Runs up to **three** blocks (each writes a subfolder + one section in ``gene_overlap_summary.json``):

1. **Significance** — CSV with ``feature`` + ``kind`` (default ``significant_risk.csv``).
2. **SHAP full list** — quantile-top SHAP tables (default ``shap_top_cpg_risk.csv`` +
   ``shap_top_snp_risk.csv``). Always separate from intersect so custom intersect paths do not
   replace this group.
3. **SHAP ∩ significance** — optional pair (default ``common_significant_shap_*_risk.csv`` if both
   exist); skipped with ``--skip-shap-intersect`` or missing files.

  - CpGs → ``Annotation.csv`` (``gene_HGNC``, split on ``;,/|``).
  - SNPs → ``--snp-gene-csv`` (``feature_id`` + ``gene_symbol`` or ``gene_ensembl``).

Outputs (default ``feature_importance/gene_overlap_from_features/``)::

  gene_overlap_summary.json
  significant_risk/ / shap_full_risk/ / shap_intersect_significant_risk/
    genes_from_cpg.txt, genes_from_snp.txt, genes_shared_cpg_snp.txt, …

Examples::

  python feature_importance/gene_overlap_from_features.py

  python feature_importance/gene_overlap_from_features.py \\
    --sig-csv feature_importance/significance/significant_either.csv \\
    --label-sig significant_either
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


def split_gene_tokens(cell: object) -> List[str]:
    """Split manifest / SNP annotation cell into HGNC-like tokens."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    genes: List[str] = []
    for p in re.split(r"[;,/\|]", s):
        g = p.strip().upper()
        if len(g) >= 2 and re.match(r"^[A-Z0-9][A-Z0-9\-]*$", g):
            genes.append(g)
    return genes


def genes_from_cpgs(probes: List[str], annot_path: Path) -> Tuple[Set[str], int, int]:
    """Return (gene set, n_probes_input, n_probes_with_any_gene)."""
    if not probes:
        return set(), 0, 0
    pr = set(str(x).strip() for x in probes)
    ann = pd.read_csv(
        annot_path,
        usecols=["probeID", "gene_HGNC"],
        low_memory=False,
        dtype={"probeID": str},
    )
    ann["probeID"] = ann["probeID"].str.strip()
    sub = ann[ann["probeID"].isin(pr)]
    genes: Set[str] = set()
    with_gene = 0
    for _, row in sub.iterrows():
        gs = split_gene_tokens(row.get("gene_HGNC", ""))
        if gs:
            with_gene += 1
        genes.update(gs)
    return genes, len(pr), with_gene


def _snp_gene_column(df: pd.DataFrame) -> str:
    if "gene_symbol" in df.columns and df["gene_symbol"].astype(str).str.strip().ne("").any():
        return "gene_symbol"
    if "gene_ensembl" in df.columns:
        return "gene_ensembl"
    raise SystemExit(f"SNP map needs gene_symbol or gene_ensembl; columns={df.columns.tolist()}")


def genes_from_snps(snp_ids: List[str], map_path: Path) -> Tuple[Set[str], int, int]:
    if not snp_ids:
        return set(), 0, 0
    sid = set(str(x).strip() for x in snp_ids)
    gmap = pd.read_csv(map_path, low_memory=False)
    idcol = "feature_id" if "feature_id" in gmap.columns else gmap.columns[0]
    gmap[idcol] = gmap[idcol].astype(str).str.strip()
    gcol = _snp_gene_column(gmap)
    sub = gmap[gmap[idcol].isin(sid)]
    genes: Set[str] = set()
    with_gene = 0
    for _, row in sub.iterrows():
        gs = split_gene_tokens(row.get(gcol, ""))
        if gs:
            with_gene += 1
        genes.update(gs)
    return genes, len(sid), with_gene


def load_features_by_kind(path: Path) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, low_memory=False)
    if "feature" not in df.columns or "kind" not in df.columns:
        raise SystemExit(f"{path}: need columns feature, kind; got {df.columns.tolist()}")
    k = df["kind"].astype(str).str.lower().str.strip()
    cpg = df.loc[k == "cpg", "feature"].astype(str).str.strip().tolist()
    snp = df.loc[k == "snp", "feature"].astype(str).str.strip().tolist()
    return cpg, snp


def write_gene_sets(out_sub: Path, g_cpg: Set[str], g_snp: Set[str]) -> Dict[str, object]:
    out_sub.mkdir(parents=True, exist_ok=True)
    shared = g_cpg & g_snp
    only_c = g_cpg - g_snp
    only_s = g_snp - g_cpg

    def _write(name: str, s: Set[str]) -> str:
        p = out_sub / name
        p.write_text("\n".join(sorted(s)) + ("\n" if s else ""), encoding="utf-8")
        return str(p.resolve())

    files = {
        "genes_from_cpg": _write("genes_from_cpg.txt", g_cpg),
        "genes_from_snp": _write("genes_from_snp.txt", g_snp),
        "genes_shared_cpg_snp": _write("genes_shared_cpg_snp.txt", shared),
        "genes_only_cpg": _write("genes_only_cpg.txt", only_c),
        "genes_only_snp": _write("genes_only_snp.txt", only_s),
    }
    stats = {
        "n_unique_genes_cpg": len(g_cpg),
        "n_unique_genes_snp": len(g_snp),
        "n_shared_genes_cpg_snp": len(shared),
        "n_union_genes": len(g_cpg | g_snp),
        "n_only_cpg_genes": len(only_c),
        "n_only_snp_genes": len(only_s),
        "files": files,
    }
    return stats


def run_one_block(
    label: str,
    cpg_csv: Path | None,
    snp_csv: Path | None,
    single_csv: Path | None,
    annot_path: Path,
    snp_map_path: Path,
    out_root: Path,
) -> Dict[str, object]:
    if single_csv is not None and single_csv.exists():
        cpg_f, snp_f = load_features_by_kind(single_csv)
        source = str(single_csv.resolve())
    elif cpg_csv is not None and snp_csv is not None and cpg_csv.exists() and snp_csv.exists():
        cpg_f = pd.read_csv(cpg_csv, usecols=["feature"], low_memory=False)["feature"].astype(str).str.strip().tolist()
        snp_f = pd.read_csv(snp_csv, usecols=["feature"], low_memory=False)["feature"].astype(str).str.strip().tolist()
        source = f"{cpg_csv.name} + {snp_csv.name}"
    else:
        return {"error": "missing inputs", "label": label}

    g_cpg, n_in_cpg, n_cpg_mapped = genes_from_cpgs(cpg_f, annot_path)
    g_snp, n_in_snp, n_snp_mapped = genes_from_snps(snp_f, snp_map_path)

    out_sub = out_root / label
    stats = write_gene_sets(out_sub, g_cpg, g_snp)
    stats.update(
        {
            "label": label,
            "source": source,
            "n_features_cpg": n_in_cpg,
            "n_features_snp": n_in_snp,
            "n_cpg_with_gene_annotation": n_cpg_mapped,
            "n_snp_with_gene_annotation": n_snp_mapped,
        }
    )
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Gene overlap from significant-risk vs SHAP feature lists.")
    p.add_argument(
        "--sig-csv",
        type=str,
        default="feature_importance/significance/significant_risk.csv",
        help="CSV with feature + kind (CpG/SNP risk-significant list).",
    )
    p.add_argument(
        "--shap-full-cpg-csv",
        type=str,
        default="feature_importance/shap/shap_top_cpg_risk.csv",
        help="SHAP-only full list: CpG top table (quantile filter).",
    )
    p.add_argument(
        "--shap-full-snp-csv",
        type=str,
        default="feature_importance/shap/shap_top_snp_risk.csv",
        help="SHAP-only full list: SNP top table.",
    )
    p.add_argument(
        "--shap-intersect-cpg-csv",
        type=str,
        default="feature_importance/shap/common_significant_shap_cpg_risk.csv",
        help="SHAP ∩ significance: CpG list (skipped if file missing and --skip-shap-intersect not set).",
    )
    p.add_argument(
        "--shap-intersect-snp-csv",
        type=str,
        default="feature_importance/shap/common_significant_shap_snp_risk.csv",
    )
    p.add_argument(
        "--annot-csv",
        type=str,
        default="Annotation.csv",
        help="EPIC manifest with probeID, gene_HGNC.",
    )
    p.add_argument(
        "--snp-gene-csv",
        type=str,
        default="analysis_out/gene_pathway_pipeline/resources/auto_snp_to_gene_ensembl.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="feature_importance/gene_overlap_from_features",
    )
    p.add_argument(
        "--label-sig",
        type=str,
        default="significant_risk",
        help="Subfolder name for significance-based run.",
    )
    p.add_argument(
        "--label-shap-full",
        type=str,
        default="shap_full_risk",
        help="Subfolder name for SHAP full (top) list.",
    )
    p.add_argument(
        "--label-shap-intersect",
        type=str,
        default="shap_intersect_significant_risk",
        help="Subfolder name for SHAP ∩ significance lists.",
    )
    p.add_argument(
        "--skip-shap-full",
        action="store_true",
        help="Omit SHAP full-list block.",
    )
    p.add_argument(
        "--skip-shap-intersect",
        action="store_true",
        help="Omit SHAP ∩ significance block.",
    )
    p.add_argument(
        "--shap-cpg-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Deprecated: overrides --shap-intersect-cpg-csv (older CLI used this for intersect pair).",
    )
    p.add_argument(
        "--shap-snp-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Deprecated: overrides --shap-intersect-snp-csv.",
    )
    p.add_argument(
        "--label-shap",
        type=str,
        default=None,
        metavar="NAME",
        help="Deprecated: overrides --label-shap-intersect.",
    )
    p.add_argument(
        "--skip-shap",
        action="store_true",
        help="Deprecated: skip both SHAP blocks (full + intersect).",
    )
    args = p.parse_args()

    if args.shap_cpg_csv is not None:
        args.shap_intersect_cpg_csv = args.shap_cpg_csv
    if args.shap_snp_csv is not None:
        args.shap_intersect_snp_csv = args.shap_snp_csv
    if args.label_shap is not None:
        args.label_shap_intersect = args.label_shap
    if args.skip_shap:
        args.skip_shap_full = True
        args.skip_shap_intersect = True

    root = Path(__file__).resolve().parent.parent

    def resolve(p: str) -> Path:
        x = Path(p)
        return x if x.is_absolute() else root / x

    out_root = resolve(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    annot_path = resolve(args.annot_csv)
    snp_map_path = resolve(args.snp_gene_csv)
    sig_path = resolve(args.sig_csv)

    for path, name in [(annot_path, "Annotation"), (snp_map_path, "SNP–gene map")]:
        if not path.exists():
            raise SystemExit(f"Missing {name}: {path}")

    summary: Dict[str, object] = {
        "annot_csv": str(annot_path.resolve()),
        "snp_gene_csv": str(snp_map_path.resolve()),
        "blocks": {},
    }

    summary["blocks"][args.label_sig] = run_one_block(
        args.label_sig,
        cpg_csv=None,
        snp_csv=None,
        single_csv=sig_path,
        annot_path=annot_path,
        snp_map_path=snp_map_path,
        out_root=out_root,
    )

    if not args.skip_shap_full:
        sh_full_c = resolve(args.shap_full_cpg_csv)
        sh_full_s = resolve(args.shap_full_snp_csv)
        summary["blocks"][args.label_shap_full] = run_one_block(
            args.label_shap_full,
            cpg_csv=sh_full_c,
            snp_csv=sh_full_s,
            single_csv=None,
            annot_path=annot_path,
            snp_map_path=snp_map_path,
            out_root=out_root,
        )

    if not args.skip_shap_intersect:
        int_c = resolve(args.shap_intersect_cpg_csv)
        int_s = resolve(args.shap_intersect_snp_csv)
        if int_c.exists() and int_s.exists():
            summary["blocks"][args.label_shap_intersect] = run_one_block(
                args.label_shap_intersect,
                cpg_csv=int_c,
                snp_csv=int_s,
                single_csv=None,
                annot_path=annot_path,
                snp_map_path=snp_map_path,
                out_root=out_root,
            )
        else:
            summary["blocks"][args.label_shap_intersect] = {
                "label": args.label_shap_intersect,
                "skipped": True,
                "reason": "missing one or both intersect CSVs",
                "paths_checked": [str(int_c.resolve()), str(int_s.resolve())],
            }

    out_json = out_root / "gene_overlap_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_json.read_text(encoding="utf-8"))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
