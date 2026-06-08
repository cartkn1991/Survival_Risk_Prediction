#!/usr/bin/env python3
"""Preranked KEGG GSEA from CpG-level and SNP-level SHAP (gene-centric ranking).

Pipeline (explainable to reviewers):

1. **CpG → gene** — Each probe maps to HGNC symbol(s) from ``Annotation.csv`` (split ``gene_HGNC``).
   Each gene receives the **sum** of per-probe mean SHAP scores (``mean`` column) across all
   probes annotated to that gene (net model attribution routed through the gene).

2. **SNP → gene** — Merge SHAP table with ``auto_snp_to_gene_ensembl.csv`` (``feature_id``).

3. **Preranked GSEA** — Genes sorted by descending score; ``gseapy.prerank`` with
   ``gene_sets='KEGG_2021_Human'`` (Curated KEGG pathways, MSigDB / Enrichr naming).

4. **Figures** — horizontal **NES bar plot**; **NES vs −log10 FDR** scatter (size ∝ leading-edge
   genes); **running-enrichment** curves via ``Prerank.plot`` (includes ranked-metric track).

Requires: ``pip install gseapy``

Outputs (default ``feature_importance/bio_relevance/kegg_gsea/``)::

  cpg_gene_ranking_for_gsea.csv, snp_gene_ranking_for_gsea.csv
  cpg/KEGG_prerank_results.csv, kegg_nes_barplot.png, kegg_nes_vs_fdr_scatter.png, gseaplot_*.png
  snp/… same
  kegg_gsea_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

try:
    import gseapy as gp
except ImportError as e:
    raise SystemExit("Install gseapy:  pip install gseapy") from e


def split_hgnc(cell: str) -> List[str]:
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


def gene_scores_from_cpgs(
    shap_cpg_csv: Path,
    annot_csv: Path,
    score_col: str = "mean",
    agg: str = "sum",
) -> pd.DataFrame:
    """Return two-column ranking: gene, score (sorted descending by caller)."""
    sh = pd.read_csv(shap_cpg_csv, low_memory=False)
    if "feature" not in sh.columns or score_col not in sh.columns:
        raise SystemExit(f"SHAP CpG CSV missing feature/{score_col}: {sh.columns.tolist()}")
    ann = pd.read_csv(annot_csv, usecols=["probeID", "gene_HGNC"], low_memory=False)
    ann["probeID"] = ann["probeID"].astype(str).str.strip()
    m = sh.merge(ann, left_on="feature", right_on="probeID", how="left")
    rows = []
    for _, r in m.iterrows():
        sc = float(r[score_col])
        for g in split_hgnc(str(r.get("gene_HGNC", ""))):
            rows.append({"gene": g, "score": sc})
    if not rows:
        raise SystemExit("No CpG–gene rows after merge.")
    df = pd.DataFrame(rows)
    if agg == "maxabs":
        df["abs"] = df["score"].abs()
        df = df.sort_values("abs", ascending=False).drop_duplicates("gene").drop(columns="abs")
    else:
        df = df.groupby("gene", as_index=False)["score"].sum()
    df = df[df["score"].notna()]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    # Tiny tie-breaker so gseapy does not treat many genes as having identical rank metrics
    df["score"] = df["score"].astype(np.float64) + np.linspace(0.0, 1e-9, len(df), dtype=np.float64)
    return df


def gene_scores_from_snps(
    shap_snp_csv: Path,
    snp_gene_csv: Path,
    score_col: str = "mean",
    agg: str = "sum",
) -> pd.DataFrame:
    sh = pd.read_csv(shap_snp_csv, low_memory=False)
    if "feature" not in sh.columns:
        raise SystemExit(f"SHAP SNP CSV missing feature: {sh.columns.tolist()}")
    gmap = pd.read_csv(snp_gene_csv, low_memory=False)
    idcol = "feature_id" if "feature_id" in gmap.columns else gmap.columns[0]
    symcol = "gene_symbol" if "gene_symbol" in gmap.columns else "gene"
    gmap[idcol] = gmap[idcol].astype(str).str.strip()
    m = sh.merge(gmap, left_on="feature", right_on=idcol, how="inner")
    if m.empty:
        raise SystemExit("No SNPs matched gene map; check feature_id format vs SHAP 'feature'.")
    rows = []
    for _, r in m.iterrows():
        sc = float(r[score_col])
        for g in split_hgnc(str(r[symcol])):
            rows.append({"gene": g, "score": sc})
    df = pd.DataFrame(rows)
    if agg == "maxabs":
        df["abs"] = df["score"].abs()
        df = df.sort_values("abs", ascending=False).drop_duplicates("gene").drop(columns="abs")
    else:
        df = df.groupby("gene", as_index=False)["score"].sum()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["score"] = df["score"].astype(np.float64) + np.linspace(0.0, 1e-9, len(df), dtype=np.float64)
    return df


def run_prerank_kegg(
    rnk: pd.DataFrame,
    out_subdir: Path,
    *,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: int,
    top_dot: int,
    top_curves: int,
) -> pd.DataFrame:
    out_subdir.mkdir(parents=True, exist_ok=True)
    # gseapy expects first col = gene, second = ranking metric
    rnk2 = rnk.rename(columns={"gene": "g", "score": "s"}).copy()
    pre = gp.prerank(
        rnk=rnk2,
        gene_sets="KEGG_2021_Human",
        organism="Human",
        outdir=str(out_subdir),
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
        verbose=False,
        format="png",
    )
    res = pre.res2d.copy()
    res.to_csv(out_subdir / "KEGG_prerank_results.csv", index=False)

    # --- Explainable summary figure: NES bar + point size = leading-edge gene count ---
    import matplotlib.pyplot as plt
    import numpy as np

    d = res.sort_values("FDR q-val", ascending=True).head(top_dot).iloc[::-1].copy()
    d["nlq"] = -np.log10(np.maximum(d["FDR q-val"].astype(float), 1e-300))
    nes = d["NES"].astype(float).values
    y = np.arange(len(d))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(d)))
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.38 * len(d))))
    bars = ax.barh(y, nes, color=colors, edgecolor="black", linewidth=0.35, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Term"].astype(str).str.replace("KEGG_", "").str.slice(0, 72), fontsize=8)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("Normalized enrichment score (NES)\n(positive ⇒ pathway genes enriched at high-SHAP end of rank list)")
    ax.set_title("KEGG preranked GSEA — top pathways by FDR")
    # second x-axis not needed; annotate FDR as text on bars
    for yi, (_, row) in enumerate(d.iterrows()):
        fdr = float(row["FDR q-val"])
        ax.text(nes[yi] + (0.02 if nes[yi] >= 0 else -0.02), yi, f"q={fdr:.2g}",
                va="center", ha="left" if nes[yi] >= 0 else "right", fontsize=7, clip_on=False)
    fig.tight_layout()
    fig.savefig(out_subdir / "kegg_nes_barplot.png", dpi=165, bbox_inches="tight")
    plt.close(fig)

    # Scatter: NES vs -log10 FDR (point size = leading edge gene count)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    dd = res.sort_values("FDR q-val").head(min(top_dot, len(res))).copy()

    def _lead_n(s) -> float:
        if pd.isna(s) or str(s).strip() == "":
            return 5.0
        return max(3.0, len(re.split(r"[;,]", str(s))))

    sizes = dd["Lead_genes"].map(_lead_n).astype(float) * 22
    sc = ax.scatter(
        dd["NES"].astype(float),
        -np.log10(np.maximum(dd["FDR q-val"].astype(float), 1e-300)),
        s=sizes,
        c=dd["NES"].astype(float),
        cmap="coolwarm",
        alpha=0.88,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("NES")
    ax.set_ylabel(r"$-\log_{10}$ FDR")
    ax.set_title("KEGG GSEA (each point = one pathway; size ∝ leading-edge genes)")
    plt.colorbar(sc, ax=ax, label="NES")
    fig.tight_layout()
    fig.savefig(out_subdir / "kegg_nes_vs_fdr_scatter.png", dpi=165, bbox_inches="tight")
    plt.close(fig)

    # --- Running-sum curves for top pathways ---
    top_terms = (
        res.sort_values("FDR q-val", ascending=True)["Term"].astype(str).head(top_curves).tolist()
    )
    for i, term in enumerate(top_terms):
        try:
            pre.plot(term, ofname=str(out_subdir / f"gseaplot_{i+1:02d}.png"), figsize=(5, 6))
        except Exception as exc:
            (out_subdir / f"gseaplot_{i+1:02d}_error.txt").write_text(str(exc), encoding="utf-8")
    return res


def run_kegg_ora_enrichr(genes: List[str], out_subdir: Path, *, title: str, top_term: int = 22) -> None:
    """Over-representation KEGG (Enrichr library) for a high-|SHAP| gene list."""
    genes = [g for g in genes if g and isinstance(g, str)]
    if len(genes) < 5:
        return
    tmp = out_subdir / "_enrichr_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=["KEGG_2021_Human"],
            organism="human",
            outdir=str(tmp),
            no_plot=True,
        )
        enr.results.to_csv(out_subdir / "KEGG_ORA_enrichr.csv", index=False)
        import matplotlib.pyplot as plt

        gp.barplot(
            enr.results.head(top_term),
            column="Adjusted P-value",
            title=title,
            cutoff=0.05,
            top_term=top_term,
            figsize=(6.5, max(5.0, 0.32 * top_term)),
            ofname=str(out_subdir / "kegg_ora_barplot.png"),
        )
        plt.close("all")
    finally:
        # keep CSV; remove enrichr clutter
        import shutil

        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/kegg_gsea")
    p.add_argument("--shap-cpg", type=str, default="feature_importance/shap/shap_all_cpg_risk.csv")
    p.add_argument("--shap-snp", type=str, default="feature_importance/shap/shap_all_snp_risk.csv")
    p.add_argument("--annot-csv", type=str, default="Annotation.csv")
    p.add_argument("--snp-gene-csv", type=str,
                   default="analysis_out/gene_pathway_pipeline/resources/auto_snp_to_gene_ensembl.csv")
    p.add_argument("--score-col", type=str, default="mean", help="SHAP column: mean (signed) or mean_abs.")
    p.add_argument("--gene-agg", type=str, default="sum", choices=["sum", "maxabs"])
    p.add_argument("--permutation-num", type=int, default=499)
    p.add_argument("--min-size", type=int, default=10,
                   help="Min overlap genes between KEGG set and ranked list (gseapy).")
    p.add_argument("--max-size", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-dot", type=int, default=20)
    p.add_argument("--top-curves", type=int, default=4)
    p.add_argument("--no-ora", action="store_true",
                   help="Skip KEGG over-representation (Enrichr) on top |SHAP| genes.")
    p.add_argument("--ora-top-genes", type=int, default=400)
    args = p.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    summary: dict = {"gene_sets": "KEGG_2021_Human", "score_col": args.score_col, "gene_agg": args.gene_agg}

    # CpG → gene GSEA
    rnk_cpg = gene_scores_from_cpgs(
        Path(args.shap_cpg), Path(args.annot_csv), score_col=args.score_col, agg=args.gene_agg,
    )
    rnk_cpg.to_csv(root / "cpg_gene_ranking_for_gsea.csv", index=False)
    summary["cpg"] = {"n_genes": int(len(rnk_cpg)), "top5": rnk_cpg.head(5).to_dict(orient="records")}
    res_c = run_prerank_kegg(
        rnk_cpg,
        root / "cpg",
        permutation_num=args.permutation_num,
        min_size=args.min_size,
        max_size=args.max_size,
        seed=args.seed,
        top_dot=args.top_dot,
        top_curves=args.top_curves,
    )
    summary["cpg"]["n_kegg_terms_tested"] = int(len(res_c))
    summary["cpg"]["top_pathway"] = res_c.sort_values("FDR q-val").iloc[0].to_dict() if len(res_c) else {}
    if not args.no_ora:
        og = rnk_cpg.assign(_a=rnk_cpg["score"].abs()).sort_values("_a", ascending=False).head(args.ora_top_genes)
        run_kegg_ora_enrichr(
            og["gene"].astype(str).tolist(),
            root / "cpg",
            title="KEGG ORA (CpG SHAP → gene): top genes by |aggregated SHAP|",
        )

    # SNP → gene GSEA
    rnk_snp = gene_scores_from_snps(
        Path(args.shap_snp), Path(args.snp_gene_csv), score_col=args.score_col, agg=args.gene_agg,
    )
    rnk_snp.to_csv(root / "snp_gene_ranking_for_gsea.csv", index=False)
    summary["snp"] = {"n_genes": int(len(rnk_snp)), "top5": rnk_snp.head(5).to_dict(orient="records")}
    res_s = run_prerank_kegg(
        rnk_snp,
        root / "snp",
        permutation_num=args.permutation_num,
        min_size=args.min_size,
        max_size=args.max_size,
        seed=args.seed + 1,
        top_dot=args.top_dot,
        top_curves=args.top_curves,
    )
    summary["snp"]["n_kegg_terms_tested"] = int(len(res_s))
    summary["snp"]["top_pathway"] = res_s.sort_values("FDR q-val").iloc[0].to_dict() if len(res_s) else {}
    if not args.no_ora:
        og = rnk_snp.assign(_a=rnk_snp["score"].abs()).sort_values("_a", ascending=False).head(args.ora_top_genes)
        run_kegg_ora_enrichr(
            og["gene"].astype(str).tolist(),
            root / "snp",
            title="KEGG ORA (SNP SHAP → gene): top genes by |aggregated SHAP|",
        )

    (root / "kegg_gsea_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nDone. Outputs under {root}/")


if __name__ == "__main__":
    main()
