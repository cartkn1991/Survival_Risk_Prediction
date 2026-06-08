#!/usr/bin/env python3
"""Gene-set enrichment (g:Profiler) for SHAP / mini-GWAS gene lists vs EPIC background.

Writes to ``feature_importance/bio_relevance/pathway/``:

  gost_<label>.tsv           significant terms (GO:BP, REAC, KEGG, MSigDB Hallmark)
  gost_<label>.json         raw API meta
  pathway_bar_<label>.png   top terms by p-value

Uses a custom background = up to ``--max-background`` unique HGNC symbols from
Annotation.csv so enrichment is not biased toward large gene families genome-wide.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bio_relevance.http_util import http_post_json

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def split_genes(cell: str) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = re.split(r"[;,/\|]", s)
    out: List[str] = []
    for p in parts:
        g = p.strip().upper()
        if g and g != "NA" and not g.startswith("RP11-"):
            # keep RP11- etc. actually many are lncRNA — keep all non-empty HGNC-like
            if re.match(r"^[A-Z0-9][A-Z0-9\-]*$", g):
                out.append(g)
    return out


def genes_from_csv(path: Path, col: str) -> Set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        return set()
    g: Set[str] = set()
    for v in df[col].astype(str):
        g.update(split_genes(v))
    return g


def collect_foreground(
    shap_annot_risk_cpg: Path,
    shap_annot_risk_snp: Path,
    minigwas_csv: Path,
    top_n: int,
) -> Set[str]:
    genes: Set[str] = set()
    for p, col in [
        (shap_annot_risk_cpg, "gene_HGNC"),
        (shap_annot_risk_snp, "gene_symbol"),
    ]:
        if p.exists():
            df = pd.read_csv(p, nrows=top_n, low_memory=False)
            if col in df.columns:
                for v in df[col]:
                    genes.update(split_genes(str(v)))
    if minigwas_csv.exists():
        mg = pd.read_csv(minigwas_csv, low_memory=False)
        mg = mg.sort_values("p", ascending=True).head(top_n * 2)
        if "gene_symbol" in mg.columns:
            for v in mg["gene_symbol"].dropna():
                genes.update(split_genes(str(v)))
    # g:Profiler wants symbols; drop obvious non-genes
    return {g for g in genes if len(g) >= 2 and len(g) <= 40}


def load_epic_background(annot_csv: Path, max_n: int, seed: int) -> List[str]:
    print(f"  reading background gene universe from {annot_csv} ...")
    usecols = ["gene_HGNC"]
    df = pd.read_csv(annot_csv, usecols=usecols, low_memory=False)
    bg: Set[str] = set()
    for v in df["gene_HGNC"].astype(str):
        bg.update(split_genes(v))
    bg.discard("")
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(bg))
    if len(arr) > max_n:
        arr = rng.choice(arr, size=max_n, replace=False)
    print(f"  background unique genes: {len(arr)} (capped at {max_n})")
    return sorted(arr.tolist())


def run_gprofiler(query: List[str], background: List[str]) -> dict:
    payload = {
        "query": sorted(set(query)),
        "background": sorted(set(background)),
        "organism": "hsapiens",
        "sources": ["GO:BP", "GO:MF", "REAC", "KEGG"],
        "user_threshold": 0.05,
        "domain_scope": "custom",
    }
    print(f"  g:Profiler POST: |query|={len(payload['query'])}  |background|={len(payload['background'])}")
    resp = http_post_json(GPROFILER_URL, payload, timeout=180)
    if resp is None:
        return {"error": "g:Profiler request failed (network/TLS)"}
    return resp


def results_table(resp: dict, *, fdr_max: float = 0.05) -> pd.DataFrame:
    rows = []
    if "result" not in resp or not resp["result"]:
        return pd.DataFrame(rows)
    for r in resp["result"]:
        apv_raw = r.get("adjusted_p_value")
        pv = float(r.get("p_value") or 1.0)
        if apv_raw is not None and str(apv_raw).strip() != "":
            if float(apv_raw) > fdr_max:
                continue
        elif pv > fdr_max:
            continue
        inter = r.get("intersections", []) or []
        flat: List[str] = []
        for x in inter[:25]:
            if isinstance(x, list):
                flat.extend(str(t) for t in x)
            else:
                flat.append(str(x))
        rows.append({
            "native": r.get("native"),
            "name": r.get("name"),
            "source": r.get("source"),
            "p_value": r.get("p_value"),
            "adjusted_p_value": r.get("adjusted_p_value"),
            "term_size": r.get("term_size"),
            "query_size": r.get("query_size"),
            "intersection_size": r.get("intersection_size"),
            "intersections": ";".join(flat),
        })
    return pd.DataFrame(rows)


def results_table_all(resp: dict) -> pd.DataFrame:
    """All g:Profiler rows (no FDR filter) for null simulations."""
    rows = []
    if "result" not in resp or not resp["result"]:
        return pd.DataFrame(rows)
    for r in resp["result"]:
        inter = r.get("intersections", []) or []
        flat: List[str] = []
        for x in inter[:25]:
            if isinstance(x, list):
                flat.extend(str(t) for t in x)
            else:
                flat.append(str(x))
        rows.append({
            "native": r.get("native"),
            "name": r.get("name"),
            "source": r.get("source"),
            "p_value": r.get("p_value"),
            "adjusted_p_value": r.get("adjusted_p_value"),
            "term_size": r.get("term_size"),
            "query_size": r.get("query_size"),
            "intersection_size": r.get("intersection_size"),
            "intersections": ";".join(flat),
        })
    return pd.DataFrame(rows)


def plot_bar(tsv_path: Path, out_png: Path, title: str, top_k: int = 15) -> None:
    d = pd.read_csv(tsv_path, sep="\t")
    if d.empty:
        return
    d = d.sort_values("adjusted_p_value", ascending=True).head(top_k)
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.35 * len(d))))
    x = -np.log10(np.maximum(d["adjusted_p_value"].astype(float), 1e-300))
    labs = (d["native"].astype(str) + "  " + d["name"].astype(str)).str.slice(0, 80)
    ax.barh(np.arange(len(d)), x, color="#2980b9", edgecolor="black", linewidth=0.3)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel(r"$-\log_{10}$ FDR (g:Profiler)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/pathway")
    p.add_argument("--annot-csv", type=str, default="Annotation.csv")
    p.add_argument("--shap-annot-cpg", type=str, default="feature_importance/shap/annot/risk_top_cpg_annotated.csv")
    p.add_argument("--shap-annot-snp", type=str, default="feature_importance/shap/annot/risk_top_snp_annotated.csv")
    p.add_argument("--minigwas-csv", type=str, default="feature_importance/minigwas/minigwas_results.csv")
    p.add_argument("--top-n", type=int, default=300, help="Rows / genes to pull from each annotation table.")
    p.add_argument("--max-background", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fg = collect_foreground(
        Path(args.shap_annot_cpg),
        Path(args.shap_annot_snp),
        Path(args.minigwas_csv),
        top_n=args.top_n,
    )
    fg_list = sorted(fg)
    print(f"foreground genes (union): {len(fg_list)}")
    (out / "foreground_genes.txt").write_text("\n".join(fg_list), encoding="utf-8")

    bg = load_epic_background(Path(args.annot_csv), args.max_background, args.seed)
    # query must be subset of background for gprofiler custom background
    fg_in_bg = sorted(set(fg_list) & set(bg))
    print(f"foreground genes also in background: {len(fg_in_bg)} / {len(fg_list)}")
    if len(fg_in_bg) < 10:
        print("WARNING: very few query genes in background; g:Profiler may return little.")

    resp = run_gprofiler(fg_in_bg if fg_in_bg else fg_list, bg)
    (out / "gost_response.json").write_text(json.dumps(resp, indent=2, default=str)[:500_000], encoding="utf-8")

    if "error" in resp:
        print(resp["error"])
        return

    df = results_table(resp)
    tsv_path = out / "gost_significant.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  wrote {tsv_path}  ({len(df)} terms)")
    meta = resp.get("meta", {})
    (out / "gost_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if not df.empty:
        plot_bar(tsv_path, out / "pathway_bar_top.png", "g:Profiler enrichment (risk SHAP + mini-GWAS genes)")


if __name__ == "__main__":
    main()
