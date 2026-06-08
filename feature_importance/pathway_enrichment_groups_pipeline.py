#!/usr/bin/env python3
"""Pathway enrichment (KEGG + GO BP) for significance vs SHAP feature groups, CpG and SNP separate.

For each **group** (default: ``significant_risk``, ``shap_full_risk``, ``shap_intersect_significant_risk``):

  - **CpG-to-gene** preranked GSEA with a signed **beta-like** ranking metric aggregated per gene
    (sum by default): ``mean_grad`` on the significance table; ``mean`` (signed SHAP) on SHAP CSVs.
    Intersect rows can use ``--intersect-cpg-score-col mean_grad_risk`` to rank by the pooled
    risk-gradient instead.
  - **SNP-to-gene** same pipeline with the corresponding SNP score column.
  - Libraries: **KEGG_2021_Human** and **GO_Biological_Process_2021** (Enrichr / gseapy).
  - **Dotplots**: custom matplotlib **GSEA-style** dot plot (NES on x, pathways on y, dot color =
    ``-log10(FDR)``, size ∝ overlap genes from ``Tag %``), saved as ``gsea_nes_dotplot.png`` for
    **both** CpG and SNP runs.

Very small SNP-derived gene lists (e.g. few loci map to genes) may leave **no** KEGG sets passing
gseapy overlap rules; those runs are marked ``skipped`` in the JSON (GO BP often still runs after
automatic ``min_size`` lowering).

Outputs (default ``feature_importance/pathway_enrichment_groups/``)::

  pathway_enrichment_summary.json
  <group>/cpg_gene_ranking_for_gsea.csv, snp_gene_ranking_for_gsea.csv
  <group>/cpg/KEGG/prerank_results.csv, gsea_nes_dotplot.png, ...
  <group>/snp/GO_BP/...

Requires: ``pip install gseapy``

Example::

  python feature_importance/pathway_enrichment_groups_pipeline.py

  python feature_importance/pathway_enrichment_groups_pipeline.py --permutation-num 199 \\
    --intersect-cpg-score-col mean_grad_risk
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import gseapy as gp
except ImportError as e:
    raise SystemExit("Install gseapy:  pip install gseapy") from e


GENE_SETS: Tuple[Tuple[str, str], ...] = (
    ("KEGG_2021_Human", "KEGG"),
    ("GO_Biological_Process_2021", "GO_BP"),
)


def effective_min_size(n_genes: int, requested: int) -> int:
    """gseapy keeps pathways only if min_size <= overlap <= max_size and overlap < n_genes."""
    if n_genes < 5:
        return 2
    cap = max(2, n_genes - 1)
    if n_genes <= 25:
        return max(2, min(requested, 5, cap))
    if n_genes <= 80:
        return max(2, min(requested, 8, cap))
    return max(2, min(requested, cap))


def split_hgnc(cell: object) -> List[str]:
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


def _tiebreak_scores(s: pd.Series) -> pd.Series:
    x = s.astype(np.float64).to_numpy()
    if len(x) == 0:
        return s
    return pd.Series(x + np.linspace(0.0, 1e-9, len(x), dtype=np.float64), index=s.index)


def gene_rank_cpg_from_table(
    feat_df: pd.DataFrame,
    annot_csv: Path,
    *,
    score_col: str,
    agg: str = "sum",
) -> pd.DataFrame:
    """feat_df columns: feature, scores in score_col (beta / mean_grad / mean SHAP)."""
    if "feature" not in feat_df.columns or score_col not in feat_df.columns:
        raise ValueError(f"CpG table needs feature,{score_col}; got {feat_df.columns.tolist()}")
    sh = feat_df[["feature", score_col]].copy()
    sh["feature"] = sh["feature"].astype(str).str.strip()
    ann = pd.read_csv(annot_csv, usecols=["probeID", "gene_HGNC"], low_memory=False)
    ann["probeID"] = ann["probeID"].astype(str).str.strip()
    m = sh.merge(ann, left_on="feature", right_on="probeID", how="left")
    rows: List[Tuple[str, float]] = []
    for _, r in m.iterrows():
        try:
            sc = float(r[score_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sc):
            continue
        for g in split_hgnc(r.get("gene_HGNC", "")):
            rows.append((g, sc))
    if not rows:
        return pd.DataFrame(columns=["gene", "score"])
    df = pd.DataFrame(rows, columns=["gene", "score"])
    if agg == "maxabs":
        df["_a"] = df["score"].abs()
        df = df.sort_values("_a", ascending=False).drop_duplicates("gene").drop(columns="_a")
    else:
        df = df.groupby("gene", as_index=False)["score"].sum()
    df = df[df["score"].notna()]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["score"] = _tiebreak_scores(df["score"])
    return df


def gene_rank_snp_from_table(
    feat_df: pd.DataFrame,
    snp_gene_csv: Path,
    *,
    score_col: str,
    agg: str = "sum",
) -> pd.DataFrame:
    if "feature" not in feat_df.columns or score_col not in feat_df.columns:
        raise ValueError(f"SNP table needs feature,{score_col}; got {feat_df.columns.tolist()}")
    sh = feat_df[["feature", score_col]].copy()
    sh["feature"] = sh["feature"].astype(str).str.strip()
    gmap = pd.read_csv(snp_gene_csv, low_memory=False)
    idcol = "feature_id" if "feature_id" in gmap.columns else gmap.columns[0]
    symcol = "gene_symbol" if "gene_symbol" in gmap.columns else "gene_ensembl"
    if symcol not in gmap.columns:
        raise SystemExit(f"SNP map missing gene column: {gmap.columns.tolist()}")
    gmap[idcol] = gmap[idcol].astype(str).str.strip()
    m = sh.merge(gmap, left_on="feature", right_on=idcol, how="inner")
    rows: List[Tuple[str, float]] = []
    for _, r in m.iterrows():
        try:
            sc = float(r[score_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sc):
            continue
        for g in split_hgnc(r.get(symcol, "")):
            rows.append((g, sc))
    if not rows:
        return pd.DataFrame(columns=["gene", "score"])
    df = pd.DataFrame(rows, columns=["gene", "score"])
    if agg == "maxabs":
        df["_a"] = df["score"].abs()
        df = df.sort_values("_a", ascending=False).drop_duplicates("gene").drop(columns="_a")
    else:
        df = df.groupby("gene", as_index=False)["score"].sum()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["score"] = _tiebreak_scores(df["score"])
    return df


def load_sig_kind_dfs(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    if "feature" not in df.columns or "kind" not in df.columns:
        raise SystemExit(f"{path}: need feature, kind")
    k = df["kind"].astype(str).str.lower().str.strip()
    cpg = df.loc[k == "cpg"].copy()
    snp = df.loc[k == "snp"].copy()
    return cpg, snp


def load_shap_pair(cpg_path: Path, snp_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cpg = pd.read_csv(cpg_path, low_memory=False)
    snp = pd.read_csv(snp_path, low_memory=False)
    return cpg, snp


def pick_dotplot_slice(res2d: pd.DataFrame, *, n_pos: int = 12, n_neg: int = 12) -> pd.DataFrame:
    """Mix strong positive- and negative-NES pathways for a symmetric dotplot."""
    if res2d is None or res2d.empty:
        return res2d
    r = res2d.copy()
    r["NES"] = r["NES"].astype(float)
    r["FDR q-val"] = r["FDR q-val"].astype(float)
    pos = r.nlargest(n_pos, "NES")
    neg = r.nsmallest(n_neg, "NES")
    out = pd.concat([pos, neg], axis=0).drop_duplicates(subset=["Term"])
    return out


def _tag_hits(tag_cell: object) -> float:
    """Parse gseapy 'Tag %%' like '8/14' -> leading hit count for sizing."""
    s = str(tag_cell).strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)", s)
    if m:
        return max(1.0, float(m.group(1)))
    return 4.0


def save_gsea_nes_dotplot(
    res2d: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    n_pos: int = 12,
    n_neg: int = 12,
) -> None:
    """GSEA dot plot: x = NES, y = pathways, color = -log10(FDR), size = overlap gene count."""
    FS_TITLE = 14
    FS_LABEL = 12
    FS_TICK = 12
    FS_LEG = 12

    sub = pick_dotplot_slice(res2d, n_pos=n_pos, n_neg=n_neg)
    if sub.empty or len(sub) < 1:
        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        ax.text(0.5, 0.5, "Too few pathways for dotplot", ha="center", va="center", fontsize=FS_LABEL)
        ax.axis("off")
        fig.patch.set_facecolor("#fafafa")
        fig.savefig(out_png, dpi=165, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    d = sub.copy()
    d["NES"] = d["NES"].astype(float)
    d["FDR q-val"] = pd.to_numeric(d["FDR q-val"], errors="coerce").clip(lower=1e-300)
    d["nlq"] = -np.log10(d["FDR q-val"].to_numpy(dtype=np.float64))
    if "Tag %" in d.columns:
        d["_hits"] = d["Tag %"].map(_tag_hits)
    else:
        d["_hits"] = 6.0

    # Sort: strongest positive NES at top of y-axis (matplotlib y increases upward)
    d = d.sort_values("NES", ascending=True).reset_index(drop=True)
    d["Term_short"] = d["Term"].astype(str).str.replace("^KEGG_", "", regex=False).str.slice(0, 72)

    y = np.arange(len(d))
    nes = d["NES"].to_numpy(dtype=np.float64)
    nlq = d["nlq"].to_numpy(dtype=np.float64)
    hits = d["_hits"].to_numpy(dtype=np.float64)

    # Size scaling (points^2 for scatter); slightly larger range on big canvas
    s_min, s_max = 55.0, 520.0
    h_lo, h_hi = np.percentile(hits, [5, 95]) if len(hits) > 2 else (hits.min(), hits.max())
    span = max(h_hi - h_lo, 1e-6)
    w = (np.clip(hits, h_lo, h_hi) - h_lo) / span
    sizes = s_min + w * (s_max - s_min)

    n_path = len(d)
    # Wide panel for NES + long pathway names; extra height for title + overlap legend row
    fig_w = 14.0
    fig_h = max(7.5, 0.42 * n_path + 2.2)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#fafafa")
    # Top: data + colorbar column; bottom: dedicated strip for overlap legend (no overlap with dots)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.14],
        width_ratios=[1.0, 0.042],
        hspace=0.22,
        wspace=0.06,
        left=0.30,
        right=0.98,
        top=0.90,
        bottom=0.07,
    )
    ax = fig.add_subplot(gs[0, 0], facecolor="#f6f6f8")
    cax = fig.add_subplot(gs[0, 1], facecolor="#fafafa")
    leg_ax = fig.add_subplot(gs[1, :])
    leg_ax.axis("off")

    sc = ax.scatter(
        nes,
        y,
        s=sizes,
        c=nlq,
        cmap="viridis",
        alpha=0.88,
        edgecolors="#2d2d2d",
        linewidths=0.45,
        zorder=3,
    )
    ax.axvline(0.0, color="#444444", linestyle="-", linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Term_short"], fontsize=FS_TICK)
    ax.set_xlabel("NES", fontsize=FS_LABEL, labelpad=6)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=FS_TITLE, pad=10)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.margins(x=0.06)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    cbar = fig.colorbar(sc, cax=cax, orientation="vertical")
    cbar.set_label(r"$-\log_{10}$(FDR)", fontsize=FS_LABEL, labelpad=8)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")

    # Size legend (reference overlap counts) — centered under plot, not on data
    if len(hits) >= 3:
        pct = np.percentile(hits, [25, 50, 75])
        h_vals = sorted({int(max(1, round(float(x)))) for x in pct if np.isfinite(x)})
    else:
        h_vals = [int(max(1, round(float(hits.min())))), int(max(1, round(float(np.median(hits)))))]
    h_vals = sorted(set(h_vals))
    if len(h_vals) < 2:
        h_vals = [max(1, int(hits.min())), max(2, int(np.median(hits)) + 1)]
    h_vals = sorted(set(h_vals))[:3]
    leg_handles = []
    for hv in h_vals:
        wv = (np.clip(float(hv), h_lo, h_hi) - h_lo) / span
        sv = s_min + wv * (s_max - s_min)
        leg_handles.append(
            plt.scatter([], [], s=sv, c="#555555", alpha=0.75, edgecolors="#222222", linewidths=0.35)
        )
    leg = leg_ax.legend(
        leg_handles,
        [f"{h} genes" for h in h_vals],
        title="Overlap (leading-edge gene count)",
        loc="center",
        ncol=min(4, len(h_vals)),
        frameon=True,
        fancybox=False,
        fontsize=FS_LEG,
        title_fontsize=FS_LEG,
        borderpad=0.65,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor("#bbbbbb")
    leg.get_frame().set_facecolor("#fbfbfb")

    fig.savefig(out_png, dpi=165, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_prerank_one_library(
    rnk: pd.DataFrame,
    out_dir: Path,
    gene_set: str,
    *,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: int,
    modality: str,
    library_short: str,
    dotplot_title: str,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rnk.empty or len(rnk) < 5:
        msg = f"too_few_genes_{len(rnk)}"
        (out_dir / "skipped.txt").write_text(msg, encoding="utf-8")
        return {"skipped": True, "reason": msg, "n_genes": int(len(rnk))}

    rnk2 = rnk.rename(columns={"gene": "g", "score": "s"}).copy()
    eff_min = effective_min_size(len(rnk), min_size)
    pre = None
    last_err: Optional[str] = None
    min_used = eff_min
    candidates = [eff_min]
    if eff_min > 2:
        candidates.append(2)
    for ms in candidates:
        try:
            pre = gp.prerank(
                rnk=rnk2,
                gene_sets=gene_set,
                organism="Human",
                outdir=str(out_dir),
                permutation_num=permutation_num,
                min_size=ms,
                max_size=max_size,
                seed=seed,
                verbose=False,
                format="png",
            )
            last_err = None
            min_used = ms
            break
        except LookupError as exc:
            last_err = str(exc)
            (out_dir / f"prerank_retry_min{ms}.log").write_text(last_err, encoding="utf-8")
            continue
    if pre is None:
        (out_dir / "skipped.txt").write_text(last_err or "prerank_failed", encoding="utf-8")
        return {
            "skipped": True,
            "reason": "gseapy_no_pathways_after_filter",
            "detail": last_err,
            "n_genes": int(len(rnk)),
            "min_size_tried": eff_min,
        }

    res = pre.res2d.copy()
    csv_path = out_dir / "prerank_results.csv"
    res.to_csv(csv_path, index=False)
    dot_path = out_dir / "gsea_nes_dotplot.png"
    save_gsea_nes_dotplot(res, dot_path, title=dotplot_title)
    top = res.sort_values("FDR q-val", ascending=True).head(1)
    topd = top.iloc[0].to_dict() if len(top) else {}
    return {
        "skipped": False,
        "n_genes": int(len(rnk)),
        "min_size_used": int(min_used),
        "n_terms": int(len(res)),
        "results_csv": str(csv_path.resolve()),
        "dotplot_png": str(dot_path.resolve()),
        "top_by_fdr": topd,
    }


def run_group(
    label: str,
    cpg_df: pd.DataFrame,
    snp_df: pd.DataFrame,
    *,
    cpg_score_col: str,
    snp_score_col: str,
    annot_csv: Path,
    snp_gene_csv: Path,
    out_root: Path,
    gene_agg: str,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: int,
) -> Dict[str, object]:
    gdir = out_root / label
    gdir.mkdir(parents=True, exist_ok=True)
    rep: Dict[str, object] = {
        "label": label,
        "cpg_score_col": cpg_score_col,
        "snp_score_col": snp_score_col,
        "n_input_cpg": int(len(cpg_df)),
        "n_input_snp": int(len(snp_df)),
        "modalities": {},
    }

    rnk_c = gene_rank_cpg_from_table(cpg_df, annot_csv, score_col=cpg_score_col, agg=gene_agg)
    rnk_s = gene_rank_snp_from_table(snp_df, snp_gene_csv, score_col=snp_score_col, agg=gene_agg)
    rnk_c.to_csv(gdir / "cpg_gene_ranking_for_gsea.csv", index=False)
    rnk_s.to_csv(gdir / "snp_gene_ranking_for_gsea.csv", index=False)
    rep["n_genes_cpg"] = int(len(rnk_c))
    rep["n_genes_snp"] = int(len(rnk_s))

    for modality, rnk in ("cpg", rnk_c), ("snp", rnk_s):
        mod_out: Dict[str, object] = {}
        for gs_name, gs_short in GENE_SETS:
            sub = gdir / modality / gs_short
            title = (
                f"{label} {modality.upper()} - {gs_short} "
                f"(gene rank: sum of {cpg_score_col if modality == 'cpg' else snp_score_col}; "
                f"dot color: -log10 FDR)"
            )
            mod_out[gs_short] = run_prerank_one_library(
                rnk,
                sub,
                gs_name,
                permutation_num=permutation_num,
                min_size=min_size,
                max_size=max_size,
                seed=seed + (1 if modality == "snp" else 0) + (0 if gs_short == "KEGG" else 17),
                modality=modality,
                library_short=gs_short,
                dotplot_title=title,
            )
        rep["modalities"][modality] = mod_out

    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="KEGG + GO BP GSEA for significance vs SHAP groups (CpG/SNP).")
    p.add_argument("--out-dir", type=str, default="feature_importance/pathway_enrichment_groups")
    p.add_argument("--sig-csv", type=str, default="feature_importance/significance/significant_risk.csv")
    p.add_argument("--shap-full-cpg", type=str, default="feature_importance/shap/shap_top_cpg_risk.csv")
    p.add_argument("--shap-full-snp", type=str, default="feature_importance/shap/shap_top_snp_risk.csv")
    p.add_argument("--shap-intersect-cpg", type=str, default="feature_importance/shap/common_significant_shap_cpg_risk.csv")
    p.add_argument("--shap-intersect-snp", type=str, default="feature_importance/shap/common_significant_shap_snp_risk.csv")
    p.add_argument("--annot-csv", type=str, default="Annotation.csv")
    p.add_argument(
        "--snp-gene-csv",
        type=str,
        default="analysis_out/gene_pathway_pipeline/resources/auto_snp_to_gene_ensembl.csv",
    )
    p.add_argument("--gene-agg", type=str, default="sum", choices=["sum", "maxabs"])
    p.add_argument("--permutation-num", type=int, default=499)
    p.add_argument("--min-size", type=int, default=10)
    p.add_argument("--max-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-shap-full", action="store_true")
    p.add_argument("--skip-shap-intersect", action="store_true")
    p.add_argument("--skip-significant", action="store_true")
    p.add_argument(
        "--intersect-cpg-score-col",
        type=str,
        default="mean",
        help="Intersect CSV column for CpG-to-gene rank (e.g. mean_grad_risk for pooled risk-gradient beta).",
    )
    p.add_argument(
        "--intersect-snp-score-col",
        type=str,
        default="mean",
        help="Intersect CSV column for SNP-to-gene rank.",
    )
    args = p.parse_args()

    root_proj = Path(__file__).resolve().parent.parent

    def R(s: str) -> Path:
        x = Path(s)
        return x if x.is_absolute() else root_proj / x

    out_root = R(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    annot = R(args.annot_csv)
    snpmap = R(args.snp_gene_csv)
    for path, name in [(annot, "Annotation.csv"), (snpmap, "SNP gene map")]:
        if not path.exists():
            raise SystemExit(f"Missing {name}: {path}")

    summary: Dict[str, object] = {
        "gene_sets": [x[0] for x in GENE_SETS],
        "annot_csv": str(annot.resolve()),
        "snp_gene_csv": str(snpmap.resolve()),
        "groups": {},
    }

    if not args.skip_significant:
        sig = R(args.sig_csv)
        if not sig.exists():
            summary["groups"]["significant_risk"] = {"skipped": True, "reason": f"missing {sig}"}
        else:
            cpg_df, snp_df = load_sig_kind_dfs(sig)
            summary["groups"]["significant_risk"] = run_group(
                "significant_risk",
                cpg_df,
                snp_df,
                cpg_score_col="mean_grad",
                snp_score_col="mean_grad",
                annot_csv=annot,
                snp_gene_csv=snpmap,
                out_root=out_root,
                gene_agg=args.gene_agg,
                permutation_num=args.permutation_num,
                min_size=args.min_size,
                max_size=args.max_size,
                seed=args.seed,
            )

    if not args.skip_shap_full:
        cpg_p, snp_p = R(args.shap_full_cpg), R(args.shap_full_snp)
        if cpg_p.exists() and snp_p.exists():
            cpg_df, snp_df = load_shap_pair(cpg_p, snp_p)
            summary["groups"]["shap_full_risk"] = run_group(
                "shap_full_risk",
                cpg_df,
                snp_df,
                cpg_score_col="mean",
                snp_score_col="mean",
                annot_csv=annot,
                snp_gene_csv=snpmap,
                out_root=out_root,
                gene_agg=args.gene_agg,
                permutation_num=args.permutation_num,
                min_size=args.min_size,
                max_size=args.max_size,
                seed=args.seed + 11,
            )
        else:
            summary["groups"]["shap_full_risk"] = {"skipped": True, "reason": "missing shap full csv"}

    if not args.skip_shap_intersect:
        cpg_p, snp_p = R(args.shap_intersect_cpg), R(args.shap_intersect_snp)
        if cpg_p.exists() and snp_p.exists():
            cpg_df, snp_df = load_shap_pair(cpg_p, snp_p)
            summary["groups"]["shap_intersect_significant_risk"] = run_group(
                "shap_intersect_significant_risk",
                cpg_df,
                snp_df,
                cpg_score_col=args.intersect_cpg_score_col,
                snp_score_col=args.intersect_snp_score_col,
                annot_csv=annot,
                snp_gene_csv=snpmap,
                out_root=out_root,
                gene_agg=args.gene_agg,
                permutation_num=args.permutation_num,
                min_size=args.min_size,
                max_size=args.max_size,
                seed=args.seed + 23,
            )
        else:
            summary["groups"]["shap_intersect_significant_risk"] = {
                "skipped": True,
                "reason": "missing intersect shap csv",
            }

    js = out_root / "pathway_enrichment_summary.json"
    js.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(js.read_text(encoding="utf-8"))
    print(f"\nWrote {js}")


if __name__ == "__main__":
    main()
