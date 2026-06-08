#!/usr/bin/env python3
"""External concordance: GWAS Catalog traits for rsIDs + CpG cis-window SNP density.

Outputs under ``feature_importance/bio_relevance/external/``:

  gwas_traits_by_rs.csv       top GWAS traits per rsID (from association/efoTraits)
  cpg_cis_gwas_density.csv    ±window bp around each top SHAP CpG: nearby GWAS SNP count
  clock_cpg_overlap.json      Fisher overlap of SHAP CpGs with Zhang2019 / GrimAge lists
  ewascatalog_probe_check.json   optional probe lookup if EWAS Catalog API responds
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from bio_relevance.http_util import http_get_json


def parse_rsids(s: str) -> List[str]:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    out = []
    for tok in re.split(r"[;,]", str(s)):
        t = tok.strip()
        if t.lower().startswith("rs"):
            out.append(t)
    return out


def collect_rsids_from_minigwas_and_cache(
    minigwas_top: Path,
    gwas_cache: Path,
    max_rs: int,
    p_max: float,
) -> List[str]:
    rs: List[str] = []
    if minigwas_top.exists():
        df = pd.read_csv(minigwas_top, low_memory=False)
        if "p" in df.columns:
            df = df[df["p"].astype(float) <= p_max]
        if "rsids" in df.columns:
            for v in df["rsids"].dropna().astype(str):
                rs.extend(parse_rsids(v))
    if gwas_cache.exists():
        df = pd.read_csv(gwas_cache, low_memory=False)
        for v in df.get("rsids", pd.Series(dtype=str)).dropna().astype(str):
            rs.extend(parse_rsids(v))
    # unique preserve order
    seen = set()
    out = []
    for r in rs:
        if r not in seen:
            seen.add(r)
            out.append(r)
        if len(out) >= max_rs:
            break
    return out


def association_efo_traits(assoc_id: str, *, sleep: float) -> List[str]:
    """GET /associations/{id}/efoTraits -> trait labels."""
    url = f"https://www.ebi.ac.uk/gwas/rest/api/associations/{assoc_id}/efoTraits?size=50"
    data = http_get_json(url, timeout=45)
    time.sleep(sleep)
    if not data or "_embedded" not in data:
        return []
    traits = []
    for t in data["_embedded"].get("efoTraits", []):
        lab = t.get("trait")
        if lab:
            traits.append(str(lab))
    return traits


def rsid_to_association_ids(rsid: str, *, max_assoc: int, sleep: float) -> List[Tuple[str, float]]:
    """Return list of (association_id, pvalue) up to max_assoc."""
    url = (
        f"https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/{rsid}"
        f"/associations?size={max_assoc + 5}"
    )
    data = http_get_json(url, timeout=60)
    time.sleep(sleep)
    if not data or "_embedded" not in data:
        return []
    out: List[Tuple[str, float]] = []
    for a in data["_embedded"].get("associations", []):
        href = a.get("_links", {}).get("self", {}).get("href", "")
        # .../associations/12954
        m = re.search(r"/associations/(\d+)", href)
        if not m:
            continue
        aid = m.group(1)
        pv = float(a.get("pvalue") or 1.0)
        out.append((aid, pv))
    out.sort(key=lambda x: x[1])
    return out[:max_assoc]


def keyword_hits(traits: List[str], keywords: Tuple[str, ...]) -> List[str]:
    low = [t.lower() for t in traits]
    hit = []
    for t in low:
        for k in keywords:
            if k in t:
                hit.append(t)
                break
    return hit


def cpg_cis_snp_density(chrom: str, pos: int, radius: int, *, sleep: float) -> int:
    url = (
        f"https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/"
        f"search/findByChromBpLocationRange?chrom={chrom}&bpStart={max(1, pos - radius)}&bpEnd={pos + radius}"
    )
    data = http_get_json(url, timeout=45)
    time.sleep(sleep)
    if not data or "_embedded" not in data:
        return 0
    return len(data["_embedded"].get("singleNucleotidePolymorphisms", []))


def load_clock_cpgs(clock_csv: Path) -> Set[str]:
    if not clock_csv.exists():
        return set()
    df = pd.read_csv(clock_csv, low_memory=False)
    for col in df.columns:
        if df[col].astype(str).str.match(r"cg\d+", case=False).any():
            s = df[col].astype(str).str.strip()
            return set(s[s.str.match(r"cg\d+", case=False)])
    return set()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/external")
    p.add_argument("--annot-csv", type=str, default="Annotation.csv")
    p.add_argument("--shap-top-cpg", type=str, default="feature_importance/shap/shap_top_cpg_risk.csv")
    p.add_argument("--minigwas-top", type=str, default="feature_importance/minigwas/minigwas_top_hits.csv")
    p.add_argument("--gwas-cache", type=str, default="feature_importance/annot/snp_gwas_cache.csv")
    p.add_argument("--clock-dir", type=str, default="feature_importance/annot/clock_lists")
    p.add_argument("--max-rs", type=int, default=35)
    p.add_argument("--max-assoc-per-rs", type=int, default=4)
    p.add_argument("--p-max", type=float, default=0.05)
    p.add_argument("--cis-window-bp", type=int, default=250_000)
    p.add_argument("--top-cpg-cis", type=int, default=25)
    p.add_argument("--sleep", type=float, default=0.06)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keywords = (
        "mortality", "death", "longevity", "lifespan", "survival",
        "cardiovascular", "coronary", "stroke", "myocardial", "lipid", "cholesterol",
        "inflammatory", "inflammation", "diabetes", "cancer", "neoplasm",
        "blood pressure", "hypertension", "aging", "ageing", "body mass",
    )

    # --- GWAS traits for rsIDs ---
    rs_list = collect_rsids_from_minigwas_and_cache(
        Path(args.minigwas_top), Path(args.gwas_cache), args.max_rs, args.p_max,
    )
    print(f"Collected {len(rs_list)} rsIDs for trait lookup")
    rows = []
    for rsid in rs_list:
        assoc_pv = rsid_to_association_ids(rsid, max_assoc=args.max_assoc_per_rs, sleep=args.sleep)
        all_traits: List[str] = []
        for aid, pv in assoc_pv:
            tr = association_efo_traits(aid, sleep=args.sleep)
            all_traits.extend(tr)
            rows.append({
                "rsid": rsid, "association_id": aid, "pvalue": pv,
                "traits": " | ".join(tr[:12]),
            })
        kh = keyword_hits(all_traits, keywords)
        if kh:
            rows.append({
                "rsid": rsid, "association_id": "KEYWORD_SUMMARY", "pvalue": np.nan,
                "traits": ";; ".join(sorted(set(kh))[:20]),
            })
    pd.DataFrame(rows).drop_duplicates(subset=["rsid", "association_id"]).to_csv(
        out / "gwas_traits_by_rs.csv", index=False
    )
    print(f"  wrote {out / 'gwas_traits_by_rs.csv'}")

    # --- CpG cis GWAS SNP density ---
    if Path(args.shap_top_cpg).exists() and Path(args.annot_csv).exists():
        top = pd.read_csv(args.shap_top_cpg, nrows=args.top_cpg_cis)
        feats = top["feature"].astype(str).tolist()
        ann = pd.read_csv(
            args.annot_csv, usecols=["probeID", "CpG_chrm", "CpG_beg"],
            low_memory=False,
        )
        ann = ann[ann["probeID"].astype(str).isin(feats)]
        dens_rows = []
        for _, r in ann.iterrows():
            cpg = str(r["probeID"])
            chrom = str(r["CpG_chrm"]).replace("chr", "")
            pos = int(r["CpG_beg"])
            n_snps = cpg_cis_snp_density(chrom, pos, args.cis_window_bp, sleep=args.sleep)
            dens_rows.append({"cpg": cpg, "chrom": chrom, "pos": pos, "n_gwas_snps_window": n_snps})
        pd.DataFrame(dens_rows).to_csv(out / "cpg_cis_gwas_density.csv", index=False)
        print(f"  wrote {out / 'cpg_cis_gwas_density.csv'}")

    # --- Offline clock overlap (Fisher) ---
    shap_cpgs = set()
    if Path(args.shap_top_cpg).exists():
        shap_cpgs = set(pd.read_csv(args.shap_top_cpg, nrows=500)["feature"].astype(str))
    clock_dir = Path(args.clock_dir)
    summary = {}
    manifest_cpgs: Set[str] = set()
    if Path(args.annot_csv).exists():
        manifest_cpgs = set(
            pd.read_csv(args.annot_csv, usecols=["probeID"], low_memory=False)["probeID"].astype(str)
        )
    universe = len(manifest_cpgs) if manifest_cpgs else 850_000
    shap_m = shap_cpgs & manifest_cpgs if manifest_cpgs else shap_cpgs
    clock_files = {"zhang2019": "clock_zhang2019.csv", "grimagev2": "clock_grimagev2.csv"}
    for name, fname in clock_files.items():
        ck = clock_dir / fname
        clock = load_clock_cpgs(ck)
        clock_m = clock & manifest_cpgs if manifest_cpgs else clock
        if not clock_m or not shap_m:
            summary[name] = {"error": "missing clock or shap"}
            continue
        a = len(shap_m & clock_m)
        b = len(shap_m - clock_m)
        c = len(clock_m - shap_m)
        d = max(0, universe - a - b - c)
        oddsr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        summary[name] = {
            "n_shap_top": len(shap_m), "n_clock": len(clock_m), "overlap": a,
            "odds_ratio": float(oddsr), "p_fisher_greater": float(p),
            "note": "2x2 contingency on EPIC manifest CpGs as universe",
        }
    (out / "clock_cpg_overlap.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out / 'clock_cpg_overlap.json'}")

    # --- EWAS Catalog probe (best-effort) ---
    probe = None
    if Path(args.shap_top_cpg).exists():
        probe = str(pd.read_csv(args.shap_top_cpg, nrows=1)["feature"].iloc[0])
    ew_url = f"https://www.ewas-catalog.org/api/associations?cpg={probe}" if probe else None
    ew = None
    if ew_url:
        ew = http_get_json(ew_url, timeout=20)
    (out / "ewascatalog_probe_check.json").write_text(
        json.dumps({"probe": probe, "url": ew_url, "response_keys": list(ew.keys()) if isinstance(ew, dict) else str(type(ew))}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
