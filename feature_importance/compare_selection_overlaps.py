#!/usr/bin/env python3
"""Overlap counts between DANN linearized weight selection and other feature lists.

Writes **one folder per comparison method** under ``--out-dir`` (default
``feature_importance/selection_overlap_vs_dann_weight/``), each containing:

  summary.json          counts + Jaccard + paths
  intersection_cpg.csv  CpGs in both sets
  intersection_snp.csv  SNPs in both sets

Also writes ``master_summary.json`` aggregating all methods.

Example::

  python feature_importance/compare_selection_overlaps.py \\
    --base-csv feature_importance/weight_selection/dann_aux_final/selected_mad_modified_z.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def _infer_kind(feature: str) -> str:
    f = str(feature).strip().lower()
    if f.startswith("cg") or f.startswith("ch"):
        return "cpg"
    return "snp"


def _load_two_sets(
    path: Path,
    feature_col: str,
    kind_col: Optional[str],
    usecols: Optional[List[str]] = None,
) -> Tuple[Set[str], Set[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    cols = usecols
    if cols is None:
        cols = [feature_col]
        if kind_col is not None:
            cols.append(kind_col)
    try:
        df = pd.read_csv(path, usecols=lambda c: c in set(cols), low_memory=False)
    except ValueError:
        df = pd.read_csv(path, low_memory=False)
    if feature_col not in df.columns:
        raise SystemExit(f"{path}: missing column {feature_col!r}; have {df.columns.tolist()}")
    feats = df[feature_col].astype(str).str.strip()
    if kind_col is not None and kind_col in df.columns:
        kinds = df[kind_col].astype(str).str.lower().str.strip()
    else:
        kinds = pd.Series([_infer_kind(x) for x in feats], index=df.index)
    cpg = set(feats[kinds == "cpg"].tolist())
    snp = set(feats[kinds == "snp"].tolist())
    return cpg, snp


def _load_pair(cpg_path: Path, snp_path: Path, feature_col: str, kind_col: Optional[str]) -> Tuple[Set[str], Set[str]]:
    c1, s1 = _load_two_sets(cpg_path, feature_col, kind_col)
    c2, s2 = _load_two_sets(snp_path, feature_col, kind_col)
    return c1 | c2, s1 | s2


def _jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    if u == 0:
        return float("nan")
    return len(a & b) / u


def discover_methods(repo_root: Path) -> Dict[str, Tuple[Path, ...]]:
    """Return method_key -> tuple of Path (single CSV or cpg+snp pair)."""
    root = repo_root / "feature_importance"
    return {
        "shap_top_risk": (
            root / "shap/shap_top_cpg_risk.csv",
            root / "shap/shap_top_snp_risk.csv",
        ),
        "shap_top_age": (
            root / "shap/shap_top_cpg_age.csv",
            root / "shap/shap_top_snp_age.csv",
        ),
        "shap_all_risk": (
            root / "shap/shap_all_cpg_risk.csv",
            root / "shap/shap_all_snp_risk.csv",
        ),
        "significant_either": (root / "significance/significant_either.csv",),
        "significant_risk": (root / "significance/significant_risk.csv",),
        "significant_age": (root / "significance/significant_age.csv",),
        "gradient_top_risk": (
            root / "risk_top_cpg.csv",
            root / "risk_top_snp.csv",
        ),
        "gradient_top_age": (
            root / "age_top_cpg.csv",
            root / "age_top_snp.csv",
        ),
        "cohort_split_risk": (
            root / "cohort_split/risk_cpg_cohort_classes.csv",
            root / "cohort_split/risk_snp_cohort_classes.csv",
        ),
        "cohort_split_age": (
            root / "cohort_split/age_cpg_cohort_classes.csv",
            root / "cohort_split/age_snp_cohort_classes.csv",
        ),
        "annot_unified_top": (root / "annot/top_unified_long.csv",),
        "minigwas_top_hits": (root / "minigwas/minigwas_top_hits.csv",),
    }


def _default_base_csv(repo_root: Path) -> Path:
    return repo_root / "feature_importance/weight_selection/dann_aux_final/selected_mad_modified_z.csv"


def load_method_features(
    paths: Tuple[Path, ...],
    *,
    pair_cpg_snp: bool,
    feature_col: str,
    kind_col: Optional[str],
    cohort_class: Optional[str],
    minigwas_snp_col: str,
) -> Tuple[Set[str], Set[str]]:
    if pair_cpg_snp and len(paths) == 2:
        return _load_pair(paths[0], paths[1], feature_col, kind_col)

    path = paths[0]
    if not path.exists():
        raise FileNotFoundError(path)

    if path.name.startswith("minigwas"):
        df = pd.read_csv(path, usecols=[minigwas_snp_col], low_memory=False)
        if minigwas_snp_col not in df.columns:
            raise SystemExit(f"{path}: need column {minigwas_snp_col!r}")
        snps = set(df[minigwas_snp_col].astype(str).str.strip().tolist())
        return set(), snps

    df = pd.read_csv(path, low_memory=False)
    if cohort_class is not None and "class" in df.columns:
        df = df[df["class"].astype(str).str.lower() == cohort_class.lower()].copy()
    if feature_col not in df.columns:
        raise SystemExit(f"{path}: missing {feature_col!r}; columns={df.columns.tolist()}")
    feats = df[feature_col].astype(str).str.strip()
    if kind_col is not None and kind_col in df.columns:
        kinds = df[kind_col].astype(str).str.lower().str.strip()
    else:
        kinds = pd.Series([_infer_kind(x) for x in feats], index=df.index)
    cpg = set(feats[kinds == "cpg"].tolist())
    snp = set(feats[kinds == "snp"].tolist())
    return cpg, snp


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def write_overlap(
    method: str,
    base_cpg: Set[str],
    base_snp: Set[str],
    o_cpg: Set[str],
    o_snp: Set[str],
    out_sub: Path,
) -> Dict[str, object]:
    out_sub.mkdir(parents=True, exist_ok=True)
    ic = base_cpg & o_cpg
    is_ = base_snp & o_snp
    summary = {
        "method": method,
        "base": {"n_cpg": len(base_cpg), "n_snp": len(base_snp), "n_total": len(base_cpg) + len(base_snp)},
        "other": {"n_cpg": len(o_cpg), "n_snp": len(o_snp), "n_total": len(o_cpg) + len(o_snp)},
        "intersection": {
            "n_cpg": len(ic),
            "n_snp": len(is_),
            "n_total": len(ic) + len(is_),
        },
        "jaccard": {
            "cpg": _jaccard(base_cpg, o_cpg),
            "snp": _jaccard(base_snp, o_snp),
            "all_features": _jaccard(base_cpg | base_snp, o_cpg | o_snp),
        },
    }
    pd.DataFrame({"feature": sorted(ic)}).to_csv(out_sub / "intersection_cpg.csv", index=False)
    pd.DataFrame({"feature": sorted(is_)}).to_csv(out_sub / "intersection_snp.csv", index=False)
    (out_sub / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Compare DANN weight selection to SHAP / significance / etc.")
    p.add_argument(
        "--base-csv",
        type=str,
        default=None,
        help="DANN weight selection CSV (columns feature, kind). Default: selected_mad_modified_z under dann_aux_final.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="feature_importance/selection_overlap_vs_dann_weight",
        help="Root output directory; each method gets a subfolder.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated method keys, or 'all' (see discover_methods in script).",
    )
    p.add_argument("--feature-col", type=str, default="feature")
    p.add_argument("--kind-col", type=str, default="kind", help="Use 'infer' to guess cpg vs snp from probe id.")
    p.add_argument(
        "--cohort-class",
        type=str,
        default="robust",
        help="For cohort_split_* methods, keep rows with this class (default robust). Empty = all rows.",
    )
    p.add_argument("--minigwas-snp-col", type=str, default="snp")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if not Path(args.out_dir).is_absolute():
        out_root = repo_root / args.out_dir
    else:
        out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    default_base = _default_base_csv(repo_root)
    base_csv = Path(args.base_csv) if args.base_csv else default_base
    if not base_csv.is_absolute():
        base_csv = repo_root / base_csv
    if not base_csv.exists():
        alts = [
            repo_root / "feature_importance/weight_selection/dann_aux_final/selected_features_union.csv",
            repo_root / "feature_importance/weight_selection/dann_aux_final/dann_aux_weight_ranking.csv",
        ]
        for a in alts:
            if a.exists():
                base_csv = a
                print(f"Using fallback base CSV: {base_csv}")
                break
        else:
            raise SystemExit(f"Base CSV not found: {base_csv}")

    kind_col: Optional[str] = None if args.kind_col.lower() == "infer" else args.kind_col
    base_cpg, base_snp = _load_two_sets(base_csv, args.feature_col, kind_col)
    print(f"Base {base_csv.name}: {len(base_cpg)} CpG + {len(base_snp)} SNP = {len(base_cpg)+len(base_snp)} features")

    methods_map = discover_methods(repo_root)
    cohort_class = args.cohort_class.strip() or None

    pair_methods = {
        "shap_top_risk",
        "shap_top_age",
        "shap_all_risk",
        "gradient_top_risk",
        "gradient_top_age",
        "cohort_split_risk",
        "cohort_split_age",
    }
    if args.methods.strip().lower() == "all":
        wanted = list(methods_map.keys())
    else:
        wanted = [x.strip() for x in args.methods.split(",") if x.strip()]

    master: Dict[str, object] = {
        "base_csv": str(base_csv.resolve()),
        "base_counts": {"cpg": len(base_cpg), "snp": len(base_snp)},
        "cohort_class_filter": cohort_class,
        "methods": {},
    }

    for key in wanted:
        if key not in methods_map:
            raise SystemExit(f"Unknown method {key!r}. Choices: {', '.join(sorted(methods_map))}")
        paths = methods_map[key]
        for path in paths:
            if not path.exists():
                print(f"  SKIP {key}: missing {path}")
                master["methods"][key] = {"error": f"missing {path}"}
                break
        else:
            try:
                o_cpg, o_snp = load_method_features(
                    paths,
                    pair_cpg_snp=key in pair_methods,
                    feature_col=args.feature_col,
                    kind_col=kind_col,
                    cohort_class=cohort_class if key.startswith("cohort_split_") else None,
                    minigwas_snp_col=args.minigwas_snp_col,
                )
            except Exception as e:
                master["methods"][key] = {"error": str(e)}
                print(f"  ERROR {key}: {e}")
                continue
            sub = out_root / sanitize(key)
            summ = write_overlap(key, base_cpg, base_snp, o_cpg, o_snp, sub)
            master["methods"][key] = summ
            print(
                f"  {key}: intersection CpG={summ['intersection']['n_cpg']} SNP={summ['intersection']['n_snp']} "
                f"(other had {summ['other']['n_cpg']}+{summ['other']['n_snp']}) -> {sub}"
            )

    (out_root / "master_summary.json").write_text(json.dumps(master, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_root / 'master_summary.json'}")


if __name__ == "__main__":
    main()
