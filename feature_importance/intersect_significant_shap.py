#!/usr/bin/env python3
"""Intersect significance-selected features with SHAP-selected features (CpG / SNP separately).

Reads ``significant_*.csv`` (must include ``feature``, ``kind``) and the SHAP top tables
(``shap_top_cpg_<target>.csv``, ``shap_top_snp_<target>.csv``), inner-joins on ``feature``,
and writes merged tables, ``common_significant_shap_summary_<target>.json``, and optional
**Venn diagrams** (``matplotlib_venn``; ``pip install matplotlib-venn`` if missing).

Example::

  python feature_importance/intersect_significant_shap.py

  python feature_importance/intersect_significant_shap.py \\
    --sig-csv feature_importance/significance/significant_risk.csv \\
    --target risk --out-dir feature_importance/shap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _split_sig(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "feature" not in df.columns or "kind" not in df.columns:
        raise SystemExit(f"Significance CSV needs feature,kind; got {df.columns.tolist()}")
    k = df["kind"].astype(str).str.lower().str.strip()
    return df.loc[k == "cpg"].copy(), df.loc[k == "snp"].copy()


def _venn_counts(set_a: Set[str], set_b: Set[str]) -> Tuple[int, int, int]:
    """Returns (|A\\B|, |B\\A|, |A∩B|) for matplotlib_venn venn2(subsets=...)."""
    inter = len(set_a & set_b)
    return len(set_a - set_b), len(set_b - set_a), inter


def draw_venn_pair(
    out_png: Path,
    title: str,
    cpg_a: Set[str],
    cpg_b: Set[str],
    snp_a: Set[str],
    snp_b: Set[str],
    label_a: str,
    label_b: str,
) -> Optional[str]:
    try:
        from matplotlib_venn import venn2
    except ImportError:
        return (
            "matplotlib_venn not installed; skip Venn diagrams. "
            "Install with:  pip install matplotlib-venn"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    panels = [
        (axes[0], cpg_a, cpg_b, "CpG"),
        (axes[1], snp_a, snp_b, "SNP"),
    ]
    for ax, sa, sb, ttl in panels:
        oa, ob, ab = _venn_counts(sa, sb)
        venn2(subsets=(oa, ob, ab), set_labels=(label_a, label_b), ax=ax)
        ax.set_title(f"{ttl}  (n∩={ab:,})")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Intersect significance list with SHAP top list.")
    p.add_argument(
        "--sig-csv",
        type=str,
        default="feature_importance/significance/significant_either.csv",
        help="Output of select_significant_features.py (feature + kind + q, etc.).",
    )
    p.add_argument(
        "--shap-cpg",
        type=str,
        default=None,
        help="SHAP CpG table (default: shap_top_cpg_<target>.csv under --shap-dir).",
    )
    p.add_argument(
        "--shap-snp",
        type=str,
        default=None,
        help="SHAP SNP table (default: shap_top_snp_<target>.csv under --shap-dir).",
    )
    p.add_argument("--shap-dir", type=str, default="feature_importance/shap")
    p.add_argument("--target", type=str, default="risk", help="SHAP target suffix (risk or age).")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write results (default: same as --shap-dir).",
    )
    p.add_argument(
        "--no-venn",
        action="store_true",
        help="Skip Venn diagram PNG (still writes CSV + summary JSON).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    sig_path = Path(args.sig_csv)
    if not sig_path.is_absolute():
        sig_path = root / sig_path
    shap_dir = Path(args.shap_dir)
    if not shap_dir.is_absolute():
        shap_dir = root / shap_dir
    out_dir = Path(args.out_dir) if args.out_dir else shap_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t = args.target.strip()
    shap_cpg = Path(args.shap_cpg) if args.shap_cpg else shap_dir / f"shap_top_cpg_{t}.csv"
    shap_snp = Path(args.shap_snp) if args.shap_snp else shap_dir / f"shap_top_snp_{t}.csv"
    if not shap_cpg.is_absolute():
        shap_cpg = root / shap_cpg
    if not shap_snp.is_absolute():
        shap_snp = root / shap_snp

    for path, label in [(sig_path, "significance"), (shap_cpg, "SHAP CpG"), (shap_snp, "SHAP SNP")]:
        if not path.exists():
            raise SystemExit(f"Missing {label} file: {path}")

    sig = pd.read_csv(sig_path, low_memory=False)
    sig_cpg, sig_snp = _split_sig(sig)
    sh_c = pd.read_csv(shap_cpg, low_memory=False)
    sh_s = pd.read_csv(shap_snp, low_memory=False)
    if "feature" not in sh_c.columns or "feature" not in sh_s.columns:
        raise SystemExit("SHAP tables must contain a 'feature' column.")

    sig_cpg["feature"] = sig_cpg["feature"].astype(str).str.strip()
    sig_snp["feature"] = sig_snp["feature"].astype(str).str.strip()
    sh_c["feature"] = sh_c["feature"].astype(str).str.strip()
    sh_s["feature"] = sh_s["feature"].astype(str).str.strip()

    sig_cpg_set: Set[str] = set(sig_cpg["feature"].tolist())
    sig_snp_set: Set[str] = set(sig_snp["feature"].tolist())
    shap_cpg_set: Set[str] = set(sh_c["feature"].tolist())
    shap_snp_set: Set[str] = set(sh_s["feature"].tolist())
    oc, sc, ic = _venn_counts(sig_cpg_set, shap_cpg_set)
    os_, ss, ins = _venn_counts(sig_snp_set, shap_snp_set)

    common_cpg = sig_cpg.merge(sh_c, on="feature", how="inner", suffixes=("_sig", "_shap"))
    common_snp = sig_snp.merge(sh_s, on="feature", how="inner", suffixes=("_sig", "_shap"))

    out_cpg = out_dir / f"common_significant_shap_cpg_{t}.csv"
    out_snp = out_dir / f"common_significant_shap_snp_{t}.csv"
    common_cpg.sort_values("mean_abs", ascending=False).to_csv(out_cpg, index=False)
    common_snp.sort_values("mean_abs", ascending=False).to_csv(out_snp, index=False)

    summ = {
        "sig_csv": str(sig_path.resolve()),
        "shap_cpg_csv": str(shap_cpg.resolve()),
        "shap_snp_csv": str(shap_snp.resolve()),
        "target": t,
        "n_sig_cpg": int(len(sig_cpg)),
        "n_sig_snp": int(len(sig_snp)),
        "n_shap_cpg": int(len(sh_c)),
        "n_shap_snp": int(len(sh_s)),
        "n_common_cpg": int(len(common_cpg)),
        "n_common_snp": int(len(common_snp)),
        "venn": {
            "cpg": {"only_significance": oc, "only_shap": sc, "intersection": ic},
            "snp": {"only_significance": os_, "only_shap": ss, "intersection": ins},
        },
        "outputs": {
            "cpg": str(out_cpg.resolve()),
            "snp": str(out_snp.resolve()),
        },
    }

    venn_png = out_dir / f"common_significant_shap_venn_{t}.png"
    if not args.no_venn:
        msg = draw_venn_pair(
            venn_png,
            title=f"Significance vs SHAP ({t}) — set sizes",
            cpg_a=sig_cpg_set,
            cpg_b=shap_cpg_set,
            snp_a=sig_snp_set,
            snp_b=shap_snp_set,
            label_a="Significance",
            label_b=f"SHAP top ({t})",
        )
        if msg:
            print(msg)
            summ["venn_note"] = msg
        else:
            summ["outputs"]["venn_png"] = str(venn_png.resolve())

    summ_path = out_dir / f"common_significant_shap_summary_{t}.json"
    summ_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")

    print(json.dumps(summ, indent=2))
    print(f"\nWrote {out_cpg.name}, {out_snp.name}, {summ_path.name} under {out_dir}")
    if not args.no_venn and summ["outputs"].get("venn_png"):
        print(f"Wrote {venn_png.name}")


if __name__ == "__main__":
    main()
