#!/usr/bin/env python3
"""Score methylation clocks + EpiClock on FHS/WHI; compare mortality C-index to AESurv log_h.

Clocks: Horvath, Hannum, PhenoAge, GrimAge V2, DunedinPACE, EpiClock (custom).
Uses coefficient tables in feature_importance/annot/clock_lists/ unless overridden.

Outputs (default feature_importance/bio_relevance/clocks/):
  clock_scores_{cohort}.parquet
  clock_mortality_cindex.json
  clock_mortality_cindex_{cohort}.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from train_dann_survival import harrell_c_index
from train_vae_cox_lite import (
    _get_truncated_feature_names,
    load_bundle_with_cache,
)

CLOCK_DIR = _ROOT / "feature_importance" / "annot" / "clock_lists"


def _load_linear_clock(path: Path) -> Tuple[Dict[str, float], float]:
    df = pd.read_csv(path)
    cpg_col = "CpGmarker" if "CpGmarker" in df.columns else df.columns[0]
    coef_col = "CoefficientTraining" if "CoefficientTraining" in df.columns else df.columns[1]
    intercept = 0.0
    coefs: Dict[str, float] = {}
    for _, row in df.iterrows():
        cpg = str(row[cpg_col]).strip()
        val = float(row[coef_col])
        if cpg.lower() == "intercept":
            intercept = val
        else:
            coefs[cpg] = val
    return coefs, intercept


def _load_epiclock_clock(path: Path, intercept: float) -> Tuple[Dict[str, float], float]:
    """EpiClock probe list (Columns, Coefficient Estimate) + external intercept."""
    df = pd.read_csv(path)
    cpg_col = df.columns[0]
    coef_col = df.columns[1]
    coefs: Dict[str, float] = {}
    ic = float(intercept)
    for _, row in df.iterrows():
        cpg = str(row[cpg_col]).strip()
        if not cpg:
            continue
        if cpg.lower() == "intercept":
            ic = float(row[coef_col])
            continue
        if not cpg.startswith("cg"):
            continue
        coefs[cpg] = float(row[coef_col])
    return coefs, ic


def _dot_clock(beta: np.ndarray, idx: Dict[str, int], coefs: Dict[str, float], intercept: float) -> Tuple[np.ndarray, float]:
    keys = [k for k in coefs if k in idx]
    if not keys:
        return np.full(beta.shape[0], np.nan), 0.0
    cols = np.array([idx[k] for k in keys], dtype=np.int64)
    w = np.array([coefs[k] for k in keys], dtype=np.float64)
    score = intercept + beta[:, cols] @ w
    return score.astype(np.float64), len(keys) / len(coefs)


def _anti_trafo(x: np.ndarray, adult_age: float = 20.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x < 0,
        (1.0 + adult_age) * np.exp(x) - 1.0,
        (1.0 + adult_age) * x + adult_age,
    )


def _score_grimage_v2(
    beta: np.ndarray,
    idx: Dict[str, int],
    age: np.ndarray,
    female: np.ndarray,
    path: Path,
) -> Tuple[np.ndarray, float]:
    df = pd.read_csv(path)
    skip = {"COX", "transform", "nan"}
    components = sorted({str(x) for x in df["Y.pred"].unique() if str(x) not in skip})
    comp_scores: Dict[str, np.ndarray] = {}
    n_cpgs_total, n_cpgs_hit = 0, 0

    for comp in components:
        sub = df[df["Y.pred"] == comp]
        lin = np.zeros(beta.shape[0], dtype=np.float64)
        for _, row in sub.iterrows():
            var = str(row["var"]).strip()
            b = float(row["beta"])
            if var.lower() == "intercept":
                lin += b
            elif var.lower() == "age":
                lin += b * age
            elif var.startswith("cg") or var.startswith("ch."):
                n_cpgs_total += 1
                j = idx.get(var)
                if j is not None:
                    n_cpgs_hit += 1
                    lin += b * beta[:, j]
        comp_scores[comp] = lin

    cox_lin = np.zeros(beta.shape[0], dtype=np.float64)
    for _, row in df[df["Y.pred"] == "COX"].iterrows():
        var = str(row["var"]).strip()
        b = float(row["beta"])
        if var == "Age":
            cox_lin += b * age
        elif var == "Female":
            cox_lin += b * female
        elif var in comp_scores:
            cox_lin += b * comp_scores[var]

    tr = df[df["Y.pred"] == "transform"]
    params = {str(r["var"]).strip(): float(r["beta"]) for _, r in tr.iterrows()}
    grim = ((cox_lin - params["m_cox"]) / params["sd_cox"]) * params["sd_age"] + params["m_age"]
    return grim.astype(np.float64), n_cpgs_hit / max(n_cpgs_total, 1)


def _to_float64(arr) -> np.ndarray:
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=np.float64)


def score_all_clocks(
    beta: np.ndarray,
    cpg_names: List[str],
    age: np.ndarray,
    sex: np.ndarray,
    epiclock_csv: Optional[Path] = None,
    epiclock_intercept: float = 32.14246737,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    idx = {str(n): i for i, n in enumerate(cpg_names)}
    age = _to_float64(age)
    female = (_to_float64(sex) == 0).astype(np.float64)

    out: Dict[str, np.ndarray] = {}
    coverage: Dict[str, float] = {}

    coefs, ic = _load_linear_clock(CLOCK_DIR / "clock_horvath.csv")
    raw, cov = _dot_clock(beta, idx, coefs, ic)
    out["Horvath"] = _anti_trafo(raw + 0.696)
    coverage["Horvath"] = cov

    coefs, ic = _load_linear_clock(CLOCK_DIR / "clock_hannum.csv")
    out["Hannum"], coverage["Hannum"] = _dot_clock(beta, idx, coefs, ic)

    coefs, ic = _load_linear_clock(CLOCK_DIR / "clock_phenoage.csv")
    out["PhenoAge"], coverage["PhenoAge"] = _dot_clock(beta, idx, coefs, ic)

    coefs, ic = _load_linear_clock(CLOCK_DIR / "clock_dunedinpace.csv")
    out["DunedinPACE"], coverage["DunedinPACE"] = _dot_clock(beta, idx, coefs, ic)

    out["GrimAge"], coverage["GrimAge"] = _score_grimage_v2(
        beta, idx, age, female, CLOCK_DIR / "clock_grimagev2.csv",
    )

    if epiclock_csv is not None and epiclock_csv.is_file():
        coefs, ic = _load_epiclock_clock(epiclock_csv, epiclock_intercept)
        out["EpiClock"], coverage["EpiClock"] = _dot_clock(beta, idx, coefs, ic)

    for name in ("Horvath", "Hannum", "PhenoAge", "GrimAge", "EpiClock"):
        if name in out:
            out[f"{name}_accel"] = out[name] - age

    return pd.DataFrame(out), coverage


_THRIFT_LIMIT = 2_147_483_647


def _open_parquet(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(
            str(path),
            thrift_string_size_limit=_THRIFT_LIMIT,
            thrift_container_size_limit=_THRIFT_LIMIT,
        )
    except TypeError:
        return pq.ParquetFile(str(path))


def _load_cohort_meta(meta_parquet: Path) -> Tuple[np.ndarray, np.ndarray]:
    want = ("age", "sex")
    try:
        pf = _open_parquet(meta_parquet)
        meta_cols = [c for c in want if c in pf.schema_arrow.names]
        meta = pf.read(columns=meta_cols).to_pandas()
    except OSError:
        meta = pd.read_parquet(meta_parquet, columns=list(want), engine="fastparquet")
    age = pd.to_numeric(meta["age"], errors="coerce").to_numpy(dtype=np.float64)
    sex = meta["sex"].to_numpy() if "sex" in meta.columns else np.zeros(len(meta))
    return age, sex


def _cindex_for_series(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> Optional[float]:
    m = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    if m.sum() < 30 or event[m].sum() < 5:
        return None
    return float(harrell_c_index(time[m], event[m].astype(np.int32), risk[m]))


def evaluate_cindices(
    time: np.ndarray,
    event: np.ndarray,
    age: np.ndarray,
    log_h: np.ndarray,
    clocks: pd.DataFrame,
    val_frac: float,
    seed: int,
    external: bool = False,
) -> Dict:
    predictors: Dict[str, np.ndarray] = {"AESurv_log_h": log_h, "chronological_age": age}
    for col in clocks.columns:
        predictors[col] = clocks[col].to_numpy(dtype=np.float64)

    results: Dict = {"predictors": {}}

    if external:
        results["n_total"] = int(len(time))
        results["n_train"] = None
        results["n_val"] = None
        for name, vec in predictors.items():
            results["predictors"][name] = {
                "cindex_all": _cindex_for_series(time, event, vec),
                "cindex_train": None,
                "cindex_val": None,
            }
        return results

    strat = event.astype(np.int32) if event.sum() >= 2 else None
    idx = np.arange(len(time))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=strat, shuffle=True,
    )
    results["n_total"] = int(len(time))
    results["n_train"] = int(len(tr_idx))
    results["n_val"] = int(len(va_idx))
    for name, vec in predictors.items():
        results["predictors"][name] = {
            "cindex_all": _cindex_for_series(time, event, vec),
            "cindex_train": _cindex_for_series(time[tr_idx], event[tr_idx], vec[tr_idx]),
            "cindex_val": _cindex_for_series(time[va_idx], event[va_idx], vec[va_idx]),
        }
    return results


def plot_cindex_bars(summary: Dict, cohort: str, out_path: Path, metric: str = "cindex_val") -> None:
    preds = summary["predictors"]
    display = {
        "AESurv_log_h": "AESurv log_h",
        "chronological_age": "Chronological age",
        "EpiClock": "EpiClock",
    }
    core = [
        "AESurv_log_h", "EpiClock", "GrimAge", "PhenoAge", "Hannum", "Horvath",
        "DunedinPACE", "chronological_age",
    ]
    order = [k for k in core if k in preds] + [
        k for k in preds if k not in core and not k.endswith("_accel")
    ]
    labels, vals, colors = [], [], []
    for k in order:
        entry = preds[k]
        val = entry.get(metric) if metric != "cindex_val" else entry.get("cindex_val")
        if val is None:
            val = entry.get("cindex_all")
        if val is None:
            continue
        labels.append(display.get(k, k))
        vals.append(val)
        colors.append("#c0392b" if k == "AESurv_log_h" else "#7f8c8d")
    if not labels:
        return

    ylab = "Harrell C-index"
    if metric == "cindex_val":
        ylab += f" ({cohort} validation, 15%)"
    else:
        ylab += f" ({cohort} full cohort)"

    fig, ax = plt.subplots(figsize=(13, 6.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=12)
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel(ylab, fontsize=13)
    n_note = f"n={summary['n_total']}"
    if summary.get("n_val"):
        n_note = f"train n={summary['n_train']}, val n={summary['n_val']}"
    ax.set_title(f"Mortality discrimination — {cohort}\n({n_note})", fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    if vals:
        ax.axhline(vals[0], color="#c0392b", ls="--", lw=0.8, alpha=0.5)
    ax.grid(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cohort(
    cohort: str,
    npz_path: Path,
    pq_path: Path,
    log_h: np.ndarray,
    epiclock_csv: Optional[Path],
    epiclock_intercept: float,
    val_frac: float,
    seed: int,
    external: bool,
) -> Tuple[pd.DataFrame, Dict]:
    X_meth, _, t, e, n_cpg, n_snp = load_bundle_with_cache(
        cohort, npz_path, pq_path, None, None, None, Path("vae_cox_cache"), True,
    )
    meth_names, _ = _get_truncated_feature_names(npz_path, None, n_cpg, n_snp)
    beta = np.asarray(X_meth, dtype=np.float64)
    assert len(beta) == len(log_h), f"{cohort}: log_h length {len(log_h)} != n={len(beta)}"

    age, sex = _load_cohort_meta(pq_path)
    clocks, coverage = score_all_clocks(
        beta, meth_names, age, sex, epiclock_csv, epiclock_intercept,
    )
    clocks["log_h"] = log_h
    clocks["age"] = age

    summary = evaluate_cindices(
        t, e, age, log_h,
        clocks.drop(columns=["log_h", "age"], errors="ignore"),
        val_frac, seed, external=external,
    )
    summary["cpg_coverage"] = coverage
    summary["cohort"] = cohort
    return clocks, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fhs-npz", type=str, default="FHS_methylation_with_snp_1milfeatures_combined_training.npz")
    p.add_argument("--fhs-pq", type=str, default="FHS_methylation_with_snp_1milfeatures.parquet")
    p.add_argument("--whi-npz", type=str, default="WHI_methylation_with_snp_merged_1milfeatures_combined_training.npz")
    p.add_argument("--whi-pq", type=str, default="WHI_methylation_with_snp_merged_1milfeatures.parquet")
    p.add_argument(
        "--epiclock-csv",
        type=str,
        default=r"G:\All_GEO_epiclock_datasets\4k data training final\6k probes.csv",
    )
    p.add_argument("--epiclock-intercept", type=float, default=32.14246737)
    p.add_argument("--bundle-dir", type=str, default="models/aesurv_final")
    p.add_argument("--proj-fhs-npz", type=str, default="vae_cox_cache/jl_proj/proj_FHS_b24e34c31a.npz")
    p.add_argument("--proj-whi-npz", type=str, default="vae_cox_cache/jl_proj/proj_WHI_b159d5bf88.npz")
    p.add_argument("--out-dir", type=str, default="feature_importance/bio_relevance/clocks")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--joint-analysis-dir", type=str, default=None)
    args = p.parse_args()

    import torch
    from bio_relevance.model_scores import pooled_predictions

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epiclock_path = Path(args.epiclock_csv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    joint_dir = Path(args.joint_analysis_dir) if args.joint_analysis_dir else None
    log_h_all, _, n_fhs, n_whi = pooled_predictions(
        Path(args.bundle_dir), Path(args.proj_fhs_npz), Path(args.proj_whi_npz), device,
        joint_analysis_dir=joint_dir,
    )
    log_h_fhs = log_h_all[:n_fhs]
    log_h_whi = log_h_all[n_fhs : n_fhs + n_whi]

    all_summaries: Dict[str, Dict] = {}

    clocks_fhs, sum_fhs = run_cohort(
        "FHS", Path(args.fhs_npz), Path(args.fhs_pq), log_h_fhs,
        epiclock_path, args.epiclock_intercept, args.val_frac, args.seed, external=False,
    )
    clocks_fhs.to_parquet(out_dir / "clock_scores_fhs.parquet", index=False)
    all_summaries["FHS"] = sum_fhs
    plot_cindex_bars(sum_fhs, "FHS", out_dir / "clock_mortality_cindex_fhs.pdf", metric="cindex_val")

    clocks_whi, sum_whi = run_cohort(
        "WHI", Path(args.whi_npz), Path(args.whi_pq), log_h_whi,
        epiclock_path, args.epiclock_intercept, args.val_frac, args.seed, external=True,
    )
    clocks_whi.to_parquet(out_dir / "clock_scores_whi.parquet", index=False)
    all_summaries["WHI"] = sum_whi
    plot_cindex_bars(sum_whi, "WHI", out_dir / "clock_mortality_cindex_whi.pdf", metric="cindex_all")

    metrics_path = Path(args.bundle_dir) / "run_metrics.json"
    if metrics_path.exists():
        ref = json.loads(metrics_path.read_text(encoding="utf-8"))
        all_summaries["reference_aesurv"] = {
            "train_cindex_fhs": ref.get("final_train_cindex"),
            "val_cindex_fhs": ref.get("final_val_cindex"),
            "test_cindex_whi": ref.get("final_whi_cindex"),
        }

    all_summaries["epiclock_coef_path"] = str(epiclock_path)
    all_summaries["epiclock_intercept"] = args.epiclock_intercept
    all_summaries["note"] = (
        "FHS: 15% validation split (seed=42). WHI: external cohort, full-sample C-index. "
        "EpiClock = intercept + sum(beta * coefficient) over 6k probes. GrimAge uses age + sex."
    )
    (out_dir / "clock_mortality_cindex.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8",
    )

    from bio_relevance.clock_methods_manifest import get_clock_methods_manifest

    manifest = get_clock_methods_manifest(
        epiclock_csv=str(epiclock_path),
        epiclock_intercept=args.epiclock_intercept,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    manifest["results_file"] = str(out_dir / "clock_mortality_cindex.json")
    (out_dir / "clock_methods.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    from bio_relevance.plot_clock_mortality_comparison import plot_clock_mortality_comparison

    plot_clock_mortality_comparison(out_dir / "clock_mortality_cindex.json", out_dir)

    for cohort, sm in (("FHS", sum_fhs), ("WHI", sum_whi)):
        key = "cindex_val" if cohort == "FHS" else "cindex_all"
        rows = {k: v.get(key) for k, v in sm["predictors"].items() if v.get(key) is not None}
        print(f"\n=== {cohort} ({key}) ===")
        print(json.dumps(rows, indent=2))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
