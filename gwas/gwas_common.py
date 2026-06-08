"""Shared utilities for FHS discovery -> WHI replication -> outlier-stratification GWAS."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.model_selection import train_test_split

from pipeline_ablation_aesurv_modalities import _pick_sex_column, _sex_to_female01

_THRIFT_LIMIT = 2_147_483_647
_SNP_RE = re.compile(r"^(chr)?([0-9XYMTxymt]+):([0-9]+)")


def open_pq(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(
            str(path),
            thrift_string_size_limit=_THRIFT_LIMIT,
            thrift_container_size_limit=_THRIFT_LIMIT,
        )
    except TypeError:
        return pq.ParquetFile(str(path))


def read_txt_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n\r") for line in fh if line.strip()]


def parse_chr_pos(snp_name: str) -> Tuple[int, int]:
    m = _SNP_RE.match(str(snp_name))
    if not m:
        return 26, -1
    chrom_raw = m.group(2).upper()
    pos = int(m.group(3))
    if chrom_raw.isdigit():
        return int(chrom_raw), pos
    if chrom_raw == "X":
        return 23, pos
    if chrom_raw == "Y":
        return 24, pos
    return 25, pos


def read_age(parquet_path: Path, id_col: str) -> np.ndarray:
    pf = open_pq(parquet_path)
    df = pf.read(columns=[id_col, "age"]).to_pandas()
    age = df["age"].to_numpy(dtype=np.float32)
    if np.isnan(age).any():
        age = np.where(np.isnan(age), np.nanmean(age), age)
    return age


def read_age_sex(parquet_path: Path, id_col: str) -> Tuple[np.ndarray, np.ndarray]:
    pf = open_pq(parquet_path)
    sch = getattr(pf, "schema_arrow", None) or pf.schema
    names = list(sch.names)
    cols = [id_col, "age"]
    sex_src = _pick_sex_column(names)
    if sex_src:
        cols.append(sex_src)
    df = pf.read(columns=cols).to_pandas()
    age = df["age"].to_numpy(dtype=np.float32)
    if np.isnan(age).any():
        age = np.where(np.isnan(age), np.nanmean(age), age)
    if sex_src:
        sex = np.array([_sex_to_female01(v) for v in df[sex_src]], dtype=np.float32)
    else:
        sex = np.full(len(df), 0.5, dtype=np.float32)
    return age, sex


def load_fhs_risk_full(
    risk_npz: Path,
    n_fhs: int,
    event: np.ndarray,
    *,
    seed: int = 42,
    val_frac: float = 0.15,
) -> np.ndarray:
    rd = dict(np.load(risk_npz, allow_pickle=False))
    risk_whi = rd["risk_whi_test"]
    idx = np.arange(n_fhs)
    strat = event if event.sum() >= 2 else None
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=strat, shuffle=True,
    )
    risk_fhs = np.zeros(n_fhs, dtype=np.float32)
    if len(tr_idx) == rd["risk_fhs_train"].size and len(va_idx) == rd["risk_fhs_val"].size:
        risk_fhs[tr_idx] = rd["risk_fhs_train"]
        risk_fhs[va_idx] = rd["risk_fhs_val"]
    else:
        risk_fhs[: rd["risk_fhs_train"].size] = rd["risk_fhs_train"]
        tail = rd["risk_fhs_train"].size
        risk_fhs[tail : tail + rd["risk_fhs_val"].size] = rd["risk_fhs_val"]
    return risk_fhs, risk_whi


def residualize(y: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (residuals, coefficients) for y ~ cov (cov includes intercept)."""
    coef, *_ = np.linalg.lstsq(cov, y.astype(np.float64), rcond=None)
    resid = y.astype(np.float64) - cov @ coef
    return resid, coef


def compute_logh_residual(
    log_h: np.ndarray,
    age: np.ndarray,
    sex: Optional[np.ndarray] = None,
    cohort: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Age-adjusted log_h residual; optional sex and cohort (0/1) terms."""
    terms: List[Tuple[str, np.ndarray]] = [("intercept", np.ones(len(log_h), dtype=np.float64))]
    terms.append(("age", age.astype(np.float64)))
    if sex is not None:
        terms.append(("sex", sex.astype(np.float64)))
    if cohort is not None:
        terms.append(("cohort", cohort.astype(np.float64)))
    active = [terms[0]] + [(n, v) for n, v in terms[1:] if np.std(v) > 1e-8]
    cov = np.column_stack([v for _, v in active])
    resid, coef = residualize(log_h, cov)
    info = {active[i][0]: float(coef[i]) for i in range(len(active))}
    for name, _ in terms:
        info.setdefault(name, 0.0)
    return resid.astype(np.float32), info


@dataclass
class OutlierGroups:
    resilient: np.ndarray
    accelerated: np.ndarray
    middle: np.ndarray
    residual: np.ndarray
    info: Dict[str, float | str]


def define_outlier_groups(
    log_h: np.ndarray,
    age: np.ndarray,
    cohort: np.ndarray,
    *,
    residual_q_lo: float = 0.10,
    residual_q_hi: float = 0.90,
) -> OutlierGroups:
    """Three-way split on age+cohort-adjusted log_h residual (matches mini-GWAS)."""
    resid, info = compute_logh_residual(log_h, age, cohort=cohort)
    info["mode"] = "residual"
    info["residual_q_lo"] = residual_q_lo
    info["residual_q_hi"] = residual_q_hi
    lo = float(np.quantile(resid, residual_q_lo))
    hi = float(np.quantile(resid, residual_q_hi))
    info["resid_lo"] = lo
    info["resid_hi"] = hi
    resilient = resid <= lo
    accelerated = resid >= hi
    middle = ~(resilient | accelerated)
    return OutlierGroups(
        resilient=resilient,
        accelerated=accelerated,
        middle=middle,
        residual=resid,
        info=info,
    )


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    bh = p_sorted * m / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1].clip(0, 1)
    q = np.empty(m, dtype=np.float64)
    q[order] = bh
    return q


def _prepare_covariates(
    age: np.ndarray,
    sex: np.ndarray,
    cohort: Optional[np.ndarray] = None,
) -> np.ndarray:
    parts = [np.ones(len(age)), age.astype(np.float64)]
    if sex is not None:
        parts.append(sex.astype(np.float64))
    if cohort is not None:
        parts.append(cohort.astype(np.float64))
    return np.column_stack(parts)


def _drop_constant_covariates(cov: np.ndarray) -> np.ndarray:
    """Remove covariate columns with zero variance (keeps intercept if present)."""
    keep = [0]  # always keep first column (intercept)
    for j in range(1, cov.shape[1]):
        if np.std(cov[:, j]) > 1e-8:
            keep.append(j)
    return cov[:, keep] if len(keep) < cov.shape[1] else cov


def batch_linear_gwas(
    y: np.ndarray,
    G: np.ndarray,
    cov: np.ndarray,
    snp_names: Sequence[str],
    *,
    maf_min: float = 0.01,
    miss_max: float = 0.05,
    min_n: int = 100,
) -> pd.DataFrame:
    """Vectorized linear GWAS: y ~ SNP + cov for a batch of SNP columns."""
    cov = _drop_constant_covariates(cov.astype(np.float64))
    n, k = G.shape
    p_cov = cov.shape[1]
    df_resid = max(n - p_cov - 1, 1)

    y64 = y.astype(np.float64)
    y_res, _ = residualize(y64, cov)
    H, *_ = np.linalg.lstsq(cov.T @ cov, cov.T, rcond=None)  # (p, n)
    G64 = G.astype(np.float64)
    G64 = np.where((G64 >= 0) & (G64 <= 2), G64, np.nan)

    miss_frac = np.mean(np.isnan(G64), axis=0)
    G_imp = np.where(np.isnan(G64), np.nanmean(G64, axis=0, keepdims=True), G64)
    mean_g = np.nanmean(G_imp, axis=0)
    maf = np.minimum(mean_g / 2.0, 1.0 - mean_g / 2.0)
    var_g = np.var(G_imp, axis=0)

    coef_g = H @ G_imp
    G_res = G_imp - cov @ coef_g

    num = G_res.T @ y_res
    den = np.sum(G_res ** 2, axis=0)
    beta = np.divide(num, den, out=np.full(k, np.nan), where=den > 1e-12)
    resid_y = y_res[:, None] - G_res * beta[None, :]
    sse = np.sum(resid_y ** 2, axis=0)
    se = np.sqrt(np.divide(sse, df_resid * den, out=np.full(k, np.nan), where=den > 1e-12))
    z = np.divide(beta, se, out=np.full(k, np.nan), where=se > 0)
    p = 2.0 * stats.norm.sf(np.abs(z))

    valid = (
        (miss_frac <= miss_max)
        & (maf >= maf_min)
        & (var_g > 0)
        & np.isfinite(beta)
        & np.isfinite(se)
        & (se > 0)
        & (np.sum(~np.isnan(G64), axis=0) >= min_n)
    )

    chrom_pos = [parse_chr_pos(s) for s in snp_names]
    df = pd.DataFrame({
        "snp": list(snp_names),
        "chrom": [t[0] for t in chrom_pos],
        "pos": [t[1] for t in chrom_pos],
        "beta": beta,
        "se": se,
        "z": z,
        "p": p,
        "maf": maf,
        "n": n,
        "miss_frac": miss_frac,
        "valid": valid,
    })
    return df[df["valid"]].drop(columns=["valid"]).reset_index(drop=True)


def clump_leads(
    df: pd.DataFrame,
    p_col: str = "p",
    *,
    p_max: float = 1e-5,
    window_bp: int = 1_000_000,
    max_leads: int = 500,
) -> pd.DataFrame:
    """Greedy distance clumping on sorted p-values."""
    sub = df[df[p_col] <= p_max].sort_values([p_col, "chrom", "pos"]).reset_index(drop=True)
    if sub.empty:
        sub = df.nsmallest(max_leads, p_col).sort_values(p_col).reset_index(drop=True)
    picked: List[int] = []
    for i, row in sub.iterrows():
        if len(picked) >= max_leads:
            break
        ch, pos = int(row["chrom"]), int(row["pos"])
        clash = False
        for j in picked:
            r = sub.iloc[j]
            if int(r["chrom"]) == ch and abs(int(r["pos"]) - pos) <= window_bp:
                clash = True
                break
        if not clash:
            picked.append(i)
    return sub.iloc[picked].reset_index(drop=True)


def fixed_effect_meta(b1: float, se1: float, b2: float, se2: float) -> Tuple[float, float, float]:
    w1 = 1.0 / (se1 ** 2 + 1e-12)
    w2 = 1.0 / (se2 ** 2 + 1e-12)
    beta = (w1 * b1 + w2 * b2) / (w1 + w2)
    se = 1.0 / np.sqrt(w1 + w2)
    p = float(2.0 * stats.norm.sf(abs(beta / se)))
    return float(beta), float(se), p


def enrichment_vs_panel(
    discovery_snps: Iterable[str],
    panel_snps: Iterable[str],
    universe_size: int,
) -> Dict[str, float | int]:
    """Hypergeometric enrichment of discovery hits in a SNP panel."""
    disc = set(discovery_snps)
    panel = set(panel_snps)
    overlap = len(disc & panel)
    k = len(disc)
    n_panel = len(panel)
    # P(X >= overlap | universe U, panel size n_panel, draws k)
    p_enrich = float(stats.hypergeom.sf(overlap - 1, universe_size, n_panel, k))
    expected = k * n_panel / max(universe_size, 1)
    return {
        "n_discovery": k,
        "n_panel": n_panel,
        "n_overlap": overlap,
        "expected_overlap": expected,
        "fold_enrichment": overlap / max(expected, 1e-9),
        "p_hypergeom": p_enrich,
        "universe_size": universe_size,
    }


def load_genotype_columns(
    npz_path: Path,
    col_indices: np.ndarray,
) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False, mmap_mode="r")
    order = np.argsort(col_indices)
    G = z["X_snp"][:, col_indices[order]].astype(np.float32)
    inv = np.argsort(order)
    return G[:, inv]


def resolve_snp_columns(
    snp_names: Sequence[str],
    snp_txt: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    all_names = read_txt_list(snp_txt)
    name_to_idx = {s: i for i, s in enumerate(all_names)}
    cols = []
    resolved = []
    for s in snp_names:
        idx = name_to_idx.get(str(s))
        if idx is not None:
            cols.append(idx)
            resolved.append(str(s))
    return np.asarray(cols, dtype=np.int64), np.asarray(resolved, dtype=object)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def genomic_lambda(pvals: np.ndarray) -> float:
    p = np.clip(pvals, 1e-300, 1.0)
    chi2 = stats.chi2.isf(p, df=1)
    return float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))


def _parse_cpg_chrom(chrm: str) -> int:
    s = str(chrm).strip().lower().replace("chr", "")
    if s.isdigit():
        return int(s)
    if s == "x":
        return 23
    if s == "y":
        return 24
    if s in ("m", "mt"):
        return 25
    return 26


def load_cpg_annotation(annot_csv: Path) -> pd.DataFrame:
    """Load probeID -> chrom, pos, gene from Annotation.csv."""
    usecols = ["probeID", "CpG_chrm", "CpG_beg", "gene"]
    ann = pd.read_csv(annot_csv, usecols=usecols, low_memory=False)
    ann["cpg"] = ann["probeID"].astype(str)
    ann["chrom"] = ann["CpG_chrm"].map(_parse_cpg_chrom)
    ann["pos"] = pd.to_numeric(ann["CpG_beg"], errors="coerce").fillna(-1).astype(int)
    ann["gene"] = ann["gene"].astype(str).replace({"nan": ""})
    return ann[["cpg", "chrom", "pos", "gene"]].drop_duplicates("cpg", keep="first")


def annotate_cpg_df(df: pd.DataFrame, annot: pd.DataFrame, *, cpg_col: str = "cpg") -> pd.DataFrame:
    out = df.merge(annot, on=cpg_col, how="left", suffixes=("", "_ann"))
    if "chrom_ann" in out.columns:
        out["chrom"] = out["chrom"].fillna(out["chrom_ann"])
        out["pos"] = out["pos"].fillna(out["pos_ann"])
        out = out.drop(columns=[c for c in out.columns if c.endswith("_ann")])
    if "gene" not in out.columns:
        out["gene"] = ""
    out["chrom"] = out["chrom"].fillna(26).astype(int)
    out["pos"] = out["pos"].fillna(-1).astype(int)
    return out


def resolve_cpg_columns(cpg_names: Sequence[str], cpg_txt: Path) -> Tuple[np.ndarray, np.ndarray]:
    all_names = read_txt_list(cpg_txt)
    name_to_idx = {s: i for i, s in enumerate(all_names)}
    cols, resolved = [], []
    for c in cpg_names:
        idx = name_to_idx.get(str(c))
        if idx is not None:
            cols.append(idx)
            resolved.append(str(c))
    return np.asarray(cols, dtype=np.int64), np.asarray(resolved, dtype=object)


def load_methylation_columns(npz_path: Path, col_indices: np.ndarray) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False, mmap_mode="r")
    order = np.argsort(col_indices)
    G = z["X_meth"][:, col_indices[order]].astype(np.float32)
    inv = np.argsort(order)
    return G[:, inv]


def load_fhs_logh_from_parquet(
    meta_pq: Path,
    id_col: str,
    risk_parquet: Path,
    n_fhs: int,
) -> np.ndarray:
    """Align FHS log_h to bundle row order via metadata IDs."""
    pf = open_pq(meta_pq)
    meta = pf.read(columns=[id_col]).to_pandas()
    if len(meta) != n_fhs:
        raise ValueError(f"meta rows {len(meta)} != expected FHS n={n_fhs}")
    risk = pd.read_parquet(risk_parquet)
    risk = risk[risk["cohort"].astype(str) == "FHS"].copy()
    key = id_col if id_col in risk.columns else "Share_ID"
    def _id_str(s: pd.Series) -> pd.Series:
        def _one(v: object) -> str:
            if pd.isna(v):
                return ""
            try:
                fv = float(v)
                if fv == int(fv):
                    return str(int(fv))
            except (TypeError, ValueError):
                pass
            return str(v).strip()

        return s.map(_one)

    risk[key] = _id_str(risk[key])
    meta[id_col] = _id_str(meta[id_col])
    merged = meta.merge(risk[[key, "log_h"]], left_on=id_col, right_on=key, how="left")
    if merged["log_h"].isna().any():
        n_miss = int(merged["log_h"].isna().sum())
        raise ValueError(f"log_h missing for {n_miss} FHS rows after merge on {key}")
    return merged["log_h"].to_numpy(dtype=np.float32)


def load_pooled_logh_from_parquet(
    fhs_meta_pq: Path,
    fhs_id_col: str,
    whi_meta_pq: Path,
    whi_id_col: str,
    risk_parquet: Path,
    n_fhs: int,
    n_whi: int,
) -> np.ndarray:
    """Return concatenated FHS-then-WHI log_h aligned to bundle row order."""
    log_h_fhs = load_fhs_logh_from_parquet(fhs_meta_pq, fhs_id_col, risk_parquet, n_fhs)
    pf = open_pq(whi_meta_pq)
    meta = pf.read(columns=[whi_id_col]).to_pandas()
    if len(meta) != n_whi:
        raise ValueError(f"WHI meta rows {len(meta)} != expected {n_whi}")
    risk = pd.read_parquet(risk_parquet)
    risk = risk[risk["cohort"].astype(str) == "WHI"].copy()
    key = whi_id_col if whi_id_col in risk.columns else "sample_ID"

    def _id_str(s: pd.Series) -> pd.Series:
        def _one(v: object) -> str:
            if pd.isna(v):
                return ""
            try:
                fv = float(v)
                if fv == int(fv):
                    return str(int(fv))
            except (TypeError, ValueError):
                pass
            return str(v).strip()

        return s.map(_one)

    risk[key] = _id_str(risk[key])
    meta[whi_id_col] = _id_str(meta[whi_id_col])
    merged = meta.merge(risk[[key, "log_h"]], left_on=whi_id_col, right_on=key, how="left")
    if merged["log_h"].isna().any():
        raise ValueError(f"log_h missing for {int(merged['log_h'].isna().sum())} WHI rows")
    log_h_whi = merged["log_h"].to_numpy(dtype=np.float32)
    return np.concatenate([log_h_fhs, log_h_whi], axis=0)


def batch_linear_ewas(
    y: np.ndarray,
    G: np.ndarray,
    cov: np.ndarray,
    cpg_names: Sequence[str],
    *,
    miss_max: float = 0.05,
    min_std: float = 0.01,
    min_n: int = 100,
) -> pd.DataFrame:
    """Vectorized EWAS: y ~ methylation + cov for a batch of CpG columns."""
    cov = _drop_constant_covariates(cov.astype(np.float64))
    n, k = G.shape
    p_cov = cov.shape[1]
    df_resid = max(n - p_cov - 1, 1)

    y64 = y.astype(np.float64)
    y_res, _ = residualize(y64, cov)
    H, *_ = np.linalg.lstsq(cov.T @ cov, cov.T, rcond=None)
    G64 = G.astype(np.float64)
    G64 = np.where(np.isfinite(G64), G64, np.nan)

    miss_frac = np.mean(np.isnan(G64), axis=0)
    G_imp = np.where(np.isnan(G64), np.nanmean(G64, axis=0, keepdims=True), G64)
    mean_g = np.nanmean(G_imp, axis=0)
    std_g = np.nanstd(G_imp, axis=0)

    coef_g = H @ G_imp
    G_res = G_imp - cov @ coef_g

    num = G_res.T @ y_res
    den = np.sum(G_res ** 2, axis=0)
    beta = np.divide(num, den, out=np.full(k, np.nan), where=den > 1e-12)
    resid_y = y_res[:, None] - G_res * beta[None, :]
    sse = np.sum(resid_y ** 2, axis=0)
    se = np.sqrt(np.divide(sse, df_resid * den, out=np.full(k, np.nan), where=den > 1e-12))
    z = np.divide(beta, se, out=np.full(k, np.nan), where=se > 0)
    p = 2.0 * stats.norm.sf(np.abs(z))

    valid = (
        (miss_frac <= miss_max)
        & (std_g >= min_std)
        & np.isfinite(beta)
        & np.isfinite(se)
        & (se > 0)
        & (np.sum(~np.isnan(G64), axis=0) >= min_n)
    )

    df = pd.DataFrame({
        "cpg": list(cpg_names),
        "beta": beta,
        "se": se,
        "z": z,
        "p": p,
        "mean_beta": mean_g,
        "std_beta": std_g,
        "n": n,
        "miss_frac": miss_frac,
        "valid": valid,
    })
    return df[df["valid"]].drop(columns=["valid"]).reset_index(drop=True)
