# Recursive feature elimination (risk-significant features)

Find the **smallest subset** of risk-selected features (from `significance/significant_risk.csv`) that preserves FHS validation C-index within a tolerance of the **full-feature** frozen AESurv-DANN-Aux model.

## Method

- **Frozen model** — no retraining; inactive features are set to the DANN scaler **mean** (same as CpG-only / SNP-only ablation).
- **Non-significant features** remain **active** (real values) throughout; only the significant pool is toggled.
- **Baseline** — all 1.37M features active (FHS val C ≈ 0.777).
- **Target** — `C_val >= C_full_val - cindex_tol` (default tol = 0.005).
- **Selection metric** — Harrell C-index on the **FHS validation split** (15%, seed 42, stratified by event).

### Backward elimination

Remove significant features in order of smallest `abs_mean_grad` (weakest risk gradient first), `step_size` features per step. Stop when C-index falls below threshold; minimal set = state before last failed step.

### Forward selection + binary refinement (recommended)

Start with all significant features **off** (scaler mean), then **add** features in descending `abs_mean_grad`. Coarse pass adds `step_size` features per step until threshold is met. Binary search between the last failing and first passing counts finds the exact minimal `n`.

| `--mode` | Behavior |
|----------|----------|
| `best` (default) | Forward + binary refine |
| `forward` | Same as `best` |
| `backward` | Backward elimination only |
| `both` | Run backward and forward; compare in `rfe_summary.json` |

## Run

```powershell
cd D:\SNP_datasets
$env:PYTHONPATH="D:\SNP_datasets"
python feature_importance/rfe_significant_features.py `
  --mode best `
  --sig-csv feature_importance/significance/significant_risk.csv `
  --bundle-dir models/aesurv_final `
  --step-size 500 `
  --cindex-tol 0.005 `
  --device cuda
```

Faster coarse search: `--step-size 1000`. Stricter match: `--cindex-tol 0.002`.

Use published baseline without recomputing: `--baseline-cindex 0.7773`

## Outputs

| File | Description |
|------|-------------|
| `rfe_forward_steps.csv` | Coarse + refine steps: `n_active_sig`, `cindex_val`, `cindex_whi`, `phase` |
| `rfe_forward_minimal_features.csv` | Minimal active significant feature list |
| `rfe_forward_curve.pdf` | C-index vs `n_active_sig` (forward) |
| `rfe_forward_summary.json` | Minimal counts, C-index, runtime |
| `rfe_summary.json` | Combined summary; `recommended_method: "forward"` |
| `rfe_steps.csv` | Backward steps (when `--mode backward` or `both`) |
| `rfe_minimal_features.csv` | Backward minimal list |
| `rfe_curve.pdf` | Backward C-index curve |

## Interpretation

### Backward (step-size 1000, tol 0.005)

| Setting | FHS val C-index |
|---------|----------------:|
| Full model (all 1.37M features) | 0.777 |
| After removing 1000 weakest significant | 0.754 (below threshold) |

→ Cannot drop even 1000 weakest features without exceeding tolerance. Minimal backward set ≈ **all 13,687** mapped significant features.

### Forward + binary refine (step-size 500, tol 0.005)

| Setting | FHS val C-index |
|---------|----------------:|
| 0 significant features active | 0.684 |
| First coarse pass at threshold | 13,500 active → 0.773 |
| **Minimal after binary refine** | **13,450 active → 0.774** |

→ Forward selection identifies **237 redundant** features at the tail of the gradient-ranked list (weakest ~1.7% of the pool). The model still requires **~98%** of mapped significant features under frozen masking—not a compact panel, but forward does bound redundancy more sharply than backward.

**Paper narrative:** contrast backward (“cannot drop 1000 weakest”) with forward (“237 weakest can stay off when top 13,450 are on”) to show joint contribution across the gradient-ranked list, with limited redundancy only in the tail.

**Caveat:** frozen masking, not retrained model—same limitation as modality ablation. WHI C-index in step CSVs is **monitoring only**; selection uses FHS val to avoid test leakage.

## Requirements

- `FHS_*_combined_training.npz`, `WHI_*_combined_training.npz`
- `models/aesurv_final/` (`dann_encoder.pt`, `aesurv_head.pt`, `dann_preprocess.npz`)
- `runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz` (or bundle copy)
