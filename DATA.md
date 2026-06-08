# Data and checkpoints

This repository contains **code only**. Cohort data and trained weights are not redistributed here.

## Required inputs

### 1. Cohort bundles (not included)

Per-cohort NPZ + parquet metadata for FHS and WHI methylation+SNP merged features. Paths are configured in `train_vae_cox_lite.py` defaults and training scripts.

Contact the Framingham Heart Study and WHI data access committees for raw data.

### 2. Pretrained checkpoints (not included)

| Artifact | Typical path | Produced by |
|----------|--------------|-------------|
| DANN encoder | `runs/mini_vae_dann_mmd_rich/mini_dann_model.pt` | `run_train.py --stage dann` |
| DANN preprocess | `runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz` | same |
| Contrastive head (optional init) | `runs/aesurv_contrastive_best/aesurv_aux_contrastive_model.pt` | `run_train.py --stage head` |
| Joint model | `runs/.../joint_dann_aesurv_model.pt` | `run_train.py --stage joint` |

For downstream analysis, copy a joint checkpoint to your analysis directory as `joint_model_epoch26.pt` and fill in `config/model_manifest.example.json` → `runs/.../model_manifest.json`.

### 3. External references (included, small)

- `external_phenotypes/` — clock probe lists and coefficients
- `feature_importance/annot/` — SNP/CpG annotation caches (code-side tables only)

## Published model metrics (epoch 26)

| Cohort | C-index |
|--------|---------|
| FHS train | 0.862 |
| FHS validation | 0.832 |
| WHI test | 0.699 |

See `config/cindex_summary.json`.
