# Multi-modal contrastive alignment (CpG ↔ SNP)

Built on the **same pipeline** as `train_aesurv_dann_latent_aux_risk_recon.py` — only the aux head differs (InfoNCE instead of risk→latent recon).

## Train (same flags as risk-recon)

```powershell
cd D:\SNP_datasets
$env:PYTHONPATH="D:\SNP_datasets"

python train_aesurv_dann_latent_aux_contrastive.py `
  --dann-encoder-ckpt runs/mini_vae_dann_mmd_rich/mini_dann_model.pt `
  --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz `
  --init-from-baseline experiments/aesurv_risk_recon/checkpoints_baseline/aesurv_aux_model.pt `
  --aux-age-weight 12.0 --aux-cell-weight 1.0 `
  --aux-contrast-weight 1.0 --contrast-tau 0.07 `
  --z-dim 8 --seed 42 --balance-events `
  --device cpu `
  --out-dir runs/aesurv_aux_contrastive_v1
```

Use `--device cuda` only if PyTorch has CUDA; otherwise `--device cpu` (same as risk-recon on CPU-only torch).

## What changed vs the old contrastive script

| Issue | Fix |
|-------|-----|
| Different CLI (`--init-from` vs `--init-from-baseline`) | Matches risk-recon defaults |
| Missing `Loading FHS / WHI bundles...` flow | Same `load_bundle_with_cache` kwargs |
| Fragile forward unpacking | `forward(x, x_cpg=, x_snp=)` returns 9 tensors like risk-recon’s 9 |
| Extra TF-only path | Optional `train_aesurv_dann_latent_aux_contrastive_tf.py` if you need TF GPU |

## Loss

`L = L_cox + α L_recon + β L_KL + γ L_adv + 12·L_age + 1·L_cell + λ₃ L_InfoNCE`

Fused DANN `mu` drives survival; `cpgs_only` / `snps_only` DANN latents drive contrastive only.

## Results (C-index)

| Model | FHS val C | WHI C (best-val ckpt) | Peak WHI (exploratory) |
|--------|-----------|------------------------|-------------------------|
| Baseline age12 cell1 (`runs/aesurv_aux_grid/age12.00_cell1.00/`) | 0.777 | 0.644 | — |
| Contrastive v1 (`runs/aesurv_aux_contrastive_v1/`) | 0.789 | 0.654 | — |
| Tune grid **A5** (FHS-val pick, `best_config.json`) | **0.790** | 0.646 | 0.658 |
| Tune grid **A3** (WHI pick, `best_config_a3.json`) | 0.789 | **0.658** | **0.662** |

**Canonical WHI deployment:** `runs/aesurv_contrastive_best/` (A3 rerun, seed 42, `aux_cell_weight=0.5`).

### Train A3 (tune-grid WHI winner)

```powershell
python train_aesurv_dann_latent_aux_contrastive.py `
  --dann-encoder-ckpt runs/mini_vae_dann_mmd_rich/mini_dann_model.pt `
  --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz `
  --init-from experiments/aesurv_risk_recon/checkpoints_baseline/aesurv_aux_model.pt `
  --aux-age-weight 12 --aux-cell-weight 0.5 `
  --aux-contrast-weight 1.0 --contrast-tau 0.07 --contrast-warmup-epochs 10 --contrast-proj-dim 64 `
  --z-dim 8 --seed 42 --balance-events --device cuda `
  --epochs 200 --patience 25 `
  --out-dir runs/aesurv_contrastive_tune/aux/A3
```

### Multi-seed ensemble (A3 config)

```powershell
python experiments/aesurv_contrastive/run_tune_seeds.py `
  --config runs/aesurv_contrastive_tune/best_config_a3.json `
  --out-root runs/aesurv_contrastive_ensemble_a3 `
  --device cuda --epochs 200
```

Outputs: `runs/aesurv_contrastive_ensemble_a3/seed_*/` and `ensemble_summary.json`.

Latest 5-seed rank-mean ensemble (A3 config): WHI C **0.647**, FHS val C **0.822** (`runs/aesurv_contrastive_ensemble_a3/ensemble_summary.json`). Single seed 42 remains best for WHI (0.658).

---

## Joint DANN + AESurv + contrastive (end-to-end)

**Frozen pipeline (above):** DANN latents are precomputed once; only the AESurv head trains. Cox gradients never reach the DANN encoder.

**Joint pipeline:** Raw (meth, SNP) → frozen JL/scaler → **trainable full `MiniVAEDANN`** → μ → **trainable `AESurvHeadAuxContrastive`**. Backprop includes:

- Head: Cox, recon/KL, cohort GRL, age/cell, InfoNCE (same as frozen run)
- DANN: recon + domain BCE + cohort/batch MMD (+ optional VAE KL), with weights from `mini_vae_dann_mmd_rich` run meta

This can recover prognosis signal that domain-invariance may have removed during DANN pretrain (no Cox in DANN stage). Risk: FHS val ↑ but WHI ↓ if `w_mmd` / `w_dom` are too low—tune those before `lr_dann`.

| Component | Trainable? |
|-----------|------------|
| `InvariantPreprocessor` (W, scaler) | No |
| `MiniVAEDANN` (encoder, decoder, domain/batch heads) | **Yes** |
| `AESurvHeadAuxContrastive` | **Yes** |

### Train joint (A3 init)

```powershell
python experiments/aesurv_contrastive/run_joint_train.py `
  --config experiments/aesurv_contrastive/joint_default_config.json `
  --out-dir runs/aesurv_joint_dann_contrastive_a3 `
  --device cuda
```

Or directly:

```powershell
python train_aesurv_joint_dann_contrastive.py `
  --dann-encoder-ckpt runs/mini_vae_dann_mmd_rich/mini_dann_model.pt `
  --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz `
  --init-head runs/aesurv_contrastive_best/aesurv_aux_contrastive_model.pt `
  --aux-age-weight 12 --aux-cell-weight 0.5 `
  --w-recon 0.5 --w-dom 0.6 --w-mmd 5.0 --w-mmd-batch 2.0 `
  --lr-dann 1e-4 --lr-head 3e-4 `
  --balance-events --device cuda --epochs 80 `
  --out-dir runs/aesurv_joint_dann_contrastive_a3
```

Outputs: `joint_dann_aesurv_model.pt`, `joint_dann_aesurv_run_meta.json`, WHI top-k ckpts.

### Ablation (frozen vs joint vs high-MMD)

```powershell
python experiments/aesurv_contrastive/run_joint_ablation.py `
  --epochs 80 --device cuda `
  --out-json runs/aesurv_joint_ablation/comparison.json
```

Use `--skip-train` to refresh `comparison.json` from existing run folders only.

### Key hyperparameters

| Flag | Default | Role |
|------|---------|------|
| `--w-mmd` | 5.0 | Cohort MMD on DANN μ (keep WHI transfer) |
| `--w-dom` | 0.6 | Domain adversary scale |
| `--w-recon` | 0.5 | DANN reconstruction in projected space |
| `--lr-dann` | 1e-4 | DANN learning rate (lower than head) |
| `--lr-head` | 3e-4 | AESurv head learning rate |

Code: [`joint_dann_aesurv_model.py`](joint_dann_aesurv_model.py), [`train_aesurv_joint_dann_contrastive.py`](../../train_aesurv_joint_dann_contrastive.py).
