# Training pipeline

Three-stage recipe for the AESURV joint survival model.

## Architecture

```
raw (CpG + SNP)
    → frozen JL/scaler
    → MiniVAEDANN (domain-adversarial encoder)
    → AESURV contrastive head
    → Cox log_h + age/cell aux + CpG↔SNP contrastive loss
```

## Quick start (joint training only)

Requires DANN + head checkpoints from earlier stages, or use bundled paths in `training/joint_default_config.json`.

```bash
pip install -r requirements.txt
pip install torch
export PYTHONPATH=.

python run_train.py --stage joint --device cuda
```

## Full recipe

```bash
# Stage 1: DANN pretrain (~hours, GPU recommended)
python run_train.py --stage dann --device cuda

# Stage 2: frozen-DANN AESURV head with contrastive aux
python run_train.py --stage head --device cuda

# Stage 3: joint end-to-end fine-tuning (epoch-26 model)
python run_train.py --stage joint --device cuda --out-dir runs/my_joint_run
```

Or all stages sequentially:

```bash
python run_train.py --stage all --device cuda
```

## Config

Edit `training/joint_default_config.json`:

| Key | Role |
|-----|------|
| `w_mmd`, `w_dom` | Cohort invariance (keep WHI transfer) |
| `lr_dann`, `lr_head` | Learning rates |
| `aux_age_weight`, `aux_cell_weight` | Auxiliary supervision |
| `init_head` | Warm-start from stage-2 head |

Direct invocation:

```bash
python train_aesurv_joint_dann_contrastive.py \
  --dann-encoder-ckpt runs/mini_vae_dann_mmd_rich/mini_dann_model.pt \
  --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz \
  --init-head runs/aesurv_contrastive_best/aesurv_aux_contrastive_model.pt \
  --out-dir runs/aesurv_joint_dann_contrastive --device cuda
```

## Outputs

- `joint_dann_aesurv_model.pt` — best validation checkpoint
- `joint_dann_aesurv_whi_top1_ep*.pt` — WHI top-k checkpoints
- `joint_dann_aesurv_run_meta.json` — metrics log

## Next step

Run downstream analysis on a saved checkpoint:

```bash
python run_pipeline.py --device cuda
```

See [ANALYSIS_README.md](../ANALYSIS_README.md).
