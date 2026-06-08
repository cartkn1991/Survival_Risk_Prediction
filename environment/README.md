# `snp_torch` environment

Conda env for AESurv / DANN / contrastive training on GPU.

## One-time setup

```powershell
cd D:\SNP_datasets
.\environment\setup_snp_torch.ps1
```

Or manually:

```powershell
conda create -n snp_torch python=3.11 pip -y
conda activate snp_torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r D:\SNP_datasets\requirements-snp_torch.txt
```

## Every session

```powershell
conda activate snp_torch
cd D:\SNP_datasets
$env:PYTHONPATH="D:\SNP_datasets"
```

## Verify GPU

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Train joint model (example)

```powershell
python train_aesurv_joint_dann_contrastive.py `
  --dann-encoder-ckpt runs/mini_vae_dann_mmd_rich/mini_dann_model.pt `
  --dann-preprocess-npz runs/mini_vae_dann_mmd_rich/mini_dann_preprocess.npz `
  --init-head runs/aesurv_contrastive_best/aesurv_aux_contrastive_model.pt `
  --aux-age-weight 12 --aux-cell-weight 0.5 `
  --w-recon 0.5 --w-dom 1.2 --w-mmd 10.0 --balance-events `
  --device cuda --epochs 80 --out-dir runs/aesurv_joint_dann_contrastive_best
```

Do **not** install PyTorch into the `base` Anaconda env; use `snp_torch` only.
