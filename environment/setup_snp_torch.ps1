# Create snp_torch env with CUDA PyTorch + SNP_datasets dependencies.
# Usage (PowerShell):  cd D:\SNP_datasets; .\environment\setup_snp_torch.ps1

$ErrorActionPreference = "Stop"
$EnvName = "snp_torch"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "=== Creating conda env: $EnvName (Python 3.11) ===" -ForegroundColor Cyan
conda create -n $EnvName python=3.11 pip -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Installing PyTorch + CUDA 12.4 (pip) ===" -ForegroundColor Cyan
conda run -n $EnvName pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Installing project dependencies ===" -ForegroundColor Cyan
conda run -n $EnvName pip install -r "$Root\requirements-snp_torch.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Verifying CUDA ===" -ForegroundColor Cyan
conda run -n $EnvName python -c @"
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device='cuda')
    print('matmul ok', (x @ x).shape)
"@

Write-Host ""
Write-Host "Done. Activate with:" -ForegroundColor Green
Write-Host "  conda activate $EnvName"
Write-Host "  cd $Root"
Write-Host '  $env:PYTHONPATH="' + $Root + '"'
