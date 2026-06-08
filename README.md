# Survival Risk Prediction

Cross-cohort **AESURV** mortality model on paired CpG methylation + SNP profiles (FHS train/val, WHI external test), with full training and downstream biological validation pipelines.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Model performance (epoch 26)

| Cohort | C-index |
|--------|---------|
| FHS train | 0.862 |
| FHS validation | 0.832 |
| WHI test | 0.699 |

Architecture: DANN encoder → AESURV head → Cox `log_h` (+ age/cell auxiliaries + CpG↔SNP contrastive alignment).

## Install

```bash
git clone https://github.com/cartkn1991/Survival_Risk_Prediction.git
cd Survival_Risk_Prediction
pip install -r requirements.txt
pip install torch   # match your CUDA version
export PYTHONPATH=.
```

## Training

```bash
# Joint end-to-end training (recommended)
python run_train.py --stage joint --device cuda

# Full recipe: DANN → head → joint
python run_train.py --stage all --device cuda
```

Details: [training/README.md](training/README.md)

## Downstream analysis

After placing a trained checkpoint in your analysis directory (see [DATA.md](DATA.md)):

```bash
python run_pipeline.py --device cuda
```

Stages: `downstream` · `gwas` · `ewas` · `figures` · `all`

Details: [ANALYSIS_README.md](ANALYSIS_README.md)

## Repository layout

| Path | Description |
|------|-------------|
| `run_train.py` | **Training entry point** |
| `run_pipeline.py` | **Analysis entry point** |
| `training/` | Joint training config + launcher |
| `pipeline/` | Analysis stage orchestrators |
| `scripts/` | Analysis helper modules |
| `bio_relevance/` | Clocks, lifestyle, pathway enrichment |
| `gwas/` | GWAS / EWAS utilities |
| `experiments/aesurv_contrastive/` | Model definitions |
| `config/` | Model manifest template |

## Data

Cohort data and checkpoints are **not included** (size + access restrictions). See [DATA.md](DATA.md).

## Maintainers

Sync implementation updates from a full local workspace:

```bash
python prepare_release.py
```

## Citation

If you use this code, please cite the associated AESURV manuscript (TBD).

## License

MIT — see [LICENSE](LICENSE).
