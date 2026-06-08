"""AESURV head + CpG/SNP InfoNCE contrastive (extends baseline aux like risk-recon variant)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_aesurv_dann_latent import grad_reverse
from train_aesurv_dann_latent_aux import AESurvHeadAux


class ModalityProjectionHead(nn.Module):
    def __init__(self, z_dim: int, proj_dim: int, hidden: int = 0, dropout: float = 0.30) -> None:
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(z_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, proj_dim),
            )
        else:
            self.net = nn.Linear(z_dim, proj_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AESurvHeadAuxContrastive(AESurvHeadAux):
    """Same fused AESURV path as baseline; optional CpG/SNP branch latents for InfoNCE."""

    def __init__(
        self,
        in_dim: int,
        enc_hidden: Tuple[int, ...] = (64, 32),
        dec_hidden: Tuple[int, ...] = (32, 64),
        z_dim: int = 16,
        cohort_hidden: int = 8,
        dropout: float = 0.30,
        n_cells: int = 6,
        contrast_proj_dim: int = 64,
        contrast_hidden: int = 32,
        contrast_tau: float = 0.07,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden,
            z_dim=z_dim,
            cohort_hidden=cohort_hidden,
            dropout=dropout,
            n_cells=n_cells,
        )
        self.contrast_tau = float(contrast_tau)

        enc_layers: list[nn.Module] = []
        prev = self.in_dim
        for h in enc_hidden:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.modality_encoder = nn.Sequential(*enc_layers)
        self.modality_mu = nn.Linear(prev, self.z_dim)

        self.contrast_proj_cpg = ModalityProjectionHead(
            self.z_dim, contrast_proj_dim, hidden=contrast_hidden, dropout=dropout,
        )
        self.contrast_proj_snp = ModalityProjectionHead(
            self.z_dim, contrast_proj_dim, hidden=contrast_hidden, dropout=dropout,
        )

    def encode_modality(self, x: torch.Tensor) -> torch.Tensor:
        h = self.modality_encoder(x)
        return self.modality_mu(h)

    def forward(
        self,
        x: torch.Tensor,
        sample_z: bool = True,
        x_cpg: torch.Tensor | None = None,
        x_snp: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar) if (sample_z and self.training) else mu
        x_rec = self.decoder(z)
        log_h = self.cox_head(z).squeeze(-1)
        age_pred = self.age_head(z).squeeze(-1)
        cell_pred = self.cell_head(z)

        p_cpg: torch.Tensor | None = None
        p_snp: torch.Tensor | None = None
        if x_cpg is not None and x_snp is not None:
            z_cpg = self.encode_modality(x_cpg)
            z_snp = self.encode_modality(x_snp)
            p_cpg = self.contrast_proj_cpg(z_cpg)
            p_snp = self.contrast_proj_snp(z_snp)

        return x_rec, z, log_h, mu, logvar, age_pred, cell_pred, p_cpg, p_snp

    def cohort_logits(self, z: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.cohort_head(grad_reverse(z, lam))

    @staticmethod
    def info_nce_loss(proj_a: torch.Tensor, proj_b: torch.Tensor, tau: float) -> torch.Tensor:
        if proj_a.shape[0] < 2:
            return proj_a.new_zeros(())
        a = F.normalize(proj_a, dim=-1)
        b = F.normalize(proj_b, dim=-1)
        logits = (a @ b.T) / max(float(tau), 1e-6)
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
