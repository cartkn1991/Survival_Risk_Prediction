#!/usr/bin/env python3
"""
Domain-adversarial AESURV model with a gradient reversal layer (GRL)
for cross-cohort survival risk prediction (FHS -> WHI).

This module defines:
  - GradientReversalFunction / GradientReversal layer
  - DomainAdversarialAESurv model (VAE + survival head + domain head)
  - MultiCohortDataset and simple FHS/WHI datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch.autograd import Function

from aesurv_model import cox_ph_loss, compute_survival_metrics


class GradientReversalFunction(Function):
    """
    Identity in forward pass, multiplies gradient by -lambda in backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lambda_ = ctx.lambda_
        return -lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """
    Gradient reversal layer (GRL) with dynamically adjustable lambda.
    """

    def __init__(self, lambda_: float = 0.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)


class ResidualBlockNoBN(nn.Module):
    """
    Residual MLP block without BatchNorm to avoid domain leakage in BN statistics.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.dropout = nn.Dropout(p=dropout)
        self.use_shortcut = in_dim != out_dim
        self.shortcut = nn.Linear(in_dim, out_dim) if self.use_shortcut else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = self.act(out)
        out = self.dropout(out)
        if self.use_shortcut:
            x = self.shortcut(x)
        out = out + x
        out = self.act(out)
        return out


class FeatureWhitening(nn.Module):
    """
    Per-domain feature normalization to reduce batch/cohort effects before the encoder.

    Critically, this is *domain-specific* BatchNorm:
      - FHS (domain=0) uses its own running mean/var and affine params.
      - WHI (domain=1) uses a separate BatchNorm with independent stats/params.

    This avoids leakage between cohorts that would occur with a single shared BatchNorm.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Separate running stats + affine for each domain
        self.bn_fhs = nn.BatchNorm1d(
            input_dim, affine=True, track_running_stats=True
        )
        self.bn_whi = nn.BatchNorm1d(
            input_dim, affine=True, track_running_stats=True
        )

    def forward(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        """domain: 0 = FHS, 1 = WHI."""
        if domain == 0:
            return self.bn_fhs(x)
        return self.bn_whi(x)


class DomainAdversarialAESurv(nn.Module):
    """
    VAE-based survival model with a domain-adversarial head.

    Components:
      - Encoder: residual MLP -> (mu, logvar) -> latent z (deterministic: z = mu)
      - Decoder: residual MLP -> reconstruction of input
      - Survival head: linear from latent -> log-hazard
      - Domain head: GRL -> MLP -> domain logit
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: Tuple[int, ...],
        latent_dim: int = 16,
        decoder_dims: Optional[Tuple[int, ...]] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if decoder_dims is None:
            decoder_dims = tuple(reversed(encoder_dims))

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in encoder_dims:
            enc_layers.append(ResidualBlockNoBN(prev, h, dropout=dropout))
            prev = h
        self.encoder_net = nn.Sequential(*enc_layers)
        self.encoder_mu = nn.Linear(prev, latent_dim)
        self.encoder_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in decoder_dims:
            dec_layers.append(ResidualBlockNoBN(prev, h, dropout=dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

        # Survival head (no GRL here)
        self.surv_head = nn.Linear(latent_dim, 1)

        # Domain head (after GRL)
        self.grl = GradientReversal(lambda_=0.0)
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = mu  # deterministic latent
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          - x_recon
          - z
          - mu
          - logvar
          - log_hazard
          - domain_logit
        """
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        log_hazard = self.surv_head(z).squeeze(-1)
        log_hazard = torch.clamp(log_hazard, min=-10.0, max=10.0)

        z_reversed = self.grl(z)
        domain_logit = self.domain_head(z_reversed).squeeze(-1)

        return {
            "x_recon": x_recon,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "log_hazard": log_hazard,
            "domain_logit": domain_logit,
        }


def vae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Standard VAE KL divergence term.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


class FhsSurvDataset(Dataset):
    """
    FHS survival dataset with domain label 0.
    """

    def __init__(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> None:
        assert X.ndim == 2
        self.X = torch.from_numpy(X.astype(np.float32))
        self.time = torch.from_numpy(time.astype(np.float32))
        self.event = torch.from_numpy(event.astype(np.float32))
        self.domain = torch.zeros(len(self.X), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.time[idx],
            self.event[idx],
            self.domain[idx],
        )


class WhiDomainDataset(Dataset):
    """
    WHI dataset for domain classification only (no survival labels).

    Returns only (x, domain) so that the default collate function never sees
    None values.
    """

    def __init__(self, X: np.ndarray) -> None:
        assert X.ndim == 2
        self.X = torch.from_numpy(X.astype(np.float32))
        self.domain = torch.ones(len(self.X), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.domain[idx]


class MultiCohortDataset(Dataset):
    """
    Combined FHS + WHI dataset with cohort-aware outputs.

    __getitem__ returns:
      - x
      - time (or None for WHI)
      - event (or None for WHI)
      - domain (0=FHS, 1=WHI)
    """

    def __init__(
        self,
        X_fhs: np.ndarray,
        time_fhs: np.ndarray,
        event_fhs: np.ndarray,
        X_whi: np.ndarray,
        time_whi: Optional[np.ndarray] = None,
        event_whi: Optional[np.ndarray] = None,
    ) -> None:
        self.X_fhs = torch.from_numpy(X_fhs.astype(np.float32))
        self.time_fhs = torch.from_numpy(time_fhs.astype(np.float32))
        self.event_fhs = torch.from_numpy(event_fhs.astype(np.float32))

        self.X_whi = torch.from_numpy(X_whi.astype(np.float32))
        self.time_whi = (
            torch.from_numpy(time_whi.astype(np.float32))
            if time_whi is not None
            else None
        )
        self.event_whi = (
            torch.from_numpy(event_whi.astype(np.float32))
            if event_whi is not None
            else None
        )

        self.n_fhs = self.X_fhs.shape[0]
        self.n_whi = self.X_whi.shape[0]

    def __len__(self) -> int:
        return self.n_fhs + self.n_whi

    def __getitem__(self, idx: int):
        if idx < self.n_fhs:
            x = self.X_fhs[idx]
            time = self.time_fhs[idx]
            event = self.event_fhs[idx]
            domain = torch.tensor(0.0, dtype=torch.float32)
        else:
            j = idx - self.n_fhs
            x = self.X_whi[j]
            time = None
            event = None
            domain = torch.tensor(1.0, dtype=torch.float32)
        return x, time, event, domain


@dataclass
class DomainLossWeights:
    alpha: float
    beta: float
    gamma_kl: float
    lambda_domain: float

