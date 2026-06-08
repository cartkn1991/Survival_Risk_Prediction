#!/usr/bin/env python3
"""
Bridge mini_vae_dann_pipeline outputs into CpGSNPFusionSurv.

Loads:
  - ``mini_dann_preprocess.npz`` (scaler mean/scale, JL matrix ``W``, ``d_in``)
  - ``mini_dann_model.pt`` (``MiniVAEDANN`` state_dict + config)

Registers frozen modules on the fusion model:
  - ``dann_inv_pre``: (meth, snp) -> standardized concat @ W  -> ``proj_dim``
  - ``dann_latent_enc``: encoder + ``fc_mu`` from the pretrained DANN VAE
  - ``meth_from_dann``: trainable MLP mapping DANN latent -> ``meth_branch_dim``

The original high-capacity ``meth_project`` (Linear from raw CpGs) is **omitted** when this
stem is active to avoid duplicate million-parameter layers.

**Preconditions**
  - AESURV must see the same ``meth_input_dim + snp_input_dim`` as ``d_in`` in the NPZ
    (same NPZ alignment / ``--max-cpg`` / ``--max-snp`` as ``mini_vae_dann_pipeline.py``).
  - Methylation preprocessing must match: NPZ key ``meth_as_mvalues`` (if present) must
    agree with ``dann_meth_as_mvalues`` passed to ``CpGSNPFusionSurv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn as nn

from mini_vae_dann_pipeline import MiniVAEDANN

if TYPE_CHECKING:
    from aesurv_model import CpGSNPFusionSurv


def beta_to_m_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b = x.clamp(min=eps, max=1.0 - eps)
    return torch.log2(b / (1.0 - b))


class InvariantPreprocessor(nn.Module):
    """
    Frozen concat(meth, snp) -> sklearn-style (x - mean) / scale -> @ W.
    ``W`` shape ``(d_in, proj_dim)``; buffers moved with ``.to(device)``.
    """

    def __init__(self, npz_path: str | Path) -> None:
        super().__init__()
        z = np.load(npz_path, allow_pickle=False)
        mean = torch.from_numpy(np.asarray(z["scaler_mean"], dtype=np.float32))
        scale = torch.from_numpy(np.asarray(z["scaler_scale"], dtype=np.float32))
        w = torch.from_numpy(np.asarray(z["W"], dtype=np.float32))
        self.register_buffer("mean_", mean, persistent=False)
        self.register_buffer("scale_", scale.clamp_min(1e-8), persistent=False)
        self.register_buffer("W", w, persistent=False)
        self.d_in = int(mean.numel())
        self.proj_dim = int(w.shape[1])

    def forward(self, x_meth: torch.Tensor, x_snp: torch.Tensor, use_m_values: bool) -> torch.Tensor:
        if use_m_values:
            xm = beta_to_m_torch(x_meth.float())
        else:
            xm = x_meth.float()
        xs = x_snp.float()
        xc = torch.cat([xm, xs], dim=1)
        if xc.shape[1] != self.d_in:
            raise ValueError(
                f"InvariantPreprocessor expected d_in={self.d_in}, got concat width {xc.shape[1]}."
            )
        z = (xc - self.mean_) / self.scale_
        return z @ self.W


class DannLatentEncoder(nn.Module):
    """Frozen ``MiniVAEDANN.encoder`` + ``fc_mu`` (+ ``fc_logvar`` for weight loading)."""

    def __init__(self, mini: MiniVAEDANN) -> None:
        super().__init__()
        self.encoder = mini.encoder
        self.fc_mu = mini.fc_mu
        self.fc_logvar = mini.fc_logvar
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x_proj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_proj)
        mu = torch.clamp(self.fc_mu(h), -6.0, 6.0)
        return mu


def _torch_load_compat(path: str | Path, map_location: torch.device | str) -> dict:
    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)


def load_mini_dann_for_fusion(
    ckpt_path: str | Path,
    map_location: torch.device | str = "cpu",
) -> Tuple[MiniVAEDANN, dict]:
    ckpt = _torch_load_compat(ckpt_path, map_location)
    cfg = ckpt["config"]
    hidden = tuple(cfg["hidden"])
    n_fhs_batches = int(cfg.get("n_fhs_batches", 0))
    state = ckpt["state_dict"]
    if n_fhs_batches <= 0 and any(k.startswith("batch_head.") for k in state.keys()):
        bh_out = state.get("batch_head.3.bias")
        if bh_out is not None:
            n_fhs_batches = int(bh_out.shape[0])

    try:
        model = MiniVAEDANN(
            proj_dim=int(cfg["proj_dim"]),
            hidden=hidden,
            latent_dim=int(cfg["latent_dim"]),
            dropout=float(cfg["dropout"]),
            n_fhs_batches=n_fhs_batches,
        )
    except TypeError:
        model = MiniVAEDANN(
            proj_dim=int(cfg["proj_dim"]),
            hidden=hidden,
            latent_dim=int(cfg["latent_dim"]),
            dropout=float(cfg["dropout"]),
        )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def attach_frozen_dann_meth_stem(
    fusion: "CpGSNPFusionSurv",
    *,
    meth_input_dim: int,
    snp_input_dim: int,
    meth_branch_dim: int,
    dropout: float,
    dann_preprocess_npz: str | Path,
    dann_encoder_ckpt: str | Path,
    dann_meth_as_mvalues: bool,
) -> None:
    znp = np.load(dann_preprocess_npz, allow_pickle=False)
    d_expect = int(znp["d_in"])
    if meth_input_dim + snp_input_dim != d_expect:
        raise ValueError(
            f"DANN preprocess d_in={d_expect} but fusion has meth_dim+snp_dim="
            f"{meth_input_dim}+{snp_input_dim}. Use the same aligned NPZs and "
            f"--max-cpg/--max-snp as mini_vae_dann_pipeline.py."
        )
    if "meth_as_mvalues" in znp.files:
        saved_mval = bool(int(znp["meth_as_mvalues"][0]))
        if saved_mval != bool(dann_meth_as_mvalues):
            raise ValueError(
                f"NPZ meth_as_mvalues={saved_mval} but fusion dann_meth_as_mvalues="
                f"{dann_meth_as_mvalues}. Match training (M-value vs beta)."
            )

    fusion.dann_inv_pre = InvariantPreprocessor(dann_preprocess_npz)
    mini, _cfg = load_mini_dann_for_fusion(dann_encoder_ckpt, map_location="cpu")
    fusion.dann_latent_enc = DannLatentEncoder(mini)
    latent_d = int(mini.fc_mu.out_features)

    fusion.meth_from_dann = nn.Sequential(
        nn.Linear(latent_d, meth_branch_dim),
        nn.BatchNorm1d(meth_branch_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(meth_branch_dim, meth_branch_dim),
        nn.BatchNorm1d(meth_branch_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )
    fusion.dann_meth_as_mvalues = bool(dann_meth_as_mvalues)
    fusion._use_dann_stem = True
    fusion.raw_meth_input_dim = int(meth_input_dim)


def freeze_dann_stem_parameters(fusion: "CpGSNPFusionSurv") -> None:
    if not getattr(fusion, "_use_dann_stem", False):
        return
    for mod in (fusion.dann_inv_pre, fusion.dann_latent_enc):
        for p in mod.parameters():
            p.requires_grad = False


def meth_branch_trainable(fusion: "CpGSNPFusionSurv") -> nn.Module:
    """Module whose parameters Stage-2 ``meth_freeze_epochs`` should toggle."""
    if getattr(fusion, "_use_dann_stem", False):
        return fusion.meth_from_dann
    return fusion.meth_project
