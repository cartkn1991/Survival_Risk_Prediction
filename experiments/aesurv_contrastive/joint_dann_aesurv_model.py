"""End-to-end MiniVAEDANN + AESurvHeadAuxContrastive (raw omics -> mu -> survival)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from dann_aesurv_bridge import InvariantPreprocessor, load_mini_dann_for_fusion
from experiments.aesurv_contrastive.aesurv_head_aux_contrastive import AESurvHeadAuxContrastive
from mini_vae_dann_pipeline import MiniVAEDANN


def apply_modality_torch(
    x_meth: torch.Tensor,
    x_snp: torch.Tensor,
    mode: str,
    mean_full: torch.Tensor,
    n_cpg: int,
    n_snp: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of ``_apply_dann_input_modality`` (no logging)."""
    if mode == "both":
        return x_meth, x_snp
    if mode == "cpgs_only":
        smean = mean_full[n_cpg:].unsqueeze(0).expand(x_meth.size(0), -1)
        return x_meth, smean
    if mode == "snps_only":
        mmean = mean_full[:n_cpg].unsqueeze(0).expand(x_meth.size(0), -1)
        return mmean, x_snp
    raise ValueError(f"unknown modality {mode!r}")


class JointDannAesurvContrastive(nn.Module):
    """Frozen JL/scaler; trainable full DANN + contrastive AESurv head."""

    def __init__(
        self,
        preproc: InvariantPreprocessor,
        dann: MiniVAEDANN,
        head: AESurvHeadAuxContrastive,
        *,
        n_cpg: int,
        n_snp: int,
        meth_as_mvalues: bool,
    ) -> None:
        super().__init__()
        self.preproc = preproc
        self.dann = dann
        self.head = head
        self.n_cpg = int(n_cpg)
        self.n_snp = int(n_snp)
        self.meth_as_mvalues = bool(meth_as_mvalues)
        for p in self.preproc.parameters():
            p.requires_grad = False
        self.register_buffer("_mean_full", preproc.mean_.detach().clone(), persistent=False)

    @classmethod
    def from_checkpoints(
        cls,
        dann_ckpt: str,
        preprocess_npz: str,
        head: AESurvHeadAuxContrastive,
        *,
        n_cpg: int,
        n_snp: int,
        meth_as_mvalues: bool,
        map_location: torch.device | str = "cpu",
    ) -> Tuple["JointDannAesurvContrastive", dict]:
        preproc = InvariantPreprocessor(preprocess_npz)
        mini, cfg = load_mini_dann_for_fusion(dann_ckpt, map_location=map_location)
        mini.train()
        joint = cls(
            preproc, mini, head,
            n_cpg=n_cpg, n_snp=n_snp, meth_as_mvalues=meth_as_mvalues,
        )
        return joint, cfg

    def set_grl_lambda(self, lam: float) -> None:
        self.dann.grl.set_lambda(float(lam))

    def encode_raw(
        self,
        x_meth: torch.Tensor,
        x_snp: torch.Tensor,
        modality: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns mu, logvar, x_proj."""
        xm, xs = apply_modality_torch(
            x_meth, x_snp, modality, self._mean_full, self.n_cpg, self.n_snp,
        )
        x_proj = self.preproc(xm, xs, use_m_values=self.meth_as_mvalues)
        mu, logvar = self.dann.encode(x_proj)
        return mu, logvar, x_proj

    def forward_dann_on_proj(self, x_proj: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.dann(x_proj)

    def forward_head(
        self,
        mu_both: torch.Tensor,
        mu_cpg: torch.Tensor,
        mu_snp: torch.Tensor,
        *,
        sample_z: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        return self.head(mu_both, sample_z=sample_z, x_cpg=mu_cpg, x_snp=mu_snp)

    @torch.no_grad()
    def predict_log_h(
        self,
        x_meth: torch.Tensor,
        x_snp: torch.Tensor,
        *,
        input_modality: str = "both",
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        mu, _, _ = self.encode_raw(x_meth, x_snp, input_modality)
        _, _, log_h, _, _, _, _, _, _ = self.forward_head(mu, mu, mu, sample_z=False)
        if was_training:
            self.train()
        return log_h
