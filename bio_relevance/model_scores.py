"""Pooled log-hazard and predicted age from frozen bundle + JL projections."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from dann_aesurv_bridge import DannLatentEncoder, load_mini_dann_for_fusion
from train_aesurv_dann_latent_aux import AESurvHeadAux


def _torch_load_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class _FrozenAgeRisk(torch.nn.Module):
    def __init__(self, encoder, head, age_mu: float, age_sd: float):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.age_mu = float(age_mu)
        self.age_sd = float(age_sd)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x_proj: torch.Tensor):
        mu_d = self.encoder(x_proj)
        _, _z, log_h, _mu, _logvar, age_z, _cell = self.head(mu_d, sample_z=False)
        age_years = age_z * self.age_sd + self.age_mu
        return log_h, age_years


@torch.no_grad()
def pooled_predictions(
    bundle_dir: Path,
    proj_fhs_npz: Path,
    proj_whi_npz: Path,
    device: torch.device,
    batch_size: int = 128,
    joint_analysis_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Returns ``(log_h, age_pred, n_fhs, n_whi)`` in FHS-then-WHI order."""
    if joint_analysis_dir is not None:
        from bio_relevance.joint_model_scores import joint_pooled_predictions
        return joint_pooled_predictions(
            Path(joint_analysis_dir), device, batch_size=batch_size,
        )
    bundle = Path(bundle_dir)
    mini, _ = load_mini_dann_for_fusion(bundle / "dann_encoder.pt", map_location=device)
    encoder = DannLatentEncoder(mini.to(device)).eval()
    head_ck = _torch_load_compat(bundle / "aesurv_head.pt", map_location=device)
    cfg = head_ck["config"]
    head = AESurvHeadAux(
        in_dim=int(cfg["in_dim"]), enc_hidden=tuple(cfg["enc_hidden"]),
        dec_hidden=tuple(cfg["dec_hidden"]), z_dim=int(cfg["z_dim"]),
        cohort_hidden=int(cfg["cohort_hidden"]), dropout=float(cfg["dropout"]),
        n_cells=int(cfg["n_cells"]),
    ).to(device).eval()
    head.load_state_dict(head_ck["state_dict"])
    age_mu = float(cfg["aux_norm"]["age_mu"])
    age_sd = float(cfg["aux_norm"]["age_sd"])
    model = _FrozenAgeRisk(encoder, head, age_mu, age_sd).to(device)

    xf = np.load(proj_fhs_npz, allow_pickle=False)["x"].astype(np.float32)
    xw = np.load(proj_whi_npz, allow_pickle=False)["x"].astype(np.float32)
    x = np.concatenate([xf, xw], axis=0)
    n_fhs, n_whi = xf.shape[0], xw.shape[0]
    out_h = np.empty((x.shape[0],), dtype=np.float32)
    out_a = np.empty((x.shape[0],), dtype=np.float32)
    for s in range(0, x.shape[0], batch_size):
        e = min(x.shape[0], s + batch_size)
        t = torch.from_numpy(x[s:e]).to(device)
        lh, ay = model(t)
        out_h[s:e] = lh.squeeze(-1).cpu().numpy()
        out_a[s:e] = ay.squeeze(-1).cpu().numpy()
    return out_h, out_a, n_fhs, n_whi
