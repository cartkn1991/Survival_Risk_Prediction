"""Load fused AESURV weights from PyTorch checkpoint into Keras model (strict=False)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _pt_state(path: Path) -> Dict[str, Any]:
  import torch

  raw = torch.load(path, map_location="cpu", weights_only=False)
  return raw.get("state_dict", raw)


def _dense(w: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
  return [w.T.astype(np.float32), b.astype(np.float32)]


def _ln(w: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
  return [w.astype(np.float32), b.astype(np.float32)]


def load_fused_head_from_pytorch(keras_model, pt_path: Path, log=None) -> Tuple[int, int]:
  """Copy encoder/decoder/cox/cohort/age/cell weights; skip new contrastive layers."""
  st = _pt_state(pt_path)
  loaded, skipped = 0, 0

  def try_set(layer, key_w: str, key_b: str | None = None, *, ln: bool = False):
    nonlocal loaded, skipped
    if key_w not in st:
      skipped += 1
      return
    if key_b is not None and key_b not in st:
      skipped += 1
      return
    w = st[key_w].detach().cpu().numpy()
    if key_b is None:
      layer.set_weights(_dense(w, np.zeros(layer.units, dtype=np.float32)))
    elif ln:
      b = st[key_b].detach().cpu().numpy()
      layer.set_weights(_ln(w, b))
    else:
      b = st[key_b].detach().cpu().numpy()
      layer.set_weights(_dense(w, b))
    loaded += 1

  m = keras_model
  # encoder: 0=Dense,1=LN,3=Dense,4=LN
  try_set(m.enc_dense1, "encoder.0.weight", "encoder.0.bias")
  try_set(m.enc_ln1, "encoder.1.weight", "encoder.1.bias", ln=True)
  try_set(m.enc_dense2, "encoder.3.weight", "encoder.3.bias")
  try_set(m.enc_ln2, "encoder.4.weight", "encoder.4.bias", ln=True)
  try_set(m.mu_head, "mu_head.weight", "mu_head.bias")
  try_set(m.logvar_head, "logvar_head.weight", "logvar_head.bias")
  try_set(m.dec_dense1, "decoder.0.weight", "decoder.0.bias")
  try_set(m.dec_ln1, "decoder.1.weight", "decoder.1.bias", ln=True)
  try_set(m.dec_dense2, "decoder.3.weight", "decoder.3.bias")
  try_set(m.dec_ln2, "decoder.4.weight", "decoder.4.bias", ln=True)
  try_set(m.dec_out, "decoder.6.weight", "decoder.6.bias")
  try_set(m.cox_head, "cox_head.weight", "cox_head.bias")
  try_set(m.age_head, "age_head.weight", "age_head.bias")
  try_set(m.cell_head, "cell_head.weight", "cell_head.bias")
  try_set(m.cohort_dense1, "cohort_head.0.weight", "cohort_head.0.bias")
  try_set(m.cohort_out, "cohort_head.3.weight", "cohort_head.3.bias")

  if log:
    log.info("PyTorch init %s: loaded %d layer groups, skipped %d", pt_path, loaded, skipped)
  return loaded, skipped
