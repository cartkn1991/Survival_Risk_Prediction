"""Keras AESURV aux + contrastive head (fused path + CpG/SNP InfoNCE)."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from experiments.aesurv_contrastive.tf_layers import GradientReversal, cox_ph_loss_tf, info_nce_tf


def _mlp_block(x, units: int, dropout: float, name: str):
  x = tf.keras.layers.Dense(units, name=f"{name}_dense")(x)
  x = tf.keras.layers.LayerNormalization(name=f"{name}_ln")(x)
  x = tf.keras.layers.Activation("gelu", name=f"{name}_gelu")(x)
  x = tf.keras.layers.Dropout(dropout, name=f"{name}_drop")(x)
  return x


class AESurvContrastiveTF(tf.keras.Model):
  """Mirrors PyTorch AESurvHeadAuxContrastive; fused mu drives survival heads."""

  def __init__(
      self,
      in_dim: int = 128,
      enc_hidden: Tuple[int, ...] = (64, 32),
      dec_hidden: Tuple[int, ...] = (32, 64),
      z_dim: int = 8,
      cohort_hidden: int = 8,
      n_cells: int = 6,
      dropout: float = 0.40,
      contrast_proj_dim: int = 64,
      contrast_hidden: int = 32,
      contrast_tau: float = 0.07,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.in_dim = int(in_dim)
    self.z_dim = int(z_dim)
    self.n_cells = int(n_cells)
    self.dropout_rate = float(dropout)
    self.contrast_tau = float(contrast_tau)
    h1, h2 = int(enc_hidden[0]), int(enc_hidden[1])
    d1, d2 = int(dec_hidden[0]), int(dec_hidden[1])

    self.enc_dense1 = tf.keras.layers.Dense(h1, name="enc_dense1")
    self.enc_ln1 = tf.keras.layers.LayerNormalization(name="enc_ln1")
    self.enc_drop1 = tf.keras.layers.Dropout(dropout)
    self.enc_dense2 = tf.keras.layers.Dense(h2, name="enc_dense2")
    self.enc_ln2 = tf.keras.layers.LayerNormalization(name="enc_ln2")
    self.enc_drop2 = tf.keras.layers.Dropout(dropout)
    self.mu_head = tf.keras.layers.Dense(z_dim, name="mu_head")
    self.logvar_head = tf.keras.layers.Dense(z_dim, name="logvar_head")

    self.dec_dense1 = tf.keras.layers.Dense(d1, name="dec_dense1")
    self.dec_ln1 = tf.keras.layers.LayerNormalization(name="dec_ln1")
    self.dec_drop1 = tf.keras.layers.Dropout(dropout)
    self.dec_dense2 = tf.keras.layers.Dense(d2, name="dec_dense2")
    self.dec_ln2 = tf.keras.layers.LayerNormalization(name="dec_ln2")
    self.dec_drop2 = tf.keras.layers.Dropout(dropout)
    self.dec_out = tf.keras.layers.Dense(in_dim, name="dec_out")

    self.cox_head = tf.keras.layers.Dense(1, name="cox_head")
    self.age_head = tf.keras.layers.Dense(1, name="age_head")
    self.cell_head = tf.keras.layers.Dense(n_cells, name="cell_head")
    self.cohort_dense1 = tf.keras.layers.Dense(cohort_hidden, name="cohort_dense1")
    self.cohort_leaky = tf.keras.layers.LeakyReLU(0.01)
    self.cohort_drop = tf.keras.layers.Dropout(dropout)
    self.cohort_out = tf.keras.layers.Dense(2, name="cohort_out")

    self.mod_dense1 = tf.keras.layers.Dense(h1, name="mod_dense1")
    self.mod_ln1 = tf.keras.layers.LayerNormalization(name="mod_ln1")
    self.mod_drop1 = tf.keras.layers.Dropout(dropout)
    self.mod_dense2 = tf.keras.layers.Dense(h2, name="mod_dense2")
    self.mod_ln2 = tf.keras.layers.LayerNormalization(name="mod_ln2")
    self.mod_drop2 = tf.keras.layers.Dropout(dropout)
    self.mod_mu = tf.keras.layers.Dense(z_dim, name="mod_mu")

    self.proj_cpg_h = tf.keras.layers.Dense(contrast_hidden, activation="gelu", name="proj_cpg_h")
    self.proj_cpg_drop = tf.keras.layers.Dropout(dropout)
    self.proj_cpg_out = tf.keras.layers.Dense(contrast_proj_dim, name="proj_cpg_out")
    self.proj_snp_h = tf.keras.layers.Dense(contrast_hidden, activation="gelu", name="proj_snp_h")
    self.proj_snp_drop = tf.keras.layers.Dropout(dropout)
    self.proj_snp_out = tf.keras.layers.Dense(contrast_proj_dim, name="proj_snp_out")

    self.grl = GradientReversal(lam=1.0)

  def _encode_fused(self, x, training: bool):
    h = self.enc_dense1(x)
    h = self.enc_ln1(h)
    h = tf.nn.gelu(h)
    h = self.enc_drop1(h, training=training)
    h = self.enc_dense2(h)
    h = self.enc_ln2(h)
    h = tf.nn.gelu(h)
    h = self.enc_drop2(h, training=training)
    mu = self.mu_head(h)
    logvar = tf.clip_by_value(self.logvar_head(h), -8.0, 8.0)
    if training:
      std = tf.exp(0.5 * logvar)
      eps = tf.random.normal(tf.shape(std))
      z = mu + eps * std
    else:
      z = mu
    return z, mu, logvar

  def _decode(self, z, training: bool):
    h = self.dec_dense1(z)
    h = self.dec_ln1(h)
    h = tf.nn.gelu(h)
    h = self.dec_drop1(h, training=training)
    h = self.dec_dense2(h)
    h = self.dec_ln2(h)
    h = tf.nn.gelu(h)
    h = self.dec_drop2(h, training=training)
    return self.dec_out(h)

  def _encode_modality(self, x, training: bool):
    h = self.mod_dense1(x)
    h = self.mod_ln1(h)
    h = tf.nn.gelu(h)
    h = self.mod_drop1(h, training=training)
    h = self.mod_dense2(h)
    h = self.mod_ln2(h)
    h = tf.nn.gelu(h)
    h = self.mod_drop2(h, training=training)
    return self.mod_mu(h)

  def _project_cpg(self, z, training: bool):
    h = self.proj_cpg_h(z)
    h = self.proj_cpg_drop(h, training=training)
    return self.proj_cpg_out(h)

  def _project_snp(self, z, training: bool):
    h = self.proj_snp_h(z)
    h = self.proj_snp_drop(h, training=training)
    return self.proj_snp_out(h)

  def fused_forward(self, mu, training: bool):
    z, mu_h, logvar = self._encode_fused(mu, training=training)
    x_rec = self._decode(z, training=training)
    log_h = tf.squeeze(self.cox_head(z), axis=-1)
    age_pred = tf.squeeze(self.age_head(z), axis=-1)
    cell_pred = self.cell_head(z)
    return x_rec, z, log_h, mu_h, logvar, age_pred, cell_pred

  def contrastive_forward(self, mu_cpg, mu_snp, training: bool):
    z_cpg = self._encode_modality(mu_cpg, training=training)
    z_snp = self._encode_modality(mu_snp, training=training)
    p_cpg = self._project_cpg(z_cpg, training=training)
    p_snp = self._project_snp(z_snp, training=training)
    return p_cpg, p_snp

  def cohort_logits(self, z, lam: float):
    self.grl.lam = float(lam)
    h = self.grl(z)
    h = self.cohort_dense1(h)
    h = self.cohort_leaky(h)
    h = self.cohort_drop(h, training=self.training)
    return self.cohort_out(h)

  def call(self, inputs, training=False):
    """Not used for training loop; explicit methods above."""
    raise NotImplementedError("Use fused_forward / contrastive_forward in train_step.")
