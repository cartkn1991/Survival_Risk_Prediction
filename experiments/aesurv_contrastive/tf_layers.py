"""TensorFlow helpers: gradient reversal, Cox loss, InfoNCE."""
from __future__ import annotations

import tensorflow as tf


class GradientReversal(tf.keras.layers.Layer):
  """Forward identity; backward scales gradients by -lambda (cohort adversary)."""

  def __init__(self, lam: float = 1.0, **kwargs):
    super().__init__(**kwargs)
    self.lam = float(lam)

  def call(self, x):
    @tf.custom_gradient
    def _grl(x_in):
      def grad(dy):
        return -self.lam * dy

      return x_in, grad

    return _grl(x)


@tf.function
def cox_ph_loss_tf(log_h: tf.Tensor, time: tf.Tensor, event: tf.Tensor) -> tf.Tensor:
  event = tf.cast(event, tf.float32)
  if tf.reduce_sum(event) <= 0:
    return tf.constant(0.0, dtype=tf.float32)
  order = tf.argsort(time, direction="DESCENDING")
  log_h = tf.gather(log_h, order)
  event = tf.gather(event, order)
  log_cum = tf.math.log(tf.math.cumsum(tf.exp(log_h)))
  per = -(log_h - log_cum) * event
  return tf.reduce_sum(per) / tf.maximum(tf.reduce_sum(event), 1.0)


@tf.function
def info_nce_tf(proj_a: tf.Tensor, proj_b: tf.Tensor, tau: float) -> tf.Tensor:
  tau = tf.maximum(tf.cast(tau, tf.float32), 1e-6)
  a = tf.math.l2_normalize(proj_a, axis=-1)
  b = tf.math.l2_normalize(proj_b, axis=-1)
  logits = tf.matmul(a, b, transpose_b=True) / tau
  n = tf.shape(logits)[0]
  labels = tf.range(n)
  l1 = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  )
  l2 = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(logits))
  )
  return 0.5 * (l1 + l2)
