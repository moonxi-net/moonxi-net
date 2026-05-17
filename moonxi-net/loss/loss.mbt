// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// A loss function wrapper: (prediction, target) -> scalar loss.
///
/// Used by `Experiment::new` to decouple loss computation from model definition.
pub(all) struct LossFn[T] {
  loss : (T, T) -> T
}

// ── Tensor-level loss functions ──

///|
/// Mean squared error loss.
///
/// Computes `mean((y_hat - y)^2)`.
///
/// # Example
/// ```mbt nocheck
/// let y_hat = @nparray.NpArray::from1d([2.0, 3.0, 4.0])
/// let y = @nparray.NpArray::from1d([1.0, 2.0, 3.0])
/// let result = @loss.mse(y_hat, y)
/// inspect(result.data[0], content="1.")
/// ```
pub fn[T : @tensor.Tensor] mse(y_hat : T, y : T) -> T {
  y_hat.sub(y).square().mean()
}

///|
/// Cross-entropy loss for classification (one-hot targets).
///
/// Applies softmax to logits, then computes cross-entropy against one-hot encoded targets.
/// Requires T to implement `ImageTensor` (which provides `softmax_cross_entropy`).
pub fn[T : @tensor.ImageTensor] cross_entropy(logits : T, targets : T) -> T {
  logits.softmax_cross_entropy(targets)
}

///|
/// Cross-entropy loss with integer class labels.
///
/// Converts integer labels to one-hot encoding internally, then computes cross-entropy.
///
/// - `labels`: tensor of shape `[batch]`, each element is an integer class index stored as float.
/// - `num_classes`: number of classes.
pub fn[T : @tensor.ImageTensor] cross_entropy_with_labels(
  logits : T,
  labels : T,
  num_classes : Int,
) -> T {
  logits.cross_entropy_with_labels(labels, num_classes)
}

// ── LossFn constructors ──

///|
/// MSE loss as a `LossFn`: mean((y_hat - y)^2)
pub fn[T : @tensor.Tensor] mse_loss() -> LossFn[T] {
  { loss: fn(y_hat : T, y : T) -> T { y_hat.sub(y).square().mean() } }
}

///|
/// Cross-entropy loss with integer class labels (avoids one-hot encoding).
pub fn[T : @tensor.ImageTensor] cross_entropy_loss(
  num_classes : Int,
) -> LossFn[T] {
  {
    loss: fn(logits : T, labels : T) -> T {
      logits.cross_entropy_with_labels(labels, num_classes)
    },
  }
}

///|
/// Cross-entropy loss with one-hot targets.
pub fn[T : @tensor.ImageTensor] cross_entropy_onehot_loss() -> LossFn[T] {
  {
    loss: fn(logits : T, targets : T) -> T {
      logits.softmax_cross_entropy(targets)
    },
  }
}
