// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub(all) struct TensorData {
  dims : FixedArray[Int]
  data : FixedArray[Float]
} derive(Debug)

///|
pub(open) trait Tensor {
  dims(Self) -> FixedArray[Int]
  zeros(dims : FixedArray[Int]) -> Self
  zeros_like(Self) -> Self
  from_host(data : FixedArray[Float], shape : Array[Int]) -> Self
  square(Self) -> Self
  sqrt(Self) -> Self
  mean(Self) -> Self
  scale(Self, Float) -> Self
  mul_elem(Self, Self) -> Self
  div_elem(Self, Self) -> Self
  scalar(Float) -> Self
  size(Self) -> Int
  reduce_sum_to(Self, target_dims : FixedArray[Int]) -> Self
  value(Self) -> TensorData
  view(Self, new_shape : FixedArray[Int]) -> Self
  add(Self, Self) -> Self
  add_into(Self, Self) -> Unit
  sub(Self, Self) -> Self
  /// Broadcast a scalar tensor to the given shape (fills target shape with the scalar value).
  broadcast_to(Self, target_shape : FixedArray[Int]) -> Self
}

///|
pub(open) trait BlasTensor {
  matmul(Self, Self) -> Self
  transpose(Self) -> Self
}

///|
pub(open) trait ImageTensor {
  // ── Forward ──
  conv2d(Self, weight : Self, bias : Self, stride : Int, padding : Int) -> Self
  relu(Self) -> Self
  maxpool2d(Self, kernel_size : Int, stride : Int) -> Self
  adaptive_avg_pool2d(Self, output_size : Int) -> Self
  batchnorm_training(
    Self,
    gamma : Self,
    beta : Self,
    running_mean : Self,
    running_var : Self,
    momentum : Float,
    eps : Float,
  ) -> (Self, Self, Self)
  batchnorm_inference(
    Self,
    gamma : Self,
    beta : Self,
    running_mean : Self,
    running_var : Self,
    eps : Float,
  ) -> Self
  softmax_cross_entropy(Self, targets : Self) -> Self
  /// Cross-entropy loss with integer class labels (stored as float tensor of shape [batch]).
  /// Equivalent to softmax_cross_entropy but avoids one-hot encoding.
  /// labels: tensor of shape [batch], each element is an integer class index.
  /// num_classes: number of classes (C), used by the kernel for logits shape [batch, C].
  cross_entropy_with_labels(Self, labels : Self, num_classes : Int) -> Self
}

///|
pub(open) trait ImageBackwardOps {
  // ── Backward ──
  /// d(conv2d)/d(input)
  conv2d_backward_data(
    grad_output : Self,
    weight : Self,
    input : Self,
    stride : Int,
    padding : Int,
  ) -> Self

  /// d(conv2d)/d(weight)
  conv2d_backward_weight(
    grad_output : Self,
    input : Self,
    weight : Self,
    stride : Int,
    padding : Int,
  ) -> Self

  /// d(conv2d)/d(bias) — reduce_sum over spatial dims
  conv2d_backward_bias(grad_output : Self) -> Self

  /// d(relu)/d(input)
  relu_backward(grad_output : Self, input : Self) -> Self

  /// d(batchnorm)/d(input), d(gamma), d(beta) — computed simultaneously
  batchnorm_backward(
    grad_output : Self,
    input : Self,
    gamma : Self,
    save_mean : Self,
    save_inv_var : Self,
    eps : Float,
  ) -> (Self, Self, Self)

  /// d(maxpool2d)/d(input)
  maxpool2d_backward(
    grad_output : Self,
    input : Self,
    kernel_size : Int,
    stride : Int,
  ) -> Self

  /// d(adaptive_avg_pool2d)/d(input)
  adaptive_avg_pool2d_backward(grad_output : Self, input : Self) -> Self

  /// d(softmax_cross_entropy)/d(logits) = (softmax(logits) - targets) / batch
  /// The result should be scaled by grad_output externally if needed.
  softmax_ce_backward(logits : Self, targets : Self) -> Self
  /// Backward for cross_entropy_with_labels.
  /// d(logits[b][j]) = (softmax[j] - indicator(j == class_idx)) / batch
  softmax_ce_backward_labels(logits : Self, labels : Self, num_classes : Int) -> Self
}
