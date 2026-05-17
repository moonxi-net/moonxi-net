// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn gpu_softmax_cross_entropy(
  logits : GpuTensor,
  targets : GpuTensor,
  _cublas : @cuda.CublasHandle,
) -> GpuTensor {
  let loss_output = GpuTensor::alloc_st(
    @ts.ShapeTensor::softmax_cross_entropy(logits.shape, targets.shape),
  )
  let softmax_output = GpuTensor::alloc_st(logits.shape)
  let batch = logits.dim(0)
  let num_classes = logits.dim(1)
  check_cuda(
    @cuda.softmax_ce_forward(
      logits.data,
      targets.data,
      loss_output.data,
      softmax_output.data,
      batch,
      num_classes,
    ),
    "softmax_ce_forward",
  )
  loss_output
}

///|
/// Cross-entropy with integer class labels (no one-hot encoding needed).
pub fn gpu_cross_entropy_with_labels(
  logits : GpuTensor,
  labels : GpuTensor,
  num_classes : Int,
) -> GpuTensor {
  let loss_output = GpuTensor::alloc_st(
    @ts.ShapeTensor::cross_entropy_with_labels(
      logits.shape,
      labels.shape,
      num_classes,
    ),
  )
  let batch = logits.dim(0)
  check_cuda(
    @cuda.softmax_ce_forward_labels(
      logits.data,
      labels.data,
      loss_output.data,
      batch,
      num_classes,
    ),
    "softmax_ce_forward_labels",
  )
  loss_output
}

///|
/// Backward pass for cross-entropy with integer class labels.
pub fn gpu_softmax_ce_backward_labels(
  logits : GpuTensor,
  labels : GpuTensor,
  num_classes : Int,
) -> GpuTensor {
  let grad_logits = GpuTensor::alloc_st(logits.shape)
  let batch = logits.dim(0)
  check_cuda(
    @cuda.softmax_ce_backward_labels(
      logits.data,
      labels.data,
      grad_logits.data,
      batch,
      num_classes,
    ),
    "softmax_ce_backward_labels",
  )
  grad_logits
}
