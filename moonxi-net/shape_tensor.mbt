// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// A shape-only tensor that computes output shapes without actual data.
/// Aborts when shape constraints are violated.
pub(all) struct ShapeTensor {
  shape : FixedArray[Int]
} derive(Debug)

///|
fn ShapeTensor::numel(self : ShapeTensor) -> Int {
  self.shape.iter().fold(init=1, fn(acc, d) { acc * d })
}

///|
/// Compute total number of elements for a shape.
fn shape_numel(shape : FixedArray[Int]) -> Int {
  shape.iter().fold(init=1, fn(acc, d) { acc * d })
}

///|
/// Broadcast two shapes. Aborts if dimensions are incompatible.
fn shape_broadcast(a : FixedArray[Int], b : FixedArray[Int]) -> FixedArray[Int] {
  let da = a.length()
  let db = b.length()
  let d = if da > db { da } else { db }
  FixedArray::makei(d, fn(i) {
    let ai = if i >= d - da { a[i - (d - da)] } else { 1 }
    let bi = if i >= d - db { b[i - (d - db)] } else { 1 }
    if ai == bi {
      ai
    } else if ai == 1 {
      bi
    } else if bi == 1 {
      ai
    } else {
      let a_debug = @debug.to_string(a)
      let b_debug = @debug.to_string(b)
      abort(
        "ShapeTensor: broadcast error at dim \{i}: \{ai} vs \{bi}, shapes \{a_debug} vs \{b_debug}",
      )
    }
  })
}

///|
pub fn ShapeTensor::from_array(shape : Array[Int]) -> ShapeTensor {
  { shape: FixedArray::from_iter(shape.iter()) }
}

///|
pub fn ShapeTensor::to_array(self : ShapeTensor) -> Array[Int] {
  self.shape.iter().collect()
}

///|
pub fn ShapeTensor::dim(self : ShapeTensor, i : Int) -> Int {
  self.shape[i]
}

///|
pub fn ShapeTensor::ndim(self : ShapeTensor) -> Int {
  self.shape.length()
}

// ═══════════════════════════════════════════════════════════════
// Tensor trait implementation
// ═══════════════════════════════════════════════════════════════

///|
pub impl Tensor for ShapeTensor with dims(st : ShapeTensor) -> FixedArray[Int] {
  st.shape
}

///|
pub impl Tensor for ShapeTensor with zeros(dims : FixedArray[Int]) -> ShapeTensor {
  { shape: dims }
}

///|
pub impl Tensor for ShapeTensor with zeros_like(st : ShapeTensor) -> ShapeTensor {
  { shape: FixedArray::makei(st.shape.length(), fn(i) { st.shape[i] }) }
}

///|
pub impl Tensor for ShapeTensor with from_host(
  _data : FixedArray[Float],
  shape : Array[Int],
) -> ShapeTensor {
  { shape: FixedArray::from_iter(shape.iter()) }
}

///|
pub impl Tensor for ShapeTensor with square(st : ShapeTensor) -> ShapeTensor {
  st
}

///|
pub impl Tensor for ShapeTensor with mean(_st : ShapeTensor) -> ShapeTensor {
  { shape: FixedArray::make(1, 1) }
}

///|
pub impl Tensor for ShapeTensor with scale(st : ShapeTensor, _s : Float) -> ShapeTensor {
  st
}

///|
pub impl Tensor for ShapeTensor with mul_elem(a : ShapeTensor, b : ShapeTensor) -> ShapeTensor {
  { shape: shape_broadcast(a.shape, b.shape) }
}

///|
pub impl Tensor for ShapeTensor with scalar(_v : Float) -> ShapeTensor {
  { shape: FixedArray::make(1, 1) }
}

///|
pub impl Tensor for ShapeTensor with size(st : ShapeTensor) -> Int {
  st.numel()
}

///|
pub impl Tensor for ShapeTensor with reduce_sum_to(
  _st : ShapeTensor,
  target_dims : FixedArray[Int],
) -> ShapeTensor {
  { shape: target_dims }
}

///|
pub impl Tensor for ShapeTensor with value(st : ShapeTensor) -> TensorData {
  let n = st.numel()
  {
    dims: FixedArray::makei(st.shape.length(), fn(i) { st.shape[i] }),
    data: FixedArray::make(n, 0.0),
  }
}

///|
pub impl Tensor for ShapeTensor with view(
  st : ShapeTensor,
  new_shape : FixedArray[Int],
) -> ShapeTensor {
  let old_numel = st.numel()
  let new_numel = shape_numel(new_shape)
  if old_numel != new_numel {
    abort(
      "ShapeTensor: view error: element count mismatch \{old_numel} vs \{new_numel}",
    )
  }
  { shape: new_shape }
}

///|
pub impl Tensor for ShapeTensor with add(a : ShapeTensor, b : ShapeTensor) -> ShapeTensor {
  { shape: shape_broadcast(a.shape, b.shape) }
}

///|
pub impl Tensor for ShapeTensor with add_into(a : ShapeTensor, b : ShapeTensor) -> Unit {
  if a.shape.length() != b.shape.length() {
    abort(
      "ShapeTensor: add_into error: ndim mismatch \{a.shape.length()} vs \{b.shape.length()}",
    )
  }
  for i in 0..<a.shape.length() {
    if a.shape[i] != b.shape[i] {
      abort(
        "ShapeTensor: add_into error: shape mismatch at dim \{i}: \{a.shape[i]} vs \{b.shape[i]}",
      )
    }
  }
}

///|
pub impl Tensor for ShapeTensor with sub(a : ShapeTensor, b : ShapeTensor) -> ShapeTensor {
  { shape: shape_broadcast(a.shape, b.shape) }
}

///|
pub impl Tensor for ShapeTensor with broadcast_to(
  _st : ShapeTensor,
  target_shape : FixedArray[Int],
) -> ShapeTensor {
  { shape: target_shape }
}

///|
pub impl Tensor for ShapeTensor with sqrt(st : ShapeTensor) -> ShapeTensor {
  st
}

///|
pub impl Tensor for ShapeTensor with div_elem(a : ShapeTensor, b : ShapeTensor) -> ShapeTensor {
  { shape: shape_broadcast(a.shape, b.shape) }
}

// ═══════════════════════════════════════════════════════════════
// BlasTensor trait implementation
// ═══════════════════════════════════════════════════════════════

///|
pub impl BlasTensor for ShapeTensor with matmul(
  a : ShapeTensor,
  b : ShapeTensor,
) -> ShapeTensor {
  if a.shape.length() != 2 {
    abort("ShapeTensor: matmul error: expected 2D, got \{a.shape.length()}D")
  }
  if b.shape.length() != 2 {
    abort("ShapeTensor: matmul error: expected 2D, got \{b.shape.length()}D")
  }
  if a.shape[1] != b.shape[0] {
    abort(
      "ShapeTensor: matmul error: inner dims mismatch \{a.shape[1]} vs \{b.shape[0]}",
    )
  }
  let n = a.shape[0]
  let p = b.shape[1]
  { shape: FixedArray::from_iter([n, p].iter()) }
}

///|
pub impl BlasTensor for ShapeTensor with transpose(st : ShapeTensor) -> ShapeTensor {
  if st.shape.length() != 2 {
    abort(
      "ShapeTensor: transpose error: expected 2D, got \{st.shape.length()}D",
    )
  }
  { shape: FixedArray::from_iter([st.shape[1], st.shape[0]].iter()) }
}

// ═══════════════════════════════════════════════════════════════
// ImageTensor trait implementation — Forward
// ═══════════════════════════════════════════════════════════════

///|
pub impl ImageTensor for ShapeTensor with conv2d(
  input : ShapeTensor,
  weight : ShapeTensor,
  bias : ShapeTensor,
  stride : Int,
  padding : Int,
) -> ShapeTensor {
  if input.shape.length() != 4 {
    abort(
      "ShapeTensor: conv2d error: input expected 4D [N,C,H,W], got \{input.shape.length()}D",
    )
  }
  if weight.shape.length() != 4 {
    abort(
      "ShapeTensor: conv2d error: weight expected 4D [C_out,C_in,KH,KW], got \{weight.shape.length()}D",
    )
  }
  if input.shape[1] != weight.shape[1] {
    abort(
      "ShapeTensor: conv2d error: C_in mismatch input\{input.shape[1]} vs weight\{weight.shape[1]}",
    )
  }
  if bias.shape.length() >= 1 && bias.shape[0] != weight.shape[0] {
    abort(
      "ShapeTensor: conv2d error: bias size \{bias.shape[0]} != C_out \{weight.shape[0]}",
    )
  }
  let n = input.shape[0]
  let c_out = weight.shape[0]
  let h = input.shape[2]
  let w = input.shape[3]
  let kh = weight.shape[2]
  let kw = weight.shape[3]
  let oh = (h + 2 * padding - kh) / stride + 1
  let ow = (w + 2 * padding - kw) / stride + 1
  { shape: FixedArray::from_iter([n, c_out, oh, ow].iter()) }
}

///|
pub impl ImageTensor for ShapeTensor with relu(st : ShapeTensor) -> ShapeTensor {
  st
}

///|
pub impl ImageTensor for ShapeTensor with maxpool2d(
  st : ShapeTensor,
  kernel_size : Int,
  stride : Int,
) -> ShapeTensor {
  if st.shape.length() != 4 {
    abort(
      "ShapeTensor: maxpool2d error: expected 4D [N,C,H,W], got \{st.shape.length()}D",
    )
  }
  let n = st.shape[0]
  let c = st.shape[1]
  let h = st.shape[2]
  let w = st.shape[3]
  let oh = (h - kernel_size) / stride + 1
  let ow = (w - kernel_size) / stride + 1
  { shape: FixedArray::from_iter([n, c, oh, ow].iter()) }
}

///|
pub impl ImageTensor for ShapeTensor with adaptive_avg_pool2d(
  st : ShapeTensor,
  output_size : Int,
) -> ShapeTensor {
  if st.shape.length() != 4 {
    abort(
      "ShapeTensor: adaptive_avg_pool2d error: expected 4D [N,C,H,W], got \{st.shape.length()}D",
    )
  }
  let n = st.shape[0]
  let c = st.shape[1]
  { shape: FixedArray::from_iter([n, c, output_size, output_size].iter()) }
}

///|
pub impl ImageTensor for ShapeTensor with batchnorm_training(
  input : ShapeTensor,
  _gamma : ShapeTensor,
  _beta : ShapeTensor,
  _running_mean : ShapeTensor,
  _running_var : ShapeTensor,
  _momentum : Float,
  _eps : Float,
) -> (ShapeTensor, ShapeTensor, ShapeTensor) {
  if input.shape.length() != 4 {
    abort(
      "ShapeTensor: batchnorm_training error: expected 4D [N,C,H,W], got \{input.shape.length()}D",
    )
  }
  let c = input.shape[1]
  let output = {
    shape: FixedArray::makei(input.shape.length(), fn(i) { input.shape[i] }),
  }
  let save_mean = { shape: FixedArray::make(1, c) }
  let save_inv_var = { shape: FixedArray::make(1, c) }
  (output, save_mean, save_inv_var)
}

///|
pub impl ImageTensor for ShapeTensor with batchnorm_inference(
  input : ShapeTensor,
  _gamma : ShapeTensor,
  _beta : ShapeTensor,
  _running_mean : ShapeTensor,
  _running_var : ShapeTensor,
  _eps : Float,
) -> ShapeTensor {
  if input.shape.length() != 4 {
    abort(
      "ShapeTensor: batchnorm_inference error: expected 4D [N,C,H,W], got \{input.shape.length()}D",
    )
  }
  { shape: FixedArray::makei(input.shape.length(), fn(i) { input.shape[i] }) }
}

///|
pub impl ImageTensor for ShapeTensor with softmax_cross_entropy(
  _logits : ShapeTensor,
  _targets : ShapeTensor,
) -> ShapeTensor {
  { shape: FixedArray::make(1, 1) }
}

///|
pub impl ImageTensor for ShapeTensor with cross_entropy_with_labels(
  _logits : ShapeTensor,
  _labels : ShapeTensor,
  _num_classes : Int,
) -> ShapeTensor {
  { shape: FixedArray::make(1, 1) }
}

// ═══════════════════════════════════════════════════════════════
// ImageBackwardOps trait implementation
// ═══════════════════════════════════════════════════════════════

///|
pub impl ImageBackwardOps for ShapeTensor with conv2d_backward_data(
  _grad_output : ShapeTensor,
  _weight : ShapeTensor,
  input : ShapeTensor,
  _stride : Int,
  _padding : Int,
) -> ShapeTensor {
  { shape: FixedArray::makei(input.shape.length(), fn(i) { input.shape[i] }) }
}

///|
pub impl ImageBackwardOps for ShapeTensor with conv2d_backward_weight(
  _grad_output : ShapeTensor,
  _input : ShapeTensor,
  weight : ShapeTensor,
  _stride : Int,
  _padding : Int,
) -> ShapeTensor {
  { shape: FixedArray::makei(weight.shape.length(), fn(i) { weight.shape[i] }) }
}

///|
pub impl ImageBackwardOps for ShapeTensor with conv2d_backward_bias(
  grad_output : ShapeTensor,
) -> ShapeTensor {
  if grad_output.shape.length() < 2 {
    abort(
      "ShapeTensor: conv2d_backward_bias error: expected >=2D, got \{grad_output.shape.length()}D",
    )
  }
  let c_out = grad_output.shape[1]
  { shape: FixedArray::make(1, c_out) }
}

///|
pub impl ImageBackwardOps for ShapeTensor with relu_backward(
  _grad_output : ShapeTensor,
  input : ShapeTensor,
) -> ShapeTensor {
  { shape: FixedArray::makei(input.shape.length(), fn(i) { input.shape[i] }) }
}

///|
pub impl ImageBackwardOps for ShapeTensor with batchnorm_backward(
  _grad_output : ShapeTensor,
  input : ShapeTensor,
  gamma : ShapeTensor,
  _save_mean : ShapeTensor,
  _save_inv_var : ShapeTensor,
  _eps : Float,
) -> (ShapeTensor, ShapeTensor, ShapeTensor) {
  let grad_input = {
    shape: FixedArray::makei(input.shape.length(), fn(i) { input.shape[i] }),
  }
  let grad_gamma = {
    shape: FixedArray::makei(gamma.shape.length(), fn(i) { gamma.shape[i] }),
  }
  let grad_beta = {
    shape: FixedArray::makei(gamma.shape.length(), fn(i) { gamma.shape[i] }),
  }
  (grad_input, grad_gamma, grad_beta)
}

///|
pub impl ImageBackwardOps for ShapeTensor with maxpool2d_backward(
  _grad_output : ShapeTensor,
  input : ShapeTensor,
  _kernel_size : Int,
  _stride : Int,
) -> ShapeTensor {
  { shape: FixedArray::makei(input.shape.length(), fn(i) { input.shape[i] }) }
}

///|
pub impl ImageBackwardOps for ShapeTensor with adaptive_avg_pool2d_backward(
  _grad_output : ShapeTensor,
  input : ShapeTensor,
) -> ShapeTensor {
  { shape: FixedArray::makei(input.shape.length(), fn(i) { input.shape[i] }) }
}

///|
pub impl ImageBackwardOps for ShapeTensor with softmax_ce_backward(
  logits : ShapeTensor,
  _targets : ShapeTensor,
) -> ShapeTensor {
  { shape: FixedArray::makei(logits.shape.length(), fn(i) { logits.shape[i] }) }
}

///|
pub impl ImageBackwardOps for ShapeTensor with softmax_ce_backward_labels(
  logits : ShapeTensor,
  _labels : ShapeTensor,
  _num_classes : Int,
) -> ShapeTensor {
  { shape: FixedArray::makei(logits.shape.length(), fn(i) { logits.shape[i] }) }
}

// ═══════════════════════════════════════════════════════════════
// Non-trait utilities for GPU ops
// ═══════════════════════════════════════════════════════════════

///|
/// Pool2d output shape with explicit padding (not in trait).
pub fn ShapeTensor::pool2d(
  self : ShapeTensor,
  kernel_size : Int,
  stride : Int,
  padding : Int,
) -> ShapeTensor {
  let n = self.shape[0]
  let c = self.shape[1]
  let h = self.shape[2]
  let w = self.shape[3]
  let oh = (h + 2 * padding - kernel_size) / stride + 1
  let ow = (w + 2 * padding - kernel_size) / stride + 1
  { shape: FixedArray::from_iter([n, c, oh, ow].iter()) }
}

///|
/// Linear layer output shape: [batch, out_features] from input [batch, in_f] and weight [out_f, in_f].
pub fn ShapeTensor::linear(
  input : ShapeTensor,
  weight : ShapeTensor,
) -> ShapeTensor {
  let batch = input.shape[0]
  let out_f = weight.shape[0]
  { shape: FixedArray::from_iter([batch, out_f].iter()) }
}
