// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Helper: assert a ShapeTensor's shape matches an expected Array[Int].
fn assert_shape(tensor : ShapeTensor, expected : Array[Int]) -> Unit {
  let s = tensor.shape
  guard s.length() == expected.length() else {
    abort("assert_shape: ndim mismatch: \{s.length()} vs \{expected.length()}")
  }
  for i in 0..<s.length() {
    guard s[i] == expected[i] else {
      abort("assert_shape: mismatch at dim \{i}: \{s[i]} vs \{expected[i]}")
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Helper function tests
// ═══════════════════════════════════════════════════════════════

///|
test "shape_numel" {
  assert_eq(shape_numel(FixedArray::from_iter([2, 3, 4].iter())), 24)
  assert_eq(shape_numel(FixedArray::from_iter([1].iter())), 1)
  assert_eq(shape_numel(FixedArray::from_iter([].iter())), 1)
}

///|
test "ShapeTensor::numel" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3].iter()) }
  assert_eq(t.numel(), 6)
}

///|
test "shape_broadcast same shapes" {
  let a = FixedArray::from_iter([2, 3].iter())
  let b = FixedArray::from_iter([2, 3].iter())
  let result = shape_broadcast(a, b)
  assert_eq(result.length(), 2)
  assert_eq(result[0], 2)
  assert_eq(result[1], 3)
}

///|
test "shape_broadcast scalar with vector" {
  let a = FixedArray::from_iter([1].iter())
  let b = FixedArray::from_iter([5].iter())
  let result = shape_broadcast(a, b)
  assert_eq(result.length(), 1)
  assert_eq(result[0], 5)
}

///|
test "shape_broadcast (3,) + (2,3)" {
  let a = FixedArray::from_iter([3].iter())
  let b = FixedArray::from_iter([2, 3].iter())
  let result = shape_broadcast(a, b)
  assert_eq(result.length(), 2)
  assert_eq(result[0], 2)
  assert_eq(result[1], 3)
}

///|
test "shape_broadcast (2,1) + (1,3)" {
  let a = FixedArray::from_iter([2, 1].iter())
  let b = FixedArray::from_iter([1, 3].iter())
  let result = shape_broadcast(a, b)
  assert_eq(result.length(), 2)
  assert_eq(result[0], 2)
  assert_eq(result[1], 3)
}

// ═══════════════════════════════════════════════════════════════
// Tensor trait tests
// ═══════════════════════════════════════════════════════════════

///|
test "Tensor: dims" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3, 4].iter()) }
  let d = ShapeTensor::dims(t)
  assert_eq(d.length(), 3)
  assert_eq(d[0], 2)
  assert_eq(d[1], 3)
  assert_eq(d[2], 4)
}

///|
test "Tensor: zeros" {
  let dims = FixedArray::from_iter([4, 5].iter())
  let t = ShapeTensor::zeros(dims)
  assert_shape(t, [4, 5])
}

///|
test "Tensor: zeros_like" {
  let original : ShapeTensor = { shape: FixedArray::from_iter([3, 7].iter()) }
  let t = ShapeTensor::zeros_like(original)
  assert_shape(t, [3, 7])
}

///|
test "Tensor: from_host" {
  let data : FixedArray[Float] = FixedArray::make(6, 0.0)
  let t = ShapeTensor::from_host(data, [2, 3])
  assert_shape(t, [2, 3])
}

///|
test "Tensor: square preserves shape" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3].iter()) }
  let s = ShapeTensor::square(t)
  assert_shape(s, [2, 3])
}

///|
test "Tensor: mean returns scalar shape" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([4, 5, 6].iter()) }
  let m = ShapeTensor::mean(t)
  assert_shape(m, [1])
}

///|
test "Tensor: scale preserves shape" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  let s = ShapeTensor::scale(t, 2.5)
  assert_shape(s, [3, 4])
}

///|
test "Tensor: mul_elem broadcasts shapes" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([2, 1].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([1, 3].iter()) }
  let result = ShapeTensor::mul_elem(a, b)
  assert_shape(result, [2, 3])
}

///|
test "Tensor: mul_elem same shape" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  let result = ShapeTensor::mul_elem(a, b)
  assert_shape(result, [3, 4])
}

///|
test "Tensor: scalar" {
  let s = ShapeTensor::scalar(42.0)
  assert_shape(s, [1])
}

///|
test "Tensor: size" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3, 4].iter()) }
  assert_eq(ShapeTensor::size(t), 24)
}

///|
test "Tensor: size scalar" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([1].iter()) }
  assert_eq(ShapeTensor::size(t), 1)
}

///|
test "Tensor: reduce_sum_to" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3, 4].iter()) }
  let target = FixedArray::from_iter([1, 3, 4].iter())
  let result = ShapeTensor::reduce_sum_to(t, target)
  assert_shape(result, [1, 3, 4])
}

///|
test "Tensor: value returns TensorData" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3].iter()) }
  let v = ShapeTensor::value(t)
  assert_eq(v.dims.length(), 2)
  assert_eq(v.dims[0], 2)
  assert_eq(v.dims[1], 3)
  assert_eq(v.data.length(), 6)
}

///|
test "Tensor: view changes shape" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3].iter()) }
  let new_shape = FixedArray::from_iter([3, 2].iter())
  let v = ShapeTensor::view(t, new_shape)
  assert_shape(v, [3, 2])
}

///|
test "Tensor: view 4D reshape" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([8].iter()) }
  let new_shape = FixedArray::from_iter([2, 2, 2].iter())
  let v = ShapeTensor::view(t, new_shape)
  assert_shape(v, [2, 2, 2])
}

///|
test "Tensor: add broadcasts" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([4, 1].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([1, 5].iter()) }
  let result = ShapeTensor::add(a, b)
  assert_shape(result, [4, 5])
}

///|
test "Tensor: add same shape" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  let result = ShapeTensor::add(a, b)
  assert_shape(result, [3, 4])
}

///|
test "Tensor: add_into matching shapes" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  ShapeTensor::add_into(a, b)
}

///|
test "Tensor: sub broadcasts" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([2, 3].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([1, 3].iter()) }
  let result = ShapeTensor::sub(a, b)
  assert_shape(result, [2, 3])
}

///|
test "Tensor: broadcast_to" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([1].iter()) }
  let target = FixedArray::from_iter([3, 4].iter())
  let result = ShapeTensor::broadcast_to(t, target)
  assert_shape(result, [3, 4])
}

// ═══════════════════════════════════════════════════════════════
// BlasTensor trait tests
// ═══════════════════════════════════════════════════════════════

///|
test "BlasTensor: matmul" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([3, 4].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([4, 5].iter()) }
  let c = ShapeTensor::matmul(a, b)
  assert_shape(c, [3, 5])
}

///|
test "BlasTensor: matmul square" {
  let a : ShapeTensor = { shape: FixedArray::from_iter([2, 2].iter()) }
  let b : ShapeTensor = { shape: FixedArray::from_iter([2, 2].iter()) }
  let c = ShapeTensor::matmul(a, b)
  assert_shape(c, [2, 2])
}

///|
test "BlasTensor: transpose" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([3, 5].iter()) }
  let r = ShapeTensor::transpose(t)
  assert_shape(r, [5, 3])
}

// ═══════════════════════════════════════════════════════════════
// ImageTensor trait tests — Forward
// ═══════════════════════════════════════════════════════════════

///|
test "ImageTensor: conv2d basic" {
  // input [1, 3, 32, 32], weight [16, 3, 3, 3], stride=1, padding=1
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([1, 3, 32, 32].iter()),
  }
  let weight : ShapeTensor = {
    shape: FixedArray::from_iter([16, 3, 3, 3].iter()),
  }
  let bias : ShapeTensor = { shape: FixedArray::from_iter([16].iter()) }
  let result = ShapeTensor::conv2d(input, weight, bias, 1, 1)
  // OH = (32 + 2*1 - 3) / 1 + 1 = 32, OW = 32
  assert_shape(result, [1, 16, 32, 32])
}

///|
test "ImageTensor: conv2d stride=2 no padding" {
  // input [2, 1, 8, 8], weight [4, 1, 3, 3], stride=2, padding=0
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([2, 1, 8, 8].iter()),
  }
  let weight : ShapeTensor = {
    shape: FixedArray::from_iter([4, 1, 3, 3].iter()),
  }
  let bias : ShapeTensor = { shape: FixedArray::from_iter([4].iter()) }
  let result = ShapeTensor::conv2d(input, weight, bias, 2, 0)
  // OH = (8 + 0 - 3) / 2 + 1 = 3, OW = 3
  assert_shape(result, [2, 4, 3, 3])
}

///|
test "ImageTensor: conv2d large kernel with padding" {
  // input [1, 3, 28, 28], weight [32, 3, 5, 5], stride=1, padding=2
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([1, 3, 28, 28].iter()),
  }
  let weight : ShapeTensor = {
    shape: FixedArray::from_iter([32, 3, 5, 5].iter()),
  }
  let bias : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let result = ShapeTensor::conv2d(input, weight, bias, 1, 2)
  // OH = (28 + 4 - 5) / 1 + 1 = 28, OW = 28
  assert_shape(result, [1, 32, 28, 28])
}

///|
test "ImageTensor: relu preserves shape" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 3, 8, 8].iter()) }
  let r = ShapeTensor::relu(t)
  assert_shape(r, [2, 3, 8, 8])
}

///|
test "ImageTensor: maxpool2d basic" {
  // input [1, 3, 32, 32], kernel=2, stride=2
  let t : ShapeTensor = { shape: FixedArray::from_iter([1, 3, 32, 32].iter()) }
  let r = ShapeTensor::maxpool2d(t, 2, 2)
  // OH = (32 - 2) / 2 + 1 = 16, OW = 16
  assert_shape(r, [1, 3, 16, 16])
}

///|
test "ImageTensor: maxpool2d non-divisible" {
  // input [1, 1, 7, 7], kernel=3, stride=2
  let t : ShapeTensor = { shape: FixedArray::from_iter([1, 1, 7, 7].iter()) }
  let r = ShapeTensor::maxpool2d(t, 3, 2)
  // OH = (7 - 3) / 2 + 1 = 3, OW = 3
  assert_shape(r, [1, 1, 3, 3])
}

///|
test "ImageTensor: adaptive_avg_pool2d" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([4, 64, 8, 8].iter()) }
  let r = ShapeTensor::adaptive_avg_pool2d(t, 1)
  assert_shape(r, [4, 64, 1, 1])
}

///|
test "ImageTensor: adaptive_avg_pool2d output_size=4" {
  let t : ShapeTensor = { shape: FixedArray::from_iter([2, 32, 16, 16].iter()) }
  let r = ShapeTensor::adaptive_avg_pool2d(t, 4)
  assert_shape(r, [2, 32, 4, 4])
}

///|
test "ImageTensor: batchnorm_training" {
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([8, 64, 16, 16].iter()),
  }
  let gamma : ShapeTensor = { shape: FixedArray::from_iter([64].iter()) }
  let beta : ShapeTensor = { shape: FixedArray::from_iter([64].iter()) }
  let running_mean : ShapeTensor = { shape: FixedArray::from_iter([64].iter()) }
  let running_var : ShapeTensor = { shape: FixedArray::from_iter([64].iter()) }
  let (output, save_mean, save_inv_var) = ShapeTensor::batchnorm_training(
    input, gamma, beta, running_mean, running_var, 0.1, 0.00001,
  )
  assert_shape(output, [8, 64, 16, 16])
  assert_shape(save_mean, [64])
  assert_shape(save_inv_var, [64])
}

///|
test "ImageTensor: batchnorm_inference" {
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([4, 32, 8, 8].iter()),
  }
  let gamma : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let beta : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let rm : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let rv : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let output = ShapeTensor::batchnorm_inference(
    input, gamma, beta, rm, rv, 0.00001,
  )
  assert_shape(output, [4, 32, 8, 8])
}

///|
test "ImageTensor: softmax_cross_entropy returns scalar" {
  let logits : ShapeTensor = { shape: FixedArray::from_iter([8, 10].iter()) }
  let targets : ShapeTensor = { shape: FixedArray::from_iter([8, 10].iter()) }
  let loss = ShapeTensor::softmax_cross_entropy(logits, targets)
  assert_shape(loss, [1])
}

///|
test "ImageTensor: cross_entropy_with_labels returns scalar" {
  let logits : ShapeTensor = { shape: FixedArray::from_iter([4, 10].iter()) }
  let labels : ShapeTensor = { shape: FixedArray::from_iter([4].iter()) }
  let loss = ShapeTensor::cross_entropy_with_labels(logits, labels, 10)
  assert_shape(loss, [1])
}

// ═══════════════════════════════════════════════════════════════
// ImageTensor trait tests — Backward
// ═══════════════════════════════════════════════════════════════

///|
test "ImageTensor: conv2d_backward_data returns input shape" {
  let grad_output : ShapeTensor = {
    shape: FixedArray::from_iter([1, 16, 32, 32].iter()),
  }
  let weight : ShapeTensor = {
    shape: FixedArray::from_iter([16, 3, 3, 3].iter()),
  }
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([1, 3, 32, 32].iter()),
  }
  let grad_input = ShapeTensor::conv2d_backward_data(
    grad_output, weight, input, 1, 1,
  )
  assert_shape(grad_input, [1, 3, 32, 32])
}

///|
test "ImageTensor: conv2d_backward_weight returns weight shape" {
  let grad_output : ShapeTensor = {
    shape: FixedArray::from_iter([2, 8, 16, 16].iter()),
  }
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([2, 3, 16, 16].iter()),
  }
  let weight : ShapeTensor = {
    shape: FixedArray::from_iter([8, 3, 3, 3].iter()),
  }
  let grad_weight = ShapeTensor::conv2d_backward_weight(
    grad_output, input, weight, 1, 0,
  )
  assert_shape(grad_weight, [8, 3, 3, 3])
}

///|
test "ImageTensor: conv2d_backward_bias returns [C_out]" {
  let grad_output : ShapeTensor = {
    shape: FixedArray::from_iter([4, 16, 8, 8].iter()),
  }
  let grad_bias = ShapeTensor::conv2d_backward_bias(grad_output)
  assert_shape(grad_bias, [16])
}

///|
test "ImageTensor: relu_backward returns input shape" {
  let grad_output : ShapeTensor = {
    shape: FixedArray::from_iter([2, 3, 4, 4].iter()),
  }
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([2, 3, 4, 4].iter()),
  }
  let grad_input = ShapeTensor::relu_backward(grad_output, input)
  assert_shape(grad_input, [2, 3, 4, 4])
}

///|
test "ImageTensor: batchnorm_backward returns all grads" {
  let grad_output : ShapeTensor = {
    shape: FixedArray::from_iter([4, 32, 8, 8].iter()),
  }
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([4, 32, 8, 8].iter()),
  }
  let gamma : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let save_mean : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let save_inv_var : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let (grad_input, grad_gamma, grad_beta) = ShapeTensor::batchnorm_backward(
    grad_output, input, gamma, save_mean, save_inv_var, 0.00001,
  )
  assert_shape(grad_input, [4, 32, 8, 8])
  assert_shape(grad_gamma, [32])
  assert_shape(grad_beta, [32])
}

///|
test "ImageTensor: maxpool2d_backward returns input shape" {
  let grad_output : ShapeTensor = {
    shape: FixedArray::from_iter([1, 3, 16, 16].iter()),
  }
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([1, 3, 32, 32].iter()),
  }
  let grad_input = ShapeTensor::maxpool2d_backward(grad_output, input, 2, 2)
  assert_shape(grad_input, [1, 3, 32, 32])
}

///|
test "ImageTensor: adaptive_avg_pool2d_backward returns input shape" {
  let grad_output : ShapeTensor = {
    shape: FixedArray::from_iter([2, 64, 1, 1].iter()),
  }
  let input : ShapeTensor = {
    shape: FixedArray::from_iter([2, 64, 8, 8].iter()),
  }
  let grad_input = ShapeTensor::adaptive_avg_pool2d_backward(grad_output, input)
  assert_shape(grad_input, [2, 64, 8, 8])
}

///|
test "ImageTensor: softmax_ce_backward returns logits shape" {
  let logits : ShapeTensor = { shape: FixedArray::from_iter([8, 10].iter()) }
  let targets : ShapeTensor = { shape: FixedArray::from_iter([8, 10].iter()) }
  let grad = ShapeTensor::softmax_ce_backward(logits, targets)
  assert_shape(grad, [8, 10])
}

///|
test "ImageTensor: softmax_ce_backward_labels returns logits shape" {
  let logits : ShapeTensor = { shape: FixedArray::from_iter([4, 10].iter()) }
  let labels : ShapeTensor = { shape: FixedArray::from_iter([4].iter()) }
  let grad = ShapeTensor::softmax_ce_backward_labels(logits, labels, 10)
  assert_shape(grad, [4, 10])
}

// ═══════════════════════════════════════════════════════════════
// ResNet-18-like shape propagation integration test
// ═══════════════════════════════════════════════════════════════

///|
test "ResNet-like forward shape propagation" {
  // Input: [1, 3, 32, 32] (CIFAR-10 sized)
  let x = ShapeTensor::from_host(
    (FixedArray::make(3072, 0.0) : FixedArray[Float]),
    [1, 3, 32, 32],
  )

  // Conv1: 3→16, kernel=3, stride=1, padding=1
  let w1 : ShapeTensor = { shape: FixedArray::from_iter([16, 3, 3, 3].iter()) }
  let b1 : ShapeTensor = { shape: FixedArray::from_iter([16].iter()) }
  let x = ShapeTensor::conv2d(x, w1, b1, 1, 1)
  assert_shape(x, [1, 16, 32, 32])

  // ReLU
  let x = ShapeTensor::relu(x)
  assert_shape(x, [1, 16, 32, 32])

  // MaxPool: kernel=2, stride=2
  let x = ShapeTensor::maxpool2d(x, 2, 2)
  assert_shape(x, [1, 16, 16, 16])

  // Conv2: 16→32, kernel=3, stride=1, padding=1
  let w2 : ShapeTensor = { shape: FixedArray::from_iter([32, 16, 3, 3].iter()) }
  let b2 : ShapeTensor = { shape: FixedArray::from_iter([32].iter()) }
  let x = ShapeTensor::conv2d(x, w2, b2, 1, 1)
  assert_shape(x, [1, 32, 16, 16])

  // ReLU
  let x = ShapeTensor::relu(x)

  // MaxPool: kernel=2, stride=2
  let x = ShapeTensor::maxpool2d(x, 2, 2)
  assert_shape(x, [1, 32, 8, 8])

  // Adaptive avg pool → [1, 32, 1, 1]
  let x = ShapeTensor::adaptive_avg_pool2d(x, 1)
  assert_shape(x, [1, 32, 1, 1])

  // Flatten: view as [1, 32]
  let x = ShapeTensor::view(x, FixedArray::from_iter([1, 32].iter()))
  assert_shape(x, [1, 32])

  // Linear (matmul): [1, 32] @ [32, 10] → [1, 10]
  let w_fc : ShapeTensor = { shape: FixedArray::from_iter([32, 10].iter()) }
  let logits = ShapeTensor::matmul(x, w_fc)
  assert_shape(logits, [1, 10])

  // Cross-entropy loss
  let labels : ShapeTensor = { shape: FixedArray::from_iter([1].iter()) }
  let loss = ShapeTensor::cross_entropy_with_labels(logits, labels, 10)
  assert_shape(loss, [1])
}
