// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub impl Add for NpArray with add(x : NpArray, y : NpArray) {
  broadcast_op((x, y) => x + y, x, y)
}

///|
pub impl Sub for NpArray with sub(x : NpArray, y : NpArray) {
  broadcast_op((x, y) => x - y, x, y)
}

///|
fn _copy_shape(s : @tensor.ShapeTensor) -> @tensor.ShapeTensor {
  { shape: FixedArray::makei(s.shape.length(), fn(i) { s.shape[i] }) }
}

///|
pub fn NpArray::from1d(arr : ArrayView[Float]) -> NpArray {
  {
    shape: @tensor.ShapeTensor::from_array([arr.length()]),
    data: FixedArray::from_array(arr),
  }
}

///|
pub fn NpArray::from2d(arr : ArrayView[ArrayView[Float]]) -> NpArray {
  let data = FixedArray::from_iter(arr.iter().map(fn(x) { x.iter() }).flatten())
  {
    shape: @tensor.ShapeTensor::from_array([arr.length(), arr[0].length()]),
    data,
  }
}

///|
pub impl @tensor.Tensor for NpArray with dims(x : NpArray) -> FixedArray[Int] {
  x.shape.shape
}

///|
pub impl @tensor.Tensor for NpArray with zeros(dims : FixedArray[Int]) -> NpArray {
  let total = dims.iter().fold(init=1, fn(acc, d) { acc * d })
  { shape: { shape: dims }, data: FixedArray::make(total, 0.0) }
}

///|
pub impl @tensor.Tensor for NpArray with zeros_like(x : NpArray) -> NpArray {
  { shape: _copy_shape(x.shape), data: FixedArray::make(x.data.length(), 0) }
}

///|
pub impl @tensor.Tensor for NpArray with from_host(
  data : FixedArray[Float],
  shape : Array[Int],
) -> NpArray {
  { shape: @tensor.ShapeTensor::from_array(shape), data }
}

///|
pub impl @tensor.Tensor for NpArray with square(x : NpArray) -> NpArray {
  let n = x.data.length()
  {
    shape: _copy_shape(x.shape),
    data: FixedArray::makei(n, fn(i) { x.data[i] * x.data[i] }),
  }
}

///|
pub impl @tensor.Tensor for NpArray with sqrt(x : NpArray) -> NpArray {
  let n = x.data.length()
  {
    shape: _copy_shape(x.shape),
    data: FixedArray::makei(n, fn(i) { Float::sqrt(x.data[i]) }),
  }
}

///|
pub impl @tensor.Tensor for NpArray with mean(x : NpArray) -> NpArray {
  let sum = x.data.fold(init=(0.0 : Float), fn(acc, v) { acc + v })
  let n = Float::from_int(x.data.length())
  {
    shape: @tensor.ShapeTensor::from_array([1]),
    data: FixedArray::make(1, sum / n),
  }
}

///|
pub impl @tensor.Tensor for NpArray with scale(x : NpArray, s : Float) -> NpArray {
  let n = x.data.length()
  {
    shape: _copy_shape(x.shape),
    data: FixedArray::makei(n, fn(i) { x.data[i] * s }),
  }
}

///|
pub impl @tensor.Tensor for NpArray with mul_elem(x : NpArray, y : NpArray) -> NpArray {
  broadcast_op((a, b) => a * b, x, y)
}

///|
pub impl @tensor.Tensor for NpArray with div_elem(x : NpArray, y : NpArray) -> NpArray {
  broadcast_op((a, b) => a / b, x, y)
}

///|
pub impl @tensor.Tensor for NpArray with scalar(v : Float) -> NpArray {
  { shape: @tensor.ShapeTensor::from_array([1]), data: FixedArray::make(1, v) }
}

///|
pub impl @tensor.Tensor for NpArray with size(x : NpArray) -> Int {
  x.data.length()
}

///|
pub impl @tensor.Tensor for NpArray with reduce_sum_to(
  x : NpArray,
  target_dims : FixedArray[Int],
) -> NpArray {
  if x.shape.shape == target_dims {
    return x
  }
  let xd = x.shape.shape.length()
  let td = target_dims.length()
  let offset = xd - td
  let mut result = x.data.copy()
  for dim_idx in 0..<offset {
    let dim_size = x.shape.shape[dim_idx]
    let stride_out = result.length() / dim_size
    for o in 0..<stride_out {
      let mut sum = (0.0 : Float)
      for k in 0..<dim_size {
        sum = sum + result[k * stride_out + o]
      }
      result[o] = sum
    }
    result = FixedArray::from_iter(result.iter().take(stride_out))
  }
  for dim_idx in 0..<td {
    if target_dims[dim_idx] == 1 && x.shape.shape[offset + dim_idx] != 1 {
      let dim_size = x.shape.shape[offset + dim_idx]
      let mut stride_after = 1
      for j in (dim_idx + 1)..<td {
        stride_after = stride_after * target_dims[j]
      }
      let stride_before = result.length() / (dim_size * stride_after)
      let new_data = FixedArray::make(
        stride_before * stride_after,
        (0.0 : Float),
      )
      for b in 0..<stride_before {
        for a in 0..<stride_after {
          let mut sum = (0.0 : Float)
          for k in 0..<dim_size {
            sum = sum +
              result[b * dim_size * stride_after + k * stride_after + a]
          }
          new_data[b * stride_after + a] = sum
        }
      }
      result = new_data
    }
  }
  { shape: { shape: target_dims }, data: result }
}

///|
pub impl @tensor.Tensor for NpArray with value(x : NpArray) -> @tensor.TensorData {
  { dims: x.shape.shape, data: x.data }
}

///|
pub impl @tensor.Tensor for NpArray with view(
  x : NpArray,
  new_shape : FixedArray[Int],
) -> NpArray {
  let total = new_shape.iter().fold(init=1, fn(acc, d) { acc * d })
  if total != x.data.length() {
    abort("view size mismatch: \{total} != \{x.data.length()}")
  }
  { shape: { shape: new_shape }, data: x.data }
}

///|
pub impl @tensor.Tensor for NpArray with add(x : NpArray, y : NpArray) -> NpArray {
  broadcast_op((a, b) => a + b, x, y)
}

///|
pub impl @tensor.Tensor for NpArray with add_into(x : NpArray, y : NpArray) -> Unit {
  let n = x.data.length()
  for i in 0..<n {
    x.data[i] = x.data[i] + y.data[i]
  }
}

///|
pub impl @tensor.Tensor for NpArray with sub(x : NpArray, y : NpArray) -> NpArray {
  broadcast_op((a, b) => a - b, x, y)
}

///|
pub impl @tensor.BlasTensor for NpArray with matmul(a : NpArray, b : NpArray) -> NpArray {
  let n = a.shape.shape[0]
  let m = a.shape.shape[1]
  debug_assert(() => a.shape.shape[1] == b.shape.shape[0])
  let p = b.shape.shape[1]
  // Transpose B for cache-friendly row-major access
  let bt : FixedArray[Float] = FixedArray::make(m * p, 0.0)
  for k in 0..<m {
    for j in 0..<p {
      bt[j * m + k] = b.data[k * p + j]
    }
  }
  // Tiled matmul with 32×32 blocks for L1 cache locality
  let data : FixedArray[Float] = FixedArray::make(n * p, 0.0)
  let tile = 32
  let mut ii = 0
  while ii < n {
    let ii_end = if ii + tile < n { ii + tile } else { n }
    let mut jj = 0
    while jj < p {
      let jj_end = if jj + tile < p { jj + tile } else { p }
      let mut i = ii
      while i < ii_end {
        let a_base = i * m
        let c_base = i * p
        let mut j = jj
        while j < jj_end {
          let bt_base = j * m
          let mut sum = (0.0 : Float)
          let mut k = 0
          while k < m {
            sum = sum + a.data[a_base + k] * bt[bt_base + k]
            k += 1
          }
          data[c_base + j] = sum
          j += 1
        }
        i += 1
      }
      jj += tile
    }
    ii += tile
  }
  { shape: @tensor.ShapeTensor::from_array([n, p]), data }
}

///|
pub impl @tensor.BlasTensor for NpArray with transpose(x : NpArray) -> NpArray {
  let n = x.shape.shape[0]
  let m = x.shape.shape[1]
  let data = FixedArray::makei(n * m, fn(idx) {
    let i = idx / n
    let j = idx % n
    x.data[j * m + i]
  })
  { shape: @tensor.ShapeTensor::from_array([m, n]), data }
}

///|
pub impl @tensor.Tensor for NpArray with broadcast_to(
  x : NpArray,
  target_shape : FixedArray[Int],
) -> NpArray {
  let scalar_val = x.data[0]
  let mut n = 1
  for i in 0..<target_shape.length() {
    n = n * target_shape[i]
  }
  let shape = FixedArray::from_iter(target_shape.iter())
  { shape: { shape, }, data: FixedArray::make(n, scalar_val) }
}

///|
/// CPU implementation of fused batch SGD+momentum with auto gradient clipping.
/// Follows the same algorithm as the GPU fused kernel.
pub impl @optimizer.BatchSgdMomentumAutoclip for NpArray with batch_sgd_momentum_autoclip(
  params : Array[NpArray],
  grads : Array[NpArray],
  velocities : Array[NpArray],
  config : @optimizer.SgdMomentumConfig,
) -> Unit {
  let { lr, momentum, weight_decay, max_grad_norm } = config
  // 1. Compute global grad norm
  let mut norm_sq : Float = 0.0
  for i in 0..<grads.length() {
    let g = grads[i]
    for j in 0..<g.data.length() {
      norm_sq = norm_sq + g.data[j] * g.data[j]
    }
  }
  let norm = norm_sq.sqrt()
  let clip_coef = if norm > max_grad_norm { max_grad_norm / norm } else { 1.0 }

  // 2. Update each param: v = μ·v + clip(g) + λ·p, then p = p - lr·v
  for i in 0..<params.length() {
    let p = params[i]
    let g = grads[i]
    let v = velocities[i]
    for j in 0..<p.data.length() {
      let clipped_grad = g.data[j] * clip_coef
      velocities[i].data[j] = momentum * v.data[j] +
        clipped_grad +
        weight_decay * p.data[j]
      params[i].data[j] = p.data[j] - lr * velocities[i].data[j]
    }
  }
}
