// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub(all) struct NpArray {
  mut shape : @tensor.ShapeTensor
  mut data : FixedArray[Float]
} derive(Debug)

///|
fn[T : Compare] max(a : T, b : T) -> T {
  if a > b {
    a
  } else {
    b
  }
}

///|
fn make_stride(dims : ArrayView[Int]) -> Array[Int] {
  let n = dims.length()
  let stride = Array::make(n, 0)
  let mut last = 1
  for i in n>..0 {
    last *= dims[i]
    stride[i] = last
  }
  stride
}

///|
fn broadcast_op(
  op : (Float, Float) -> Float,
  x : NpArray,
  y : NpArray,
) -> NpArray {
  // Fast path: identical shapes - skip stride tracking entirely
  if x.shape.shape == y.shape.shape {
    let n = x.data.length()
    let o_data = FixedArray::makei(n, fn(i) { op(x.data[i], y.data[i]) })
    return { shape: { shape: x.shape.shape.copy() }, data: o_data }
  }
  let dx = x.shape.shape.length()
  let dy = y.shape.shape.length()
  let d = max(dx, dy)
  let x_dim = FixedArray::makei(d, i => {
    x.shape.shape.get(i - (d - dx)).unwrap_or(1)
  })
  let y_dim = FixedArray::makei(d, i => {
    y.shape.shape.get(i - (d - dy)).unwrap_or(1)
  })
  let o_dim = FixedArray::makei(d, i => {
    match (x_dim[i], y_dim[i]) {
      (l, r) if l == r => l
      (1, r) => r
      (l, 1) => l
      (l, r) =>
        abort(
          "invalid dim at pos \{i} : \{l} != \{r}, x_dim \{@debug.to_string(x_dim)} y_dim \{@debug.to_string(y_dim)}",
        )
    }
  })
  let x_stride = make_stride(x_dim)
  let y_stride = make_stride(y_dim)
  let o_stride = make_stride(o_dim)
  let x_idx = FixedArray::make(d, 0)
  let y_idx = FixedArray::make(d, 0)
  let o_data : FixedArray[Float] = FixedArray::make(o_stride[0], 0.0)
  let mut xi = 0
  let mut yi = 0
  fn step(
    i : Int,
    dim : FixedArray[Int],
    o_dim : FixedArray[Int],
    stride : Array[Int],
    idx : FixedArray[Int],
  ) -> Int {
    let mut i = i
    i += 1
    for p in d>..0 {
      idx[p] += 1
      if idx[p] != o_dim[p] {
        if dim[p] == 1 {
          i -= stride.get(p + 1).unwrap_or(1)
        }
        break
      }
      idx[p] = 0
    }
    i
  }
  for i in 0..<o_data.length() {
    o_data[i] = op(x.data[xi], y.data[yi])
    xi = step(xi, x_dim, o_dim, x_stride, x_idx)
    yi = step(yi, y_dim, o_dim, y_stride, y_idx)
  }
  { shape: { shape: o_dim }, data: o_data }
}
