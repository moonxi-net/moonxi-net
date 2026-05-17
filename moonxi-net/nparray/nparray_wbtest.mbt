// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn np_eq(a : NpArray, b : Array[Float]) -> Bool {
  if a.data.length() != b.length() {
    return false
  }
  for i in 0..<a.data.length() {
    if a.data[i] != b[i] {
      return false
    }
  }
  true
}

///|
test "broadcast_op same shape addition" {
  let x = NpArray::from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let y = NpArray::from2d([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
  let result = broadcast_op(fn(a, b) { a + b }, x, y)
  assert_true(np_eq(result, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]))
  assert_eq(result.shape.shape[0], 2)
  assert_eq(result.shape.shape[1], 3)
}

///|
test "broadcast_op 1d with scalar-like" {
  let x : NpArray = {
    shape: @tensor.ShapeTensor::from_array([3]),
    data: FixedArray::from_array([1.0, 2.0, 3.0]),
  }
  let y : NpArray = {
    shape: @tensor.ShapeTensor::from_array([1]),
    data: FixedArray::from_array([10.0]),
  }
  let result = broadcast_op(fn(a, b) { a + b }, x, y)
  assert_true(np_eq(result, [11.0, 12.0, 13.0]))
}

///|
test "broadcast_op (2,3) + (4,1,3) -> (4,2,3)" {
  let x = NpArray::from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let y_data : FixedArray[Float] = FixedArray::from_array([
    10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
  ])
  let y : NpArray = {
    shape: @tensor.ShapeTensor::from_array([4, 1, 3]),
    data: y_data,
  }
  let result = broadcast_op(fn(a, b) { a + b }, x, y)
  assert_eq(result.shape.shape.length(), 3)
  assert_eq(result.shape.shape[0], 4)
  assert_eq(result.shape.shape[1], 2)
  assert_eq(result.shape.shape[2], 3)
  assert_true(
    np_eq(result, [
      11.0, 22.0, 33.0, 14.0, 25.0, 36.0, 41.0, 52.0, 63.0, 44.0, 55.0, 66.0, 71.0,
      82.0, 93.0, 74.0, 85.0, 96.0, 101.0, 112.0, 123.0, 104.0, 115.0, 126.0,
    ]),
  )
}

///|
test "broadcast_op multiplication" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let y = NpArray::from2d([[10.0, 20.0], [30.0, 40.0]])
  let result = broadcast_op(fn(a, b) { a * b }, x, y)
  assert_true(np_eq(result, [10.0, 40.0, 90.0, 160.0]))
}

///|
test "broadcast_op subtraction" {
  let x = NpArray::from2d([[5.0, 6.0], [7.0, 8.0]])
  let y = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let result = broadcast_op(fn(a, b) { a - b }, x, y)
  assert_true(np_eq(result, [4.0, 4.0, 4.0, 4.0]))
}

///|
test "broadcast_op (1,3) + (2,1) -> (2,3)" {
  let x : NpArray = {
    shape: @tensor.ShapeTensor::from_array([1, 3]),
    data: FixedArray::from_array([1.0, 2.0, 3.0]),
  }
  let y : NpArray = {
    shape: @tensor.ShapeTensor::from_array([2, 1]),
    data: FixedArray::from_array([10.0, 20.0]),
  }
  let result = broadcast_op(fn(a, b) { a + b }, x, y)
  assert_eq(result.shape.shape[0], 2)
  assert_eq(result.shape.shape[1], 3)
  assert_true(np_eq(result, [11.0, 12.0, 13.0, 21.0, 22.0, 23.0]))
}

///|
test "broadcast_op (3,) + (3,) same shape 1d" {
  let x : NpArray = {
    shape: @tensor.ShapeTensor::from_array([3]),
    data: FixedArray::from_array([1.0, 2.0, 3.0]),
  }
  let y : NpArray = {
    shape: @tensor.ShapeTensor::from_array([3]),
    data: FixedArray::from_array([4.0, 5.0, 6.0]),
  }
  let result = broadcast_op(fn(a, b) { a + b }, x, y)
  assert_true(np_eq(result, [5.0, 7.0, 9.0]))
}

///|
test "Add impl" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let y = NpArray::from2d([[5.0, 6.0], [7.0, 8.0]])
  let result = x + y
  assert_true(np_eq(result, [6.0, 8.0, 10.0, 12.0]))
}

///|
test "Sub impl" {
  let x = NpArray::from2d([[5.0, 6.0], [7.0, 8.0]])
  let y = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let result = x - y
  assert_true(np_eq(result, [4.0, 4.0, 4.0, 4.0]))
}

///|
test "Tensor: zeros" {
  let arr = NpArray::zeros([2, 3])
  assert_eq(arr.shape.shape[0], 2)
  assert_eq(arr.shape.shape[1], 3)
  assert_eq(arr.data.length(), 6)
  assert_true(np_eq(arr, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
}

///|
test "Tensor: zeros_like" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let y = NpArray::zeros_like(x)
  assert_eq(y.shape.shape[0], 2)
  assert_eq(y.shape.shape[1], 2)
  assert_true(np_eq(y, [0.0, 0.0, 0.0, 0.0]))
}

///|
test "Tensor: square" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let y = NpArray::square(x)
  assert_true(np_eq(y, [1.0, 4.0, 9.0, 16.0]))
}

///|
test "Tensor: mean" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let m = NpArray::mean(x)
  assert_eq(m.data.length(), 1)
  assert_eq(m.data[0], 2.5)
}

///|
test "Tensor: scale" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let y = NpArray::scale(x, 2.0)
  assert_true(np_eq(y, [2.0, 4.0, 6.0, 8.0]))
}

///|
test "Tensor: scalar" {
  let s = NpArray::scalar(42.0)
  assert_eq(s.data.length(), 1)
  assert_eq(s.data[0], 42.0)
}

///|
test "Tensor: size" {
  let x = NpArray::from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  assert_eq(NpArray::size(x), 6)
}

///|
test "Tensor: mul_elem" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let y = NpArray::from2d([[2.0, 3.0], [4.0, 5.0]])
  let result = NpArray::mul_elem(x, y)
  assert_true(np_eq(result, [2.0, 6.0, 12.0, 20.0]))
}

///|
test "Tensor: view" {
  let x = NpArray::from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let y = NpArray::view(x, [3, 2])
  assert_eq(y.shape.shape[0], 3)
  assert_eq(y.shape.shape[1], 2)
  assert_true(np_eq(y, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

///|
test "Tensor: reduce_sum_to" {
  let x = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  // Reduce first dim from [2,2] to [1,2]: sum along axis 0
  let result = NpArray::reduce_sum_to(x, [1, 2])
  assert_true(np_eq(result, [4.0, 6.0]))
}

///|
test "BlasTensor: matmul" {
  let a = NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let b = NpArray::from2d([[5.0, 6.0], [7.0, 8.0]])
  let c = NpArray::matmul(a, b)
  // [1,2] [5,6] -> [1*5+2*7, 1*6+2*8] = [19, 22]
  // [3,4] [5,6] -> [3*5+4*7, 3*6+4*8] = [43, 50]
  assert_true(np_eq(c, [19.0, 22.0, 43.0, 50.0]))
  assert_eq(c.shape.shape[0], 2)
  assert_eq(c.shape.shape[1], 2)
}

///|
test "BlasTensor: transpose" {
  let x = NpArray::from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let y = NpArray::transpose(x)
  assert_eq(y.shape.shape[0], 3)
  assert_eq(y.shape.shape[1], 2)
  // transposed: [[1,4], [2,5], [3,6]] = [1,4,2,5,3,6]
  assert_true(np_eq(y, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]))
}
