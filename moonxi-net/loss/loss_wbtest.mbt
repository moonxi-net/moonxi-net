// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn float_near(a : Float, b : Float, eps : Float) -> Bool {
  let d = a - b
  d < eps && d > 0.0 - eps
}

///|
test "mse loss identical inputs" {
  let y = @nparray.NpArray::from1d([1.0, 2.0, 3.0])
  let result = mse(y, y)
  // sub = [0,0,0], square = [0,0,0], mean = 0.0
  guard result is { shape: _, data: [v] } else {
    fail("result should be scalar")
  }
  assert_true(float_near(v, 0.0, 0.000001))
}

///|
test "mse loss constant difference" {
  let y_hat = @nparray.NpArray::from1d([2.0, 3.0, 4.0])
  let y = @nparray.NpArray::from1d([1.0, 2.0, 3.0])
  let result = mse(y_hat, y)
  // sub = [1,1,1], square = [1,1,1], mean = 1.0
  guard result is { shape: _, data: [v] } else {
    fail("result should be scalar")
  }
  assert_true(float_near(v, 1.0, 0.000001))
}

///|
test "mse loss non-trivial" {
  let y_hat = @nparray.NpArray::from1d([3.0, 5.0])
  let y = @nparray.NpArray::from1d([1.0, 3.0])
  let result = mse(y_hat, y)
  // sub = [2,2], square = [4,4], mean = 4.0
  guard result is { shape: _, data: [v] } else {
    fail("result should be scalar")
  }
  assert_true(float_near(v, 4.0, 0.000001))
}

///|
test "mse loss single element" {
  let y_hat = @nparray.NpArray::from1d([5.0])
  let y = @nparray.NpArray::from1d([3.0])
  let result = mse(y_hat, y)
  // sub = [2], square = [4], mean = 4.0
  guard result is { shape: _, data: [v] } else {
    fail("result should be scalar")
  }
  assert_true(float_near(v, 4.0, 0.000001))
}
