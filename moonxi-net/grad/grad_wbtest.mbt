// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
test "grad add backward" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  x.grad = Some(None)
  let y = Grad::init(@nparray.NpArray::from1d([4.0, 5.0, 6.0]))
  y.grad = Some(None)
  let z = x.add(y)
  let loss = z.mean()
  loss.backward()
  // d(x)/d(add) = 1, d(mean)/d(add) = 1/N
  // grad(x) = 1/N = [1/3, 1/3, 1/3]
  let expected = @nparray.NpArray::scale(
    @nparray.NpArray::from1d([1.0, 1.0, 1.0]),
    1.0 / 3.0,
  )
  guard x.grad is Some(Some({ shape: _, data: [dg0, dg1, dg2] })) else {
    fail("x.grad mismatch")
  }
  let exp_arr = expected.data
  assert_true(dg0 == exp_arr[0] && dg1 == exp_arr[1] && dg2 == exp_arr[2])
}

///|
test "grad sub backward" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  x.grad = Some(None)
  let y = Grad::init(@nparray.NpArray::from1d([4.0, 5.0, 6.0]))
  y.grad = Some(None)
  let z = x.sub(y)
  let loss = z.mean()
  loss.backward()
  // d(x)/d(sub) = 1, d(y)/d(sub) = -1
  // grad(x) = 1/N, grad(y) = -1/N
  let expected_x = @nparray.NpArray::scale(
    @nparray.NpArray::from1d([1.0, 1.0, 1.0]),
    1.0 / 3.0,
  )
  let expected_y = @nparray.NpArray::scale(
    @nparray.NpArray::from1d([1.0, 1.0, 1.0]),
    -1.0 / 3.0,
  )
  guard x.grad is Some(Some({ shape: _, data: [gx0, gx1, gx2] })) else {
    fail("x.grad mismatch")
  }
  guard y.grad is Some(Some({ shape: _, data: [gy0, gy1, gy2] })) else {
    fail("y.grad mismatch")
  }
  let ex = expected_x.data
  let ey = expected_y.data
  assert_true(gx0 == ex[0] && gx1 == ex[1] && gx2 == ex[2])
  assert_true(gy0 == ey[0] && gy1 == ey[1] && gy2 == ey[2])
}

///|
test "grad square backward" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  x.grad = Some(None)
  let y = x.square()
  let loss = y.mean()
  loss.backward()
  // y = x^2, dy/dx = 2x
  // d(loss)/d(x) = d(loss)/d(y) * dy/dx = (1/N) * 2x = 2x/N
  // x = [1,2,3], N=3, so grad = 2*[1,2,3]/3 = [2/3, 4/3, 2]
  let expected = @nparray.NpArray::scale(
    @nparray.NpArray::from1d([1.0, 2.0, 3.0]),
    2.0 / 3.0,
  )
  guard x.grad is Some(Some({ shape: _, data: [g0, g1, g2] })) else {
    fail("x.grad mismatch")
  }
  let exp = expected.data
  assert_true(g0 == exp[0] && g1 == exp[1] && g2 == exp[2])
}

///|
test "grad chain rule" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  x.grad = Some(None)
  let y = x.square()
  let loss = y.mean()
  loss.backward()
  // y = x^2, loss = mean(y)
  // d(loss)/d(x) = d(loss)/d(y) * dy/dx = (1/N) * 2x = 2x/N
  // x = [1,2,3], N=3, so grad = 2*[1,2,3]/3 = [2/3, 4/3, 2]
  let expected = @nparray.NpArray::scale(
    @nparray.NpArray::from1d([1.0, 2.0, 3.0]),
    2.0 / 3.0,
  )
  guard x.grad is Some(Some({ shape: _, data: [g0, g1, g2] })) else {
    fail("x.grad mismatch")
  }
  let exp = expected.data
  assert_true(g0 == exp[0] && g1 == exp[1] && g2 == exp[2])
}

///|
test "grad matmul backward" {
  clear_tape()
  let a = Grad::init(@nparray.NpArray::from2d([[1.0, 2.0], [3.0, 4.0]]))
  a.grad = Some(None)
  let b = Grad::init(@nparray.NpArray::from2d([[5.0, 6.0], [7.0, 8.0]]))
  b.grad = Some(None)
  let c = a.matmul(b)
  let loss = c.mean()
  loss.backward()
  // c = a @ b, loss = mean(c), d(loss)/d(c) = [[0.25,0.25],[0.25,0.25]]
  // d(loss)/d(a) = d(loss)/d(c) @ b^T = [[0.25,0.25],[0.25,0.25]] @ [[5,7],[6,8]]
  //   = [[2.75, 3.75], [2.75, 3.75]]
  // d(loss)/d(b) = a^T @ d(loss)/d(c) = [[1,3],[2,4]] @ [[0.25,0.25],[0.25,0.25]]
  //   = [[1.0, 1.0], [1.5, 1.5]]
  let expected_a_grad = @nparray.NpArray::from2d([[2.75, 3.75], [2.75, 3.75]])
  let expected_b_grad = @nparray.NpArray::from2d([[1.0, 1.0], [1.5, 1.5]])
  guard a.grad is Some(Some({ shape: _, data: ga })) else {
    fail("a.grad mismatch")
  }
  guard b.grad is Some(Some({ shape: _, data: gb })) else {
    fail("b.grad mismatch")
  }
  let ega = expected_a_grad.data
  let egb = expected_b_grad.data
  assert_true(
    ga[0] == ega[0] && ga[1] == ega[1] && ga[2] == ega[2] && ga[3] == ega[3],
  )
  assert_true(
    gb[0] == egb[0] && gb[1] == egb[1] && gb[2] == egb[2] && gb[3] == egb[3],
  )
}

///|
test "grad scale backward" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  x.grad = Some(None)
  let y = x.scale(2.0)
  let loss = y.mean()
  loss.backward()
  // y = 2*x, loss = mean(y)
  // d(loss)/d(x) = d(loss)/d(y) * dy/dx = (1/N) * 2 = 2/N
  let expected = @nparray.NpArray::scale(
    @nparray.NpArray::from1d([1.0, 1.0, 1.0]),
    2.0 / 3.0,
  )
  guard x.grad is Some(Some({ shape: _, data: [g0, g1, g2] })) else {
    fail("x.grad mismatch")
  }
  let exp = expected.data
  assert_true(g0 == exp[0] && g1 == exp[1] && g2 == exp[2])
}

///|
test "grad mul_elem backward" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  x.grad = Some(None)
  let y = Grad::init(@nparray.NpArray::from1d([4.0, 5.0, 6.0]))
  y.grad = Some(None)
  let z = x.mul_elem(y)
  let loss = z.mean()
  loss.backward()
  // z = x * y, loss = mean(z)
  // d(loss)/d(x) = d(loss)/d(z) * dz/dx = (1/N) * y
  // d(loss)/d(y) = (1/N) * x
  let n : Float = 3.0
  let expected_x = @nparray.NpArray::scale(y.value, 1.0 / n)
  let expected_y = @nparray.NpArray::scale(x.value, 1.0 / n)
  guard x.grad is Some(Some({ shape: _, data: [gx0, gx1, gx2] })) else {
    fail("x.grad mismatch")
  }
  guard y.grad is Some(Some({ shape: _, data: [gy0, gy1, gy2] })) else {
    fail("y.grad mismatch")
  }
  let ex = expected_x.data
  let ey = expected_y.data
  assert_true(gx0 == ex[0] && gx1 == ex[1] && gx2 == ex[2])
  assert_true(gy0 == ey[0] && gy1 == ey[1] && gy2 == ey[2])
}

///|
test "grad init and value" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  guard x.value is { shape: _, data: [v0, v1, v2] } else {
    fail("value mismatch")
  }
  assert_true(v0 == 1.0 && v1 == 2.0 && v2 == 3.0)
  // After init, grad should be None (not tracking)
  assert_true(x.grad is None)
}

///|
test "grad clear tape" {
  clear_tape()
  let x = Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  x.grad = Some(None)
  let _y = x.square()
  clear_tape()
  // Tape is cleared, backward should not affect x.grad
  _y.backward()
  // grad should still be None (pending, never written)
  assert_true(x.grad is Some(None))
}
