// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn float_near(a : Float, b : Float, eps : Float) -> Bool {
  let d = a - b
  d < eps && d > 0.0 - eps
}

///|
test "Linear forward: x @ w + b" {
  // w shape [2, 3]: maps input dim 2 -> output dim 3 (Linear::forward = x @ w + b)
  let w = @nparray.NpArray::from2d([
    [1.0, 0.0, 1.0], // col 0 of output
    [0.0, 1.0, 1.0], // col 1 of output
  ])
  // b shape [3]
  let b = @nparray.NpArray::from1d([0.0, 0.0, 0.0])
  let linear : Linear[@nparray.NpArray] = { w, b }
  // input shape [1, 2] (batch=1, features=2)
  let x = @nparray.NpArray::from2d([[3.0, 5.0]])
  let out = linear.forward(x)
  // x @ w = [[3,5]] @ [[1,0,1],[0,1,1]] = [[3*1+5*0, 3*0+5*1, 3*1+5*1]] = [[3, 5, 8]]
  assert_eq(out.dims()[0], 1)
  assert_eq(out.dims()[1], 3)
  assert_true(float_near(out.data[0], 3.0, 0.0001))
  assert_true(float_near(out.data[1], 5.0, 0.0001))
  assert_true(float_near(out.data[2], 8.0, 0.0001))
}

///|
test "Linear forward: with non-zero bias" {
  // w shape [2, 2]: input dim 2 -> output dim 2
  let w = @nparray.NpArray::from2d([[2.0, 4.0], [3.0, 5.0]])
  let b = @nparray.NpArray::from1d([10.0, 20.0])
  let linear : Linear[@nparray.NpArray] = { w, b }
  let x = @nparray.NpArray::from2d([[1.0, 1.0]])
  let out = linear.forward(x)
  // x @ w = [[1,1]] @ [[2,4],[3,5]] = [[2+3, 4+5]] = [[5, 9]]
  // + b = [10, 20] broadcast => [[15, 29]]
  assert_eq(out.data.length(), 2)
  assert_true(float_near(out.data[0], 15.0, 0.0001))
  assert_true(float_near(out.data[1], 29.0, 0.0001))
}

///|
test "Linear::as_module returns trainable_params [w, b]" {
  // w shape [2, 2]
  let w = @nparray.NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let b = @nparray.NpArray::from1d([0.5, 0.5])
  let linear : Linear[@nparray.NpArray] = { w, b }
  let mod_ = linear.as_module()
  let params = (mod_.trainable_params)()
  assert_eq(params.length(), 2)
  // First param should be w (shape [2, 2])
  assert_eq(params[0].dims()[0], 2)
  assert_eq(params[0].dims()[1], 2)
  // Second param should be b (shape [2])
  assert_eq(params[1].dims()[0], 2)
}

///|
test "Module::sequential chains two modules and collects params" {
  // First linear: input dim 2 -> output dim 3 (w shape [2, 3])
  let w1 = @nparray.NpArray::from2d([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
  let b1 = @nparray.NpArray::from1d([0.0, 0.0, 0.0])
  let l1 : Linear[@nparray.NpArray] = { w: w1, b: b1 }
  let m1 = l1.as_module()

  // Second linear: input dim 3 -> output dim 2 (w shape [3, 2])
  let w2 = @nparray.NpArray::from2d([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
  let b2 = @nparray.NpArray::from1d([0.0, 0.0])
  let l2 : Linear[@nparray.NpArray] = { w: w2, b: b2 }
  let m2 = l2.as_module()

  let seq = sequential([m1, m2])

  // Check params: should be [w1, b1, w2, b2]
  let params = (seq.trainable_params)()
  assert_eq(params.length(), 4)
  assert_eq(params[0].dims()[0], 2) // w1 shape [2,3]
  assert_eq(params[0].dims()[1], 3)
  assert_eq(params[1].dims()[0], 3) // b1 shape [3]
  assert_eq(params[2].dims()[0], 3) // w2 shape [3,2]
  assert_eq(params[2].dims()[1], 2)
  assert_eq(params[3].dims()[0], 2) // b2 shape [2]

  // Check forward: input [1,2] -> l1 -> [1,3] -> l2 -> [1,2]
  let x = @nparray.NpArray::from2d([[1.0, 2.0]])
  let out = (seq.forward)(x, false)
  // l1: [1,2] @ [[1,0,1],[0,1,1]] = [1+0, 0+2, 1+2] = [1,2,3]
  // l2: [1,2,3] @ [[1,0],[0,1],[0,0]] = [1,2]
  assert_eq(out.dims()[0], 1)
  assert_eq(out.dims()[1], 2)
  assert_true(float_near(out.data[0], 1.0, 0.0001))
  assert_true(float_near(out.data[1], 2.0, 0.0001))
}

///|
test "fn_module creates parameter-free module" {
  let relu_mod : Module[@nparray.NpArray] = fn_module(fn(x) { x.relu() })
  let params = (relu_mod.trainable_params)()
  assert_eq(params.length(), 0)
}
