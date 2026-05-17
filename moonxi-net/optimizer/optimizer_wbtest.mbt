// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn float_near(a : Float, b : Float, eps : Float) -> Bool {
  let d = a - b
  d < eps && d > 0.0 - eps
}

///|
test "StepLR scheduler lr decay" {
  let scheduler = StepLR::new(0.1, 3, 0.1)
  // Steps 1-2: no decay (step_count 1,2 not divisible by step_size=3)
  assert_true(float_near(scheduler.step(), 0.1, 0.000001))
  assert_true(float_near(scheduler.step(), 0.1, 0.000001))
  // Step 3: decay (3%3==0), lr = 0.1 * 0.1 = 0.01
  assert_true(float_near(scheduler.step(), 0.01, 0.000001))
  // Steps 4-5: no decay
  assert_true(float_near(scheduler.step(), 0.01, 0.000001))
  assert_true(float_near(scheduler.step(), 0.01, 0.000001))
  // Step 6: decay again, lr = 0.01 * 0.1 = 0.001
  assert_true(float_near(scheduler.step(), 0.001, 0.000001))
  // get_lr should match last step
  assert_true(float_near(scheduler.get_lr(), 0.001, 0.000001))
}

///|
test "StepLR scheduler gamma=0.5" {
  let scheduler = StepLR::new(1.0, 2, 0.5)
  // Step 1: no decay
  assert_true(float_near(scheduler.step(), 1.0, 0.000001))
  // Step 2: decay, lr = 1.0 * 0.5 = 0.5
  assert_true(float_near(scheduler.step(), 0.5, 0.000001))
  // Step 3: no decay
  assert_true(float_near(scheduler.step(), 0.5, 0.000001))
  // Step 4: decay, lr = 0.5 * 0.5 = 0.25
  assert_true(float_near(scheduler.step(), 0.25, 0.000001))
}

///|
test "zero_grad clears gradients" {
  @grad.clear_tape()
  let param = @grad.Grad::init(@nparray.NpArray::from1d([1.0, 2.0, 3.0]))
  param.grad = Some(Some(@nparray.NpArray::from1d([0.5, 0.5, 0.5])))
  let optim = momentum_sgd([param], 0.9, 0.0001, 35.0)
  assert_true(param.grad is Some(Some(_)))
  optim.zero_grad()
  assert_true(param.grad is Some(None))
}

///|
test "zero_grad with multiple params" {
  @grad.clear_tape()
  let p1 = @grad.Grad::init(@nparray.NpArray::from1d([1.0]))
  p1.grad = Some(Some(@nparray.NpArray::from1d([1.0])))
  let p2 = @grad.Grad::init(@nparray.NpArray::from1d([2.0]))
  p2.grad = Some(Some(@nparray.NpArray::from1d([2.0])))
  let optim = momentum_sgd([p1, p2], 0.9, 0.0001, 35.0)
  optim.zero_grad()
  assert_true(p1.grad is Some(None))
  assert_true(p2.grad is Some(None))
}
