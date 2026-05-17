// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// A tensor wrapper for automatic differentiation via reverse-mode autograd (tape-based).
///
/// # Type parameter
/// - `T` — the underlying tensor type (e.g. `NpArray`, `GpuTensor`).
///   Must implement `@tensor.Tensor`.
///
/// # Fields
/// - `value` — the forward-pass tensor value.
/// - `grad` — the gradient, using a **three-state option** (`T??`):
///   - `None` — **no tracking**: gradient computation is disabled for this node.
///     Used for constant data (inputs, labels) created via `no_grad()`.
///   - `Some(None)` — **pending**: this node requires a gradient, but it has not
///     been computed yet. Set by `zero_grad()` before each forward pass.
///   - `Some(Some(g))` — **computed**: the gradient `g` has been materialized.
///     During backward, the first write assigns directly; subsequent writes
///     accumulate via `add_into` (for nodes with multiple consumers).
///
/// # Lifecycle per training step
/// 1. `zero_grad()` / manual reset → `grad = Some(None)` for all parameters.
/// 2. Forward pass → operations build up the tape via `record_accumulate()`.
/// 3. `backward()` on the loss → tape replays in reverse, filling `grad = Some(Some(g))`.
/// 4. Optimizer reads `grad` to update `value`.
pub(all) struct Grad[T] {
  mut value : T
  mut grad : T??
}

///|
let tape : Array[() -> Unit] = Array::new()

///|
pub fn clear_tape() -> Unit {
  tape.clear()
}

///|
pub fn tape_length() -> Int {
  tape.length()
}

///|
/// Wrap a raw tensor as a Grad with no gradient tracking.
pub fn[T] no_grad(t : T) -> Grad[T] {
  { value: t, grad: None }
}

///|
pub fn[T] Grad::init(v : T) -> Grad[T] {
  no_grad(v)
}

///|
/// Combine two Grad values for a binary operation.
/// Marks grad as pending if either input has gradient tracking.
fn[T] Grad::join(x : Grad[T], y : Grad[T], f : (T, T) -> T) -> Grad[T] {
  let result = f(x.value, y.value)
  let grad = if x.grad is Some(_) || y.grad is Some(_) {
    Some(None)
  } else {
    None
  }
  { value: result, grad }
}

///|
/// Record a backward closure on the tape using lazy grad allocation.
/// grad_fn receives the output gradient and returns the transformed input gradient.
/// First write assigns directly; subsequent writes accumulate via add_into.
pub fn[T : @tensor.Tensor] record_accumulate(
  input : Grad[T],
  output : Grad[T],
  grad_fn : (T) -> T,
) -> Unit {
  if output.grad is Some(_) {
    tape.push(fn() -> Unit {
      match (input.grad, output.grad) {
        (Some(None), Some(Some(g))) => input.grad = Some(Some(grad_fn(g)))
        (Some(Some(acc)), Some(Some(g))) => acc.add_into(grad_fn(g))
        _ => ()
      }
    })
  }
}

///|
pub fn[T : @tensor.Tensor] Grad::backward(self : Grad[T]) -> Unit {
  self.grad = Some(Some(T::scalar((1.0 : Float))))
  let n = tape.length()
  for i in 0..<n {
    tape[n - 1 - i]()
  }
}

///|
/// Execute backward with per-closure timing.
/// timer_fn returns current wall-clock time in milliseconds.
/// Returns array of per-closure execution times in milliseconds,
/// indexed by execution order (last closure first = index 0).
pub fn[T : @tensor.Tensor] Grad::backward_profiled(
  self : Grad[T],
  timer_fn : () -> Double,
) -> Array[Double] {
  self.grad = Some(Some(T::scalar((1.0 : Float))))
  let n = tape.length()
  let times : Array[Double] = Array::new()
  for i in 0..<n {
    let t0 = timer_fn()
    tape[n - 1 - i]()
    let t1 = timer_fn()
    times.push(t1 - t0)
  }
  times
}

///|
pub fn[T] Grad::value(self : Grad[T]) -> T {
  self.value
}

///|
pub fn[T] Grad::grad(self : Grad[T]) -> T?? {
  self.grad
}

// =============================================================================
// Tensor impl for Grad[T]
// =============================================================================

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with dims(x : Grad[T]) -> FixedArray[
  Int,
] {
  x.value.dims()
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with zeros(
  dims : FixedArray[Int],
) -> Grad[T] {
  no_grad(T::zeros(dims))
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with zeros_like(
  x : Grad[T],
) -> Grad[T] {
  no_grad(T::zeros_like(x.value))
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with from_host(
  data : FixedArray[Float],
  shape : Array[Int],
) -> Grad[T] {
  no_grad(T::from_host(data, shape))
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with add(
  x : Grad[T],
  y : Grad[T],
) -> Grad[T] {
  let out = x.join(y, (a, b) => a.add(b))
  let x_dim = x.dims()
  record_accumulate(x, out, g => g.reduce_sum_to(x_dim))
  let y_dim = y.dims()
  record_accumulate(y, out, g => g.reduce_sum_to(y_dim))
  out
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with sub(
  x : Grad[T],
  y : Grad[T],
) -> Grad[T] {
  let out = x.join(y, (a, b) => a.sub(b))
  let x_dim = x.dims()
  record_accumulate(x, out, g => g.reduce_sum_to(x_dim))
  let y_dim = y.dims()
  record_accumulate(y, out, g => {
    g.scale((0.0 : Float) - (1.0 : Float)).reduce_sum_to(y_dim)
  })
  out
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with square(x : Grad[T]) -> Grad[
  T,
] {
  let result = T::square(x.value)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_x = x.value
  record_accumulate(x, g, og => saved_x.scale((2.0 : Float)).mul_elem(og))
  g
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with mean(x : Grad[T]) -> Grad[
  T,
] {
  let result = T::mean(x.value)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let n = T::size(x.value)
  let target_dims = x.value.dims()
  record_accumulate(x, g, og => {
    og.scale((1.0 : Float) / Float::from_int(n)).broadcast_to(target_dims)
  })
  g
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with scale(
  x : Grad[T],
  s : Float,
) -> Grad[T] {
  let result = T::scale(x.value, s)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  record_accumulate(x, g, og => og.scale(s))
  g
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with mul_elem(
  x : Grad[T],
  y : Grad[T],
) -> Grad[T] {
  let out = x.join(y, (a, b) => a.mul_elem(b))
  let saved_y = y.value
  record_accumulate(x, out, og => og.mul_elem(saved_y))
  let saved_x = x.value
  record_accumulate(y, out, og => og.mul_elem(saved_x))
  out
}

///|
/// sqrt(x) — forward: √x, backward: og / (2√x)
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with sqrt(x : Grad[T]) -> Grad[
  T,
] {
  let result = T::sqrt(x.value)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_result = result
  record_accumulate(x, g, og => og.div_elem(saved_result.scale((2.0 : Float))))
  g
}

///|
/// div_elem(x, y) — forward: x/y, backward: 1/y for x, -x/y² for y
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with div_elem(
  x : Grad[T],
  y : Grad[T],
) -> Grad[T] {
  let out = x.join(y, (a, b) => a.div_elem(b))
  let saved_y = y.value
  record_accumulate(x, out, og => og.div_elem(saved_y))
  let saved_x = x.value
  record_accumulate(y, out, og => {
    og
    .mul_elem(saved_x)
    .div_elem(saved_y.mul_elem(saved_y))
    .scale((-1.0 : Float))
  })
  out
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with add_into(
  x : Grad[T],
  y : Grad[T],
) -> Unit {
  x.value.add_into(y.value)
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with scalar(v : Float) -> Grad[
  T,
] {
  no_grad(T::scalar(v))
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with size(x : Grad[T]) -> Int {
  T::size(x.value)
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with reduce_sum_to(
  x : Grad[T],
  target_dims : FixedArray[Int],
) -> Grad[T] {
  no_grad(T::reduce_sum_to(x.value, target_dims))
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with value(x : Grad[T]) -> @tensor.TensorData {
  x.value.value()
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with view(
  x : Grad[T],
  new_shape : FixedArray[Int],
) -> Grad[T] {
  let result = T::view(x.value, new_shape)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let target_dims = x.value.dims()
  record_accumulate(x, g, og => T::view(og, target_dims))
  g
}

///|
pub impl[T : @tensor.Tensor] @tensor.Tensor for Grad[T] with broadcast_to(
  x : Grad[T],
  target_shape : FixedArray[Int],
) -> Grad[T] {
  no_grad(T::broadcast_to(x.value, target_shape))
}

// =============================================================================
// BlasTensor impl for Grad[T]
// =============================================================================
