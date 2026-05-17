// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub impl[T : @tensor.Tensor + @tensor.BlasTensor] @tensor.BlasTensor for Grad[T] with transpose(
  x : Grad[T],
) -> Grad[T] {
  let result = T::transpose(x.value)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  record_accumulate(x, g, og => T::transpose(og))
  g
}

///|
pub impl[T : @tensor.Tensor + @tensor.BlasTensor] @tensor.BlasTensor for Grad[T] with matmul(
  x : Grad[T],
  y : Grad[T],
) -> Grad[T] {
  let result = T::matmul(x.value, y.value)
  let grad = if x.grad is Some(_) || y.grad is Some(_) {
    Some(None)
  } else {
    None
  }
  let out : Grad[T] = { value: result, grad }
  let saved_y = y.value
  record_accumulate(x, out, og => og.matmul(T::transpose(saved_y)))
  let saved_x = x.value
  record_accumulate(y, out, og => T::transpose(saved_x).matmul(og))
  out
}
