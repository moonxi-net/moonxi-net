// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// RMSprop optimizer factory — uses fused batch kernel on GPU, in-place ops on CPU.
///
///   v ← α·v + (1-α)·g²    (exponential moving average of squared gradients)
///   p ← p - lr · g / (√v + ε)
///
/// With momentum: buf ← μ·buf + g/(√v+ε), then p ← p - lr·buf.
pub fn[T : @tensor.Tensor + BatchRmspropAutoclip] rmsprop(
  params : Array[@grad.Grad[T]],
  alpha : Double,
  eps : Double,
  weight_decay : Double,
  momentum : Double,
) -> Optimizer[T] {
  let v = params.map(fn(p : @grad.Grad[T]) -> T { T::zeros_like(p.value) })
  let buf = params.map(fn(p : @grad.Grad[T]) -> T { T::zeros_like(p.value) })
  let alpha_f = Float::from_double(alpha)
  let one_minus_alpha = Float::from_double(1.0 - alpha)
  let eps_f = Float::from_double(eps)
  let wd_f = Float::from_double(weight_decay)
  let momentum_f = Float::from_double(momentum)
  let has_momentum = momentum > 0.0
  {
    params,
    _step: fn(params : Array[@grad.Grad[T]], lr : Float) -> Unit {
      let mut has_grad = false
      let n = params.length()
      let param_vals : Array[T] = Array::makei(n, fn(i) { params[i].value })
      let grad_vals : Array[T] = Array::makei(n, fn(i) {
        match params[i].grad {
          Some(Some(g)) => {
            has_grad = true
            g
          }
          Some(None) | None => v[i] // dummy
        }
      })
      if has_grad == false {
        return
      }
      T::batch_rmsprop_autoclip(param_vals, grad_vals, v, buf, {
        lr,
        alpha: alpha_f,
        one_minus_alpha,
        eps: eps_f,
        weight_decay: wd_f,
        momentum: momentum_f,
        has_momentum,
        max_grad_norm: 35.0,
      })
    },
  }
}
