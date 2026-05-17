// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Adam optimizer factory — uses fused batch kernel on GPU, in-place ops on CPU.
///
///   m ← β₁·m + (1-β₁)·g           (first moment / momentum)
///   v ← β₂·v + (1-β₂)·g⊙g         (second moment / uncentered variance)
///   p ← p - lr · m̂ / (√v̂ + ε)
///
/// where m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ) are bias-corrected estimates.
pub fn[T : @tensor.Tensor + BatchAdamAutoclip] adam(
  params : Array[@grad.Grad[T]],
  beta1 : Double,
  beta2 : Double,
  eps : Double,
  weight_decay : Double,
) -> Optimizer[T] {
  let m = params.map(fn(p : @grad.Grad[T]) -> T { T::zeros_like(p.value) })
  let v = params.map(fn(p : @grad.Grad[T]) -> T { T::zeros_like(p.value) })
  let mut t = 0
  let beta1_f = beta1
  let beta2_f = beta2
  let eps_f = Float::from_double(eps)
  let wd_f = Float::from_double(weight_decay)
  let one_minus_beta1 = Float::from_double(1.0 - beta1)
  let one_minus_beta2 = Float::from_double(1.0 - beta2)
  {
    params,
    _step: fn(params : Array[@grad.Grad[T]], lr : Float) -> Unit {
      t = t + 1
      let bias_correction1 = Float::from_double(
        1.0 / (1.0 - @math.pow(beta1_f, Double::from_int(t))),
      )
      let bias_correction2 = Float::from_double(
        1.0 / (1.0 - @math.pow(beta2_f, Double::from_int(t))),
      )
      let mut has_grad = false
      let n = params.length()
      let param_vals : Array[T] = Array::makei(n, fn(i) { params[i].value })
      let grad_vals : Array[T] = Array::makei(n, fn(i) {
        match params[i].grad {
          Some(Some(g)) => {
            has_grad = true
            g
          }
          Some(None) | None => m[i] // dummy, won't be used
        }
      })
      if has_grad == false {
        return
      }
      T::batch_adam_autoclip(param_vals, grad_vals, m, v, {
        lr,
        beta1: Float::from_double(beta1_f),
        beta2: Float::from_double(beta2_f),
        one_minus_beta1,
        one_minus_beta2,
        bias_correction1,
        bias_correction2,
        eps: eps_f,
        weight_decay: wd_f,
        max_grad_norm: 35.0,
      })
    },
  }
}
