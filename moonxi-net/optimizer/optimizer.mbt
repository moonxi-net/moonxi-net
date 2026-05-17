// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Generic Optimizer. Closures capture optimizer-specific state and step logic.
///   T — tensor type (e.g. GpuTensor)
///
/// Each optimizer factory (momentum_sgd, adam, etc.) creates an Optimizer[T]
/// whose _step closure captures all optimizer-specific state internally.
/// LR is passed explicitly to step(lr) — the Optimizer does not hold mutable LR.
///
/// TODO: Future state management API
///   load_state : (Optimizer[T], Map[String, Array[T]]) -> Unit
///   store_state : (Optimizer[T]) -> Map[String, Array[T]]
/// Each optimizer will expose its own state dict for checkpoint/restore.
pub(all) struct Optimizer[T] {
  params : Array[@grad.Grad[T]]
  _step : (Array[@grad.Grad[T]], Float) -> Unit
}

///|
/// Perform one optimization step with the given learning rate.
pub fn[T] Optimizer::step(self : Optimizer[T], lr : Float) -> Unit {
  (self._step)(self.params, lr)
}

///|
/// Clear tape and reset all parameter gradients to pending state.
/// Call before forward pass in each training iteration.
pub fn[T] Optimizer::zero_grad(self : Optimizer[T]) -> Unit {
  @grad.clear_tape()
  for p in self.params {
    p.grad = Some(None)
  }
}

///|
/// Build optimizer from name string and model params.
/// Returns (optimizer_factory, default_lr) — each optimizer carries its
/// recommended learning rate so users don't need to guess.
/// Supported: momentum_sgd, adam, rmsprop
pub fn[
  T : @tensor.Tensor + BatchSgdMomentumAutoclip + BatchAdamAutoclip + BatchRmspropAutoclip,
] make_optimizer(
  name : String,
) -> ((Array[@grad.Grad[T]]) -> Optimizer[T], Double) raise {
  match name {
    "momentum_sgd" =>
      (fn(params) { momentum_sgd(params, 0.9, 0.0001, 35.0) }, 0.01)
    "adam" =>
      (fn(params) { adam(params, 0.9, 0.999, 0.00000001, 0.0001) }, 0.001)
    "rmsprop" =>
      (fn(params) { rmsprop(params, 0.99, 0.00000001, 0.0001, 0.9) }, 0.001)
    _ =>
      raise @tensor.UnsupportedError(
        "unknown optimizer: '\{name}'. Supported: momentum_sgd, adam, rmsprop",
      )
  }
}
