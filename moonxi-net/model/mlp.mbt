// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// MLP (Multi-Layer Perceptron) for digit classification.
/// Architecture: 784 → 128 → ReLU → 64 → ReLU → 10
pub(all) struct MLP[T] {
  mut fc1 : Linear[T] // 784 → 128
  mut fc2 : Linear[T] // 128 → 64
  mut fc3 : Linear[T] // 64  → 10
}

///|
/// MLP forward pass: fc1 → relu → fc2 → relu → fc3
/// Input shape: [batch, 784], Output shape: [batch, 10]
pub fn[T : @tensor.Tensor + @tensor.BlasTensor + @tensor.ImageTensor] MLP::forward(
  self : MLP[T],
  input : T,
) -> T {
  let x = input.matmul(self.fc1.w.transpose()).add(self.fc1.b).relu()
  let x = x.matmul(self.fc2.w.transpose()).add(self.fc2.b).relu()
  x.matmul(self.fc3.w.transpose()).add(self.fc3.b)
}

///|
/// Collect all trainable parameters from MLP.
pub fn[T] MLP::params(self : MLP[T]) -> Array[T] {
  [self.fc1.w, self.fc1.b, self.fc2.w, self.fc2.b, self.fc3.w, self.fc3.b]
}

///|
/// Initialize MLP with Xavier/Glorot uniform initialization.
pub fn[T : @tensor.Tensor] init_mlp() -> MLP[T] {
  fn xavier_init(fan_in : Int, fan_out : Int) -> T {
    let std = Float::sqrt(
      Float::from_double(2.0) / Float::from_int(fan_in + fan_out),
    )
    let n = fan_out * fan_in
    let data = FixedArray::makei(n, fn(_i) { _box_muller() * std })
    T::from_host(data, [fan_out, fan_in])
  }

  MLP::{
    fc1: Linear::{ w: xavier_init(784, 128), b: T::zeros([128]) },
    fc2: Linear::{ w: xavier_init(128, 64), b: T::zeros([64]) },
    fc3: Linear::{ w: xavier_init(64, 10), b: T::zeros([10]) },
  }
}

///|
/// Box-Muller transform for generating normally-distributed random floats.
fn _box_muller() -> Float {
  let u1 = Float::from_double(_mlp_rng.double())
  let u2 = Float::from_double(_mlp_rng.double())
  Float::sqrt(
    -Float::from_double(2.0) * Float::from_double(@math.ln(u1.to_double())),
  ) *
  Float::from_double(@math.cos(2.0 * @math.PI * u2.to_double()))
}

///|
/// Lift MLP into a Module[T] for use with Experiment::new().
pub fn[T : @tensor.Tensor + @tensor.BlasTensor + @tensor.ImageTensor] MLP::as_module(
  self : MLP[T],
) -> Module[T] {
  {
    forward: fn(input : T, _is_training : Bool) -> T { self.forward(input) },
    trainable_params: fn() -> Array[T] { self.params() },
  }
}

///|
let _mlp_rng : @random.Rand = @random.Rand::chacha8()
