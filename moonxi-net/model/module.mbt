// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Composable module wrapping a forward pass and parameter collection.
///
/// MoonBit's trait system cannot express a `Module` trait for generic structs
/// (trait methods can't constrain type params to match Self's type param),
/// so we use the same closure-based pattern as Optimizer[T] and Trainer[T].
///
/// Usage:
///   let mod = conv.as_module(stride=1, padding=1)
///   let out = (mod.forward)(input, true)
///   let params = (mod.trainable_params)()
pub(all) struct Module[T] {
  /// Forward pass: (input, is_training) -> output
  forward : (T, Bool) -> T
  /// Collect trainable parameters (excludes running stats like BN running_mean/var)
  trainable_params : () -> Array[T]
}

///|
/// Chain modules sequentially. All parameters are collected in order.
pub fn[T] sequential(modules : Array[Module[T]]) -> Module[T] {
  {
    forward: fn(input : T, is_training : Bool) -> T {
      let mut x = input
      for m in modules {
        x = (m.forward)(x, is_training)
      }
      x
    },
    trainable_params: fn() -> Array[T] {
      let result : Array[T] = []
      for m in modules {
        for p in (m.trainable_params)() {
          result.push(p)
        }
      }
      result
    },
  }
}

///|
/// Create a parameter-free module from a simple function (e.g., ReLU, pooling).
pub fn[T] fn_module(f : (T) -> T) -> Module[T] {
  {
    forward: fn(x : T, _is_training : Bool) -> T { f(x) },
    trainable_params: fn() -> Array[T] { [] },
  }
}

// ── Layer lift methods: Conv2d → Module ──

///|
/// Lift a Conv2d into a Module with fixed stride/padding.
pub fn[T : @tensor.ImageTensor] Conv2d::as_module(
  self : Conv2d[T],
  stride? : Int = 1,
  padding? : Int = 0,
) -> Module[T] {
  {
    forward: fn(input : T, _is_training : Bool) -> T {
      input.conv2d(self.weight, self.bias, stride, padding)
    },
    trainable_params: fn() -> Array[T] { [self.weight, self.bias] },
  }
}

///|
/// Lift a BatchNorm2d into a Module. Uses is_training to switch between
/// training (updates running stats) and inference (uses running stats).
pub fn[T : @tensor.ImageTensor] BatchNorm2d::as_module(
  self : BatchNorm2d[T],
  momentum? : Float = 0.1,
  eps? : Float = 0.00001,
) -> Module[T] {
  {
    forward: fn(input : T, is_training : Bool) -> T {
      if is_training {
        input.batchnorm_training(
          self.gamma,
          self.beta,
          self.running_mean,
          self.running_var,
          momentum,
          eps,
        ).0
      } else {
        input.batchnorm_inference(
          self.gamma,
          self.beta,
          self.running_mean,
          self.running_var,
          eps,
        )
      }
    },
    trainable_params: fn() -> Array[T] { [self.gamma, self.beta] },
  }
}

///|
/// Lift a Linear layer into a Module.
pub fn[T : @tensor.Tensor + @tensor.BlasTensor] Linear::as_module(
  self : Linear[T],
) -> Module[T] {
  {
    forward: fn(input : T, _is_training : Bool) -> T {
      input.matmul(self.w).add(self.b)
    },
    trainable_params: fn() -> Array[T] { [self.w, self.b] },
  }
}

///|
/// Lift ResNet-18 into a Module.
pub fn[T : @tensor.Tensor + @tensor.BlasTensor + @tensor.ImageTensor] ResNet18::as_module(
  self : ResNet18[T],
) -> Module[T] {
  {
    forward: fn(input : T, is_training : Bool) -> T {
      self.forward(input, is_training~)
    },
    trainable_params: fn() -> Array[T] { self.params() },
  }
}
