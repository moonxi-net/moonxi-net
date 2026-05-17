// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Step learning rate scheduler.
/// Decays the learning rate by gamma every step_size epochs.
/// Mirrors PyTorch's torch.optim.lr_scheduler.StepLR.
///
/// Example:
///   let scheduler = StepLR::new(0.1, step_size=30, gamma=0.1)
///   for epoch in 0..<100 {
///     train_one_epoch(...)
///     optim.lr = scheduler.step()  // lr *= 0.1 every 30 epochs
///   }
pub(all) struct StepLR {
  step_size : Int
  gamma : Float
  mut step_count : Int
  mut lr : Float
}

///|
pub fn StepLR::new(
  initial_lr : Double,
  step_size : Int,
  gamma : Double,
) -> StepLR {
  {
    step_size,
    gamma: Float::from_double(gamma),
    step_count: 0,
    lr: Float::from_double(initial_lr),
  }
}

///|
/// Advance the scheduler by one step (call once per epoch).
/// Returns the updated learning rate.
pub fn StepLR::step(self : StepLR) -> Float {
  self.step_count = self.step_count + 1
  if self.step_count % self.step_size == 0 {
    self.lr = self.lr * self.gamma
  }
  self.lr
}

///|
/// Get the current learning rate without advancing.
pub fn StepLR::get_lr(self : StepLR) -> Float {
  self.lr
}
