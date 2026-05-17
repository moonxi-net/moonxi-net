// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// =============================================================================
// Experiment[T] — researcher's experiment config + training methods
// =============================================================================

///|
/// Experiment config: what the researcher defines once.
///   forward:        model forward pass (input → output, in Grad wrapper)
///   loss:           loss computation (prediction, target → loss)
///   optim:          optimizer (holds params, step/zero_grad)
///   lr_scheduler:   learning rate scheduler (e.g. StepLR)
pub(all) struct Experiment[T] {
  forward : (@grad.Grad[T], Bool) -> @grad.Grad[T]
  loss : (@grad.Grad[T], @grad.Grad[T]) -> @grad.Grad[T]
  optim : @optimizer.Optimizer[T]
  lr_scheduler : @optimizer.StepLR
}

///|
/// Build Experiment from a Module, LossFn, optimizer factory, and scheduler.
/// Extracts forward closure and trainable params from the module internally,
/// so the caller never writes lambdas or references params separately.
pub fn[T] Experiment::new(
  model~ : @model.Module[@grad.Grad[T]],
  loss~ : @loss.LossFn[@grad.Grad[T]],
  optim~ : (Array[@grad.Grad[T]]) -> @optimizer.Optimizer[T],
  scheduler~ : @optimizer.StepLR,
) -> Experiment[T] {
  let optim = optim((model.trainable_params)())
  { forward: model.forward, loss: loss.loss, optim, lr_scheduler: scheduler }
}

///|
/// Single training step: zero_grad → forward → loss → backward → step.
/// Returns the loss tensor (read scalar before calling step if needed).
/// NOTE: For GPU, caller must call gpu_workspace_reset() after this.
///       For CPU, no workspace cleanup needed.
pub fn[T : @tensor.Tensor] Experiment::train_step(
  self : Experiment[T],
  x : T,
  y : T,
) -> T {
  self.optim.zero_grad()
  let gx = @grad.no_grad(x)
  let gy = @grad.no_grad(y)
  let y_hat = (self.forward)(gx, true)
  let loss = (self.loss)(y_hat, gy)
  loss.backward()
  let loss_val = loss.value
  self.optim.step(self.lr_scheduler.lr)
  loss_val
}

///|
/// Single eval forward pass (no grad, no optim step).
/// Returns the raw output tensor (not wrapped in Grad).
pub fn[T] Experiment::eval_step(self : Experiment[T], x : T) -> T {
  (self.forward)(@grad.no_grad(x), false).value
}

///|
/// Run the full training loop (CPU or any Tensor backend).
/// No workspace management needed — safe for all backends.
pub fn[T : @tensor.Tensor] Experiment::run(
  self : Experiment[T],
  train_loader : @dl.DataLoader[T],
  epochs? : Int = 10,
  batch_size? : Int = 32,
  on_batch_end? : (Int, T) -> Unit,
  on_epoch_end? : (Int, Int) -> Unit,
  post_batch? : () -> Unit,
) -> Unit {
  for epoch in 0..<epochs {
    let mut n = 0
    for batch in train_loader.batches(batch_size, shuffle=true) {
      let (input, labels) = batch
      let loss_val = self.train_step(input, labels)
      ignore(loss_val)
      n += 1
      match on_batch_end {
        Some(cb) => cb(n, loss_val)
        None => ()
      }
      match post_batch {
        Some(cb) => cb()
        None => ()
      }
    }
    match on_epoch_end {
      Some(cb) => cb(epoch + 1, n)
      None => ()
    }
    ignore(self.lr_scheduler.step())
  }
}

///|
/// Run the full training loop on CPU with eval and CSV logging.
/// Mirrors @gpu_train.run_gpu API for consistent usage across backends.
pub fn[T : @tensor.Tensor] run_cpu(
  experiment : Experiment[T],
  train_loader : @dl.DataLoader[T],
  epochs? : Int = 10,
  batch_size? : Int = 128,
  eval_loader? : @dl.DataLoader[T],
  on_batch_end? : (Int, Int, Float) -> Unit,
  on_epoch_end? : (Int, Double, Double, Double) -> Unit,
) -> Unit {
  let num_samples = train_loader.num_samples()
  let total_batches = num_samples / batch_size +
    (if num_samples % batch_size > 0 { 1 } else { 0 })

  println("epoch,lr,loss,train_acc,test_acc,epoch_s")

  for epoch in 0..<epochs {
    let mut total_loss = 0.0
    let mut n = 0
    let epoch_start = _now_ms()

    for batch in train_loader.batches(batch_size, shuffle=true) {
      let (input, labels) = batch
      let loss_val = experiment.train_step(input, labels)
      let loss_scalar = loss_val.value().data[0]
      total_loss = total_loss + loss_scalar.to_double()
      ignore(loss_val)
      n += 1
      match on_batch_end {
        Some(cb) => cb(epoch + 1, n, loss_scalar)
        None =>
          if n % 100 == 0 || n == total_batches {
            println(
              "epoch=\{epoch + 1}/\{epochs} batch=\{n}/\{total_batches} loss=\{loss_scalar}",
            )
          }
      }
    }

    let avg_loss = total_loss / n.to_double()
    let epoch_s = (_now_ms() - epoch_start).to_double() / 1000.0

    let (train_acc, test_acc) = match eval_loader {
      Some(eval_ld) => {
        let train_acc = _eval_accuracy(experiment, train_loader)
        let test_acc = _eval_accuracy(experiment, eval_ld)
        (train_acc, test_acc)
      }
      None => (0.0, 0.0)
    }

    match on_epoch_end {
      Some(cb) => cb(epoch + 1, avg_loss, train_acc, test_acc)
      None => {
        let lr = experiment.lr_scheduler.lr
        println(
          "Epoch \{epoch + 1}/\{epochs} lr=\{lr} loss=\{avg_loss} train_acc=\{train_acc * 100.0}% test_acc=\{test_acc * 100.0}% time=\{epoch_s}s",
        )
        println(
          "\{epoch + 1},\{lr},\{avg_loss},\{train_acc * 100.0},\{test_acc * 100.0},\{epoch_s}",
        )
      }
    }

    ignore(experiment.lr_scheduler.step())
  }
}

///|
extern "c" fn train_get_time_ms() -> Double = "train_get_time_ms"

///|
/// Get current time in milliseconds.
fn _now_ms() -> Int {
  train_get_time_ms().to_int()
}

///|
/// Evaluate accuracy by iterating eval batches and checking argmax
/// on output vs integer labels.
fn[T : @tensor.Tensor] _eval_accuracy(
  experiment : Experiment[T],
  loader : @dl.DataLoader[T],
) -> Double {
  let mut correct = 0
  let mut total = 0
  for pair in experiment.eval(loader) {
    let (output, labels) = pair
    let o = output.value()
    let l = labels.value()
    // output: [batch, num_classes], labels: [batch]
    let bs = o.dims[0]
    let num_classes = o.dims[1]
    for b in 0..<bs {
      let base = b * num_classes
      let mut max_idx = 0
      let mut max_val = o.data[base]
      for c in 1..<num_classes {
        if o.data[base + c] > max_val {
          max_val = o.data[base + c]
          max_idx = c
        }
      }
      let label_idx = l.data[b].to_int()
      if max_idx == label_idx {
        correct = correct + 1
      }
      total = total + 1
    }
  }
  correct.to_double() / total.to_double()
}

///|
/// Run eval loop: feed all batches through eval_step, returning
/// an array of (output, labels) for metric computation.
pub fn[T] Experiment::eval(
  self : Experiment[T],
  loader : @dl.DataLoader[T],
  batch_size? : Int = 256,
) -> Array[(T, T)] {
  let result : Array[(T, T)] = []
  for batch in loader.batches(batch_size, shuffle=false) {
    let (input, labels) = batch
    let output = self.eval_step(input)
    result.push((output, labels))
  }
  result
}
