// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// =============================================================================
// GPU training: run_gpu — full training loop with workspace management
// =============================================================================

///|
/// Run the full training loop on GPU.
/// Auto-inits CUDA and memory pool. Handles gpu_workspace_reset() internally.
/// Default logging: batch progress every 100 steps, epoch summary with CSV.
///
/// Researcher never touches CUDA init, workspace management, or logging.
pub fn run_gpu(
  experiment : @cpu_train.Experiment[@gt.GpuTensor],
  train_loader : @dl.DataLoader[@gt.GpuTensor],
  epochs? : Int = 10,
  batch_size? : Int = 128,
  eval_loader? : @dl.DataLoader[@gt.GpuTensor],
  device_id? : Int = 0,
  pool_capacity? : Int64 = 6_442_450_944L,
  on_batch_end? : (Int, Int, Float) -> Unit,
  on_epoch_end? : (Int, Double, Double, Double) -> Unit,
) -> Unit {
  // Auto-init CUDA and memory pool
  @cuda.gpu_init(device_id)
  @gt.gpu_enable_pool(pool_capacity)

  let num_samples = train_loader.num_samples()
  let total_batches = num_samples / batch_size +
    (if num_samples % batch_size > 0 { 1 } else { 0 })

  // Print CSV header
  println("epoch,lr,loss,train_acc,test_acc,epoch_s")

  for epoch in 0..<epochs {
    let mut total_loss = 0.0
    let mut n = 0
    let epoch_start = @cuda.get_time_ms()

    for batch in train_loader.batches(batch_size, shuffle=true) {
      let (input, labels) = batch

      // Inline train_step to read loss BEFORE step (GPU-specific ordering)
      experiment.optim.zero_grad()
      let gx = @grad.no_grad(input)
      let gy = @grad.no_grad(labels)
      let y_hat = (experiment.forward)(gx, true)
      let loss = (experiment.loss)(y_hat, gy)
      loss.backward()
      // Read loss BEFORE step (fused optimizers may reset workspace)
      let loss_scalar = loss.value.item()
      total_loss = total_loss + loss_scalar.to_double()
      experiment.optim.step(experiment.lr_scheduler.lr)
      @gt.gpu_workspace_reset()

      n += 1
      match on_batch_end {
        Some(cb) => cb(epoch + 1, n, loss_scalar)
        None =>
          // Default: log every 100 batches or last batch
          if n % 100 == 0 || n == total_batches {
            @log.info(
              "epoch=\{epoch + 1}/\{epochs} batch=\{n}/\{total_batches} loss=\{loss_scalar}",
            )
          }
      }
    }

    let avg_loss = total_loss / n.to_double()
    let epoch_train_s = (@cuda.get_time_ms() - epoch_start) / 1000.0

    // Eval if eval_loader provided
    let (train_acc, test_acc) = match eval_loader {
      Some(eval_ld) => {
        let train_acc = _gpu_eval_accuracy(experiment, train_loader)
        let test_acc = _gpu_eval_accuracy(experiment, eval_ld)
        (train_acc, test_acc)
      }
      None => (0.0, 0.0)
    }

    match on_epoch_end {
      Some(cb) => cb(epoch + 1, avg_loss, train_acc, test_acc)
      None => {
        // Default: log epoch summary + CSV row
        let lr = experiment.lr_scheduler.lr
        @log.info(
          "Epoch \{epoch + 1}/\{epochs} lr=\{lr} loss=\{avg_loss} train_acc=\{train_acc * 100.0}% test_acc=\{test_acc * 100.0}% time=\{epoch_train_s}s",
        )
        println(
          "\{epoch + 1},\{lr},\{avg_loss},\{train_acc * 100.0},\{test_acc * 100.0},\{epoch_train_s}",
        )
      }
    }

    // Step lr scheduler at end of epoch
    ignore(experiment.lr_scheduler.step())
  }
}

///|
/// Evaluate accuracy using DataLoader batches with GPU argmax.
fn _gpu_eval_accuracy(
  experiment : @cpu_train.Experiment[@gt.GpuTensor],
  loader : @dl.DataLoader[@gt.GpuTensor],
) -> Double {
  @gt.gpu_with_inference(true, fn() {
    let eval_batch_size = 256
    let mut correct = 0
    let mut total = 0
    for batch in loader.batches(eval_batch_size) {
      let (input, labels) = batch
      let logits = experiment.eval_step(input)
      let predictions = logits.argmax()
      let preds_h = predictions.to_host()
      let labels_h = labels.to_host()
      let count = labels_h.length()
      for b in 0..<count {
        if preds_h[b].to_int() == labels_h[b].to_int() {
          correct = correct + 1
        }
        total = total + 1
      }
      @gt.gpu_workspace_reset()
    }
    correct.to_double() / total.to_double()
  })
}
