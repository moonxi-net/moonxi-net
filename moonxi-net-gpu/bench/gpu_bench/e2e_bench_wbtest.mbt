// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// End-to-end train_step benchmark with mock data (no CIFAR-10 loading)

///|
test (b : @bench.T) {
  let batch_size = 128
  let num_classes = 10

  // Mock data: zeros instead of loading CIFAR-10
  let h_img = FixedArray::make(
    batch_size * 3 * 32 * 32,
    Float::from_double(0.0),
  )
  let h_lbl = FixedArray::make(
    batch_size * num_classes,
    Float::from_double(0.0),
  )

  // Upload mock data to GPU once
  let img_bytes = Int64::from_int(batch_size * 3 * 32 * 32 * 4)
  let lbl_bytes = Int64::from_int(batch_size * num_classes * 4)
  let imgs_gpu = @cuda.alloc(img_bytes)
  let _ = @cuda.copy_h2d(imgs_gpu, h_img)
  let lbls_gpu = @cuda.alloc(lbl_bytes)
  let _ = @cuda.copy_h2d(lbls_gpu, h_lbl)

  let image_size = 3 * 32 * 32
  let d_idx = @cuda.alloc(Int64::from_int(batch_size * 4))
  let h_idx = FixedArray::make(batch_size, 0)

  // Model + optimizer
  let model = @model.init_resnet18(num_classes)
  let optim = @optimizer.momentum_sgd(model.params(), 0.9, 0.0001, 35.0)
  let lr : Float = 0.01
  let experiment = @train.Experiment::{
    forward: model.as_module().forward,
    loss: @loss.cross_entropy_onehot_loss().loss,
    optim,
    lr_scheduler: @optimizer.StepLR::new(lr.to_double(), 1, 1.0),
  }

  // Warmup 1 batch
  for i in 0..<1 {
    for j in 0..<batch_size {
      h_idx[j] = i * batch_size + j
    }
    let _ = @cuda.raw_h2d(d_idx, h_idx)
    let input_gpu = @gt.GpuTensor::alloc_shape([batch_size, 3, 32, 32])
    let _ = @cuda.batch_gather(
      imgs_gpu,
      d_idx,
      input_gpu.data,
      image_size,
      batch_size,
    )
    let target_gpu = @gt.GpuTensor::alloc_shape([batch_size, num_classes])
    let _ = @cuda.batch_gather(
      lbls_gpu,
      d_idx,
      target_gpu.data,
      num_classes,
      batch_size,
    )
    let _ = experiment.train_step(input_gpu, target_gpu)
    @gt.gpu_workspace_reset()
  }

  // Bench: GPU-resident train_step with batch_gather
  let n_bench = 1

  b.bench(name="gpu_resident_train_step (mock data)", fn() {
    for i in 0..<n_bench {
      let b_start = i * batch_size
      for j in 0..<batch_size {
        h_idx[j] = b_start + j
      }
      let _ = @cuda.raw_h2d(d_idx, h_idx)
      let input_gpu = @gt.GpuTensor::alloc_shape([batch_size, 3, 32, 32])
      let _ = @cuda.batch_gather(
        imgs_gpu,
        d_idx,
        input_gpu.data,
        image_size,
        batch_size,
      )
      let target_gpu = @gt.GpuTensor::alloc_shape([batch_size, num_classes])
      let _ = @cuda.batch_gather(
        lbls_gpu,
        d_idx,
        target_gpu.data,
        num_classes,
        batch_size,
      )
      let _ = experiment.train_step(input_gpu, target_gpu)
      @gt.gpu_workspace_reset()
    }
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // Bench: H2D streaming train_step (copy each batch from host)
  let stream_img = FixedArray::make(
    batch_size * image_size,
    Float::from_double(0.0),
  )
  let stream_lbl = FixedArray::make(
    batch_size * num_classes,
    Float::from_double(0.0),
  )

  b.bench(name="h2d_streaming_train_step (mock data)", fn() {
    for _ in 0..<n_bench {
      let input = @gt.GpuTensor::alloc_shape([batch_size, 3, 32, 32])
      let _ = @cuda.copy_h2d(input.data, stream_img)
      let target = @gt.GpuTensor::alloc_shape([batch_size, num_classes])
      let _ = @cuda.copy_h2d(target.data, stream_lbl)
      let _ = experiment.train_step(input, target)
      @gt.gpu_workspace_reset()
    }
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })
}
