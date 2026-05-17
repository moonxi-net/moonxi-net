// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
test (b : @bench.T) {
  // Pre-allocate ALL tensors

  // Conv2d Forward
  let stem_in = @gt.GpuTensor::zeros([128, 3, 32, 32])
  let stem_w = @gt.GpuTensor::zeros([64, 3, 7, 7])
  let stem_b = @gt.GpuTensor::zeros([64])

  let c3_in = @gt.GpuTensor::zeros([128, 64, 8, 8])
  let c3_w = @gt.GpuTensor::zeros([64, 64, 3, 3])
  let c3_b = @gt.GpuTensor::zeros([64])

  let c3_w128 = @gt.GpuTensor::zeros([128, 64, 3, 3])
  let c3_b128 = @gt.GpuTensor::zeros([128])

  let c1_w = @gt.GpuTensor::zeros([128, 64, 1, 1])
  let c1_b = @gt.GpuTensor::zeros([128])

  // Conv2d Backward
  let stem_grad = @gt.GpuTensor::zeros([128, 64, 16, 16])
  let c3_grad = @gt.GpuTensor::zeros([128, 64, 8, 8])

  // BatchNorm
  let bn_in = @gt.GpuTensor::zeros([128, 64, 16, 16])
  let bn_g = @gt.GpuTensor::zeros([64])
  let bn_b = @gt.GpuTensor::zeros([64])
  let bn_rm = @gt.GpuTensor::zeros([64])
  let bn_rv = @gt.GpuTensor::zeros([64])
  let bn_sm = @gt.GpuTensor::zeros([64])
  let bn_siv = @gt.GpuTensor::zeros([64])
  let bn_grad = @gt.GpuTensor::zeros([128, 64, 16, 16])
  let bn_in8 = @gt.GpuTensor::zeros([128, 64, 8, 8])
  let bn_grad8 = @gt.GpuTensor::zeros([128, 64, 8, 8])
  let bn_sm8 = @gt.GpuTensor::zeros([64])
  let bn_siv8 = @gt.GpuTensor::zeros([64])

  // ReLU
  let relu_in = @gt.GpuTensor::zeros([128, 64, 16, 16])
  let relu_grad = @gt.GpuTensor::zeros([128, 64, 16, 16])

  // MaxPool
  let mp_in = @gt.GpuTensor::zeros([128, 64, 16, 16])
  let mp_grad = @gt.GpuTensor::zeros([128, 64, 8, 8])

  // AdaptiveAvgPool
  let aap_in = @gt.GpuTensor::zeros([128, 512, 4, 4])
  let aap_grad = @gt.GpuTensor::zeros([128, 512, 1, 1])

  // Linear
  let lin_in = @gt.GpuTensor::zeros([128, 512])
  let lin_w = @gt.GpuTensor::zeros([512, 10])

  // Softmax CE
  let ce_logits = @gt.GpuTensor::zeros([128, 10])
  let ce_targets = @gt.GpuTensor::zeros([128, 10])

  // Elementwise
  let ew_a = @gt.GpuTensor::zeros([128, 64, 16, 16])
  let ew_b = @gt.GpuTensor::zeros([128, 64, 16, 16])

  // Optimizer
  let opt_p = @gt.GpuTensor::zeros([64, 64, 3, 3])
  let opt_g = @gt.GpuTensor::zeros([64, 64, 3, 3])
  let opt_v = @gt.GpuTensor::zeros([64, 64, 3, 3])
  let opt_n = opt_p.numel()
  let opt_buf = @gt.GpuTensor::zeros([1])

  // ── Conv2d Forward ──

  b.bench(name="conv2d 7x7 stem (128,3,32,32)->(128,64,16,16)", fn() {
    let _ = @gt.GpuTensor::conv2d(stem_in, stem_w, stem_b, 2, 3)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="conv2d 3x3 L1   (128,64,8,8)->(128,64,8,8)", fn() {
    let _ = @gt.GpuTensor::conv2d(c3_in, c3_w, c3_b, 1, 1)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="conv2d 3x3 L2s (128,64,8,8)->(128,128,4,4)", fn() {
    let _ = @gt.GpuTensor::conv2d(c3_in, c3_w128, c3_b128, 2, 1)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="conv2d 1x1 sc  (128,64,8,8)->(128,128,4,4)", fn() {
    let _ = @gt.GpuTensor::conv2d(c3_in, c1_w, c1_b, 2, 0)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── Conv2d Backward ──

  b.bench(name="conv2d_bwd_data stem", fn() {
    let _ = @gt.GpuTensor::conv2d_backward_data(
      stem_grad, stem_w, stem_in, 2, 3,
    )
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="conv2d_bwd_filt stem", fn() {
    let _ = @gt.GpuTensor::conv2d_backward_weight(
      stem_grad, stem_in, stem_w, 2, 3,
    )
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="conv2d_bwd_bias stem", fn() {
    let _ = @gt.GpuTensor::conv2d_backward_bias(stem_grad)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="conv2d_bwd_data 3x3 L1", fn() {
    let _ = @gt.GpuTensor::conv2d_backward_data(c3_grad, c3_w, c3_in, 1, 1)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="conv2d_bwd_filt 3x3 L1", fn() {
    let _ = @gt.GpuTensor::conv2d_backward_weight(c3_grad, c3_in, c3_w, 1, 1)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── BatchNorm ──

  b.bench(name="bn_train fwd  (128,64,16,16)", fn() {
    let _ = @gt.GpuTensor::batchnorm_training(
      bn_in, bn_g, bn_b, bn_rm, bn_rv, 0.1, 0.00001,
    )
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="bn_bwd        (128,64,16,16)", fn() {
    let _ = @gt.GpuTensor::batchnorm_backward(
      bn_grad, bn_in, bn_g, bn_sm, bn_siv, 0.00001,
    )
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="bn_train fwd  (128,64,8,8)", fn() {
    let _ = @gt.GpuTensor::batchnorm_training(
      bn_in8, bn_g, bn_b, bn_rm, bn_rv, 0.1, 0.00001,
    )
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="bn_bwd        (128,64,8,8)", fn() {
    let _ = @gt.GpuTensor::batchnorm_backward(
      bn_grad8, bn_in8, bn_g, bn_sm8, bn_siv8, 0.00001,
    )
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── ReLU ──

  b.bench(name="relu fwd      (128,64,16,16)", fn() {
    let _ = @gt.GpuTensor::relu(relu_in)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="relu bwd      (128,64,16,16)", fn() {
    let _ = @gt.GpuTensor::relu_backward(relu_grad, relu_in)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── MaxPool ──

  b.bench(name="maxpool fwd   (128,64,16,16)->(128,64,8,8)", fn() {
    let _ = @gt.GpuTensor::maxpool2d(mp_in, 3, 2)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="maxpool bwd   (128,64,16,16)", fn() {
    let _ = @gt.GpuTensor::maxpool2d_backward(mp_grad, mp_in, 3, 2)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── AdaptiveAvgPool ──

  b.bench(name="adaptpool fwd (128,512,4,4)->(128,512,1,1)", fn() {
    let _ = @gt.GpuTensor::adaptive_avg_pool2d(aap_in, 1)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="adaptpool bwd (128,512,4,4)", fn() {
    let _ = @gt.GpuTensor::adaptive_avg_pool2d_backward(aap_grad, aap_in)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── Linear (matmul) ──

  b.bench(name="matmul fwd   (128,512)@(10,512)->(128,10)", fn() {
    let _ = @gt.GpuTensor::matmul(lin_in, lin_w)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── Softmax CE ──

  b.bench(name="softmax_ce fwd (128,10)", fn() {
    let _ = @gt.GpuTensor::softmax_cross_entropy(ce_logits, ce_targets)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="softmax_ce bwd (128,10)", fn() {
    let _ = @gt.GpuTensor::softmax_ce_backward(ce_logits, ce_targets)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  // ── Elementwise ops ──

  b.bench(name="add fwd      (128,64,16,16)", fn() {
    let _ = @gt.GpuTensor::add(ew_a, ew_b)
    @cuda.sync()
    @gt.gpu_workspace_reset()
    @cuda.sync()
  })

  b.bench(name="add_into     (128,64,16,16)", fn() {
    ew_a.add_into(ew_b)
    @cuda.sync()
  })

  // ── Optimizer kernels ──

  b.bench(name="norm_sq      36864 elems", fn() {
    let _ = @cuda.norm_sq(opt_g.data, opt_n, opt_buf.data)
    @cuda.sync()
  })

  b.bench(name="read_scalar  (D2H sync)", fn() {
    let _ = @cuda.read_scalar(opt_buf.data)
    @cuda.sync()
  })

  b.bench(name="sgd_momentum 36864 elems", fn() {
    let _ = @cuda.sgd_momentum_step(
      opt_p.data,
      opt_g.data,
      opt_v.data,
      opt_n,
      0.01,
      0.9,
      0.0001,
      1.0,
    )
    @cuda.sync()
  })
}
