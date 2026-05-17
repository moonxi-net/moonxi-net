// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// cuda.mbt — Safe public API wrapping the raw FFI layer

///|
pub fn device_count() -> Int {
  moonbit_cuda_device_count()
}

///|
pub fn check_error() -> Int {
  moonbit_cuda_check_error()
}

///|
pub fn peek_error() -> Int {
  moonbit_cuda_peek_error()
}

///|
pub fn device_name() -> String {
  let bytes = moonbit_cuda_device_name()
  @utf8.decode_lossy(bytes)
}

///|
pub fn compute_capability() -> (Int, Int) {
  (
    moonbit_cuda_compute_capability_major(),
    moonbit_cuda_compute_capability_minor(),
  )
}

///|
pub fn alloc(bytes : Int64) -> GpuBuffer {
  moonbit_cuda_alloc(bytes)
}

///|
pub fn copy_h2d(dst : GpuBuffer, src : FixedArray[Float]) -> Int {
  moonbit_cuda_copy_h2d(dst, src, Int64::from_int(src.length()))
}

///|
pub fn raw_h2d(dst : GpuBuffer, src : FixedArray[Int]) -> Int {
  moonbit_cuda_raw_h2d(dst, src, Int64::from_int(src.length() * 4))
}

///|
pub fn copy_ptrs_h2d(dst : GpuBuffer, src : FixedArray[GpuBuffer]) -> Int {
  moonbit_cuda_copy_ptrs_h2d(dst, src, src.length())
}

///|
pub fn memset_zero(dst : GpuBuffer, count : Int64) -> Int {
  moonbit_cuda_memset_zero(dst, count)
}

///|
pub fn copy_d2h(dst : FixedArray[Float], src : GpuBuffer) -> Int {
  moonbit_cuda_copy_d2h(dst, src, Int64::from_int(dst.length()))
}

///|
pub fn copy_h2d_offset(
  dst : GpuBuffer,
  src_base : FixedArray[Float],
  offset_elements : Int,
  count_elements : Int,
) -> Int {
  moonbit_cuda_copy_h2d_offset(dst, src_base, offset_elements, count_elements)
}

///|
pub fn vec_add(a : GpuBuffer, b : GpuBuffer, out : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_vec_add(a, b, out, n)
}

///|
pub fn cublas_create() -> CublasHandle {
  moonbit_cublas_create()
}

///|
pub fn cublas_destroy(handle : CublasHandle) -> Unit {
  moonbit_cublas_destroy(handle)
}

///|
pub fn cublas_sgemv(
  handle : CublasHandle,
  m : Int,
  n : Int,
  alpha : Float,
  d_A : GpuBuffer,
  d_x : GpuBuffer,
  beta : Float,
  d_y : GpuBuffer,
) -> Int {
  moonbit_cublas_sgemv(handle, m, n, alpha, d_A, d_x, beta, d_y)
}

///|
pub fn cublas_sgemm(
  handle : CublasHandle,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  d_A : GpuBuffer,
  d_B : GpuBuffer,
  beta : Float,
  d_C : GpuBuffer,
) -> Int {
  moonbit_cublas_sgemm(handle, m, n, k, alpha, d_A, d_B, beta, d_C)
}

// === cuDNN Handle ===

///|
pub fn cudnn_create() -> CudnnHandle {
  moonbit_cudnn_create()
}

///|
pub fn cudnn_destroy(handle : CudnnHandle) -> Unit {
  moonbit_cudnn_destroy(handle)
}

// === Conv2d ===

///|
pub fn conv2d_create(
  n : Int,
  c : Int,
  h : Int,
  w : Int,
  out_c : Int,
  kh : Int,
  kw : Int,
  pad_h : Int,
  pad_w : Int,
  stride_h : Int,
  stride_w : Int,
  dilation_h : Int,
  dilation_w : Int,
) -> Conv2dContext {
  moonbit_conv2d_create(
    n, c, h, w, out_c, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
  )
}

///|
pub fn conv2d_forward(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  weight : GpuBuffer,
  bias : GpuBuffer,
  output : GpuBuffer,
) -> Int {
  moonbit_conv2d_forward(ctx, cudnn_handle, input, weight, bias, output)
}

///|
pub fn conv2d_forward_bias_relu(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  weight : GpuBuffer,
  bias : GpuBuffer,
  output : GpuBuffer,
) -> Int {
  moonbit_conv2d_forward_bias_relu(
    ctx, cudnn_handle, input, weight, bias, output,
  )
}

///|
pub fn conv2d_backward_data(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  grad_output : GpuBuffer,
  weight : GpuBuffer,
  grad_input : GpuBuffer,
) -> Int {
  moonbit_conv2d_backward_data(
    ctx, cudnn_handle, grad_output, weight, grad_input,
  )
}

///|
pub fn conv2d_backward_filter(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  grad_output : GpuBuffer,
  grad_weight : GpuBuffer,
) -> Int {
  moonbit_conv2d_backward_filter(
    ctx, cudnn_handle, input, grad_output, grad_weight,
  )
}

///|
pub fn conv2d_backward_bias(
  cudnn_handle : CudnnHandle,
  grad_output : GpuBuffer,
  grad_bias : GpuBuffer,
  batch : Int,
  out_c : Int,
  out_h : Int,
  out_w : Int,
) -> Int {
  moonbit_conv2d_backward_bias(
    cudnn_handle, grad_output, grad_bias, batch, out_c, out_h, out_w,
  )
}

///|
pub fn conv2d_destroy(ctx : Conv2dContext) -> Unit {
  moonbit_conv2d_destroy(ctx)
}

// === BatchNorm ===

///|
pub fn batchnorm_create(n : Int, c : Int, h : Int, w : Int) -> BatchNormContext {
  moonbit_batchnorm_create(n, c, h, w)
}

///|
pub fn batchnorm_forward(
  ctx : BatchNormContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  output : GpuBuffer,
  bn_scale : GpuBuffer,
  bn_bias : GpuBuffer,
  bn_running_mean : GpuBuffer,
  bn_running_var : GpuBuffer,
  bn_save_mean : GpuBuffer,
  bn_save_inv_var : GpuBuffer,
  momentum : Float,
  epsilon : Float,
  is_training : Bool,
) -> Int {
  moonbit_batchnorm_forward(
    ctx,
    cudnn_handle,
    input,
    output,
    bn_scale,
    bn_bias,
    bn_running_mean,
    bn_running_var,
    bn_save_mean,
    bn_save_inv_var,
    momentum,
    epsilon,
    if is_training {
      1
    } else {
      0
    },
  )
}

///|
pub fn cuda_batchnorm_inference(
  input : GpuBuffer,
  output : GpuBuffer,
  gamma : GpuBuffer,
  beta : GpuBuffer,
  running_mean : GpuBuffer,
  running_var : GpuBuffer,
  epsilon : Float,
  n : Int,
  c : Int,
  hw : Int,
) -> Int {
  moonbit_cuda_batchnorm_inference(
    input, output, gamma, beta, running_mean, running_var, epsilon, n, c, hw,
  )
}

///|
pub fn batchnorm_backward(
  ctx : BatchNormContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  grad_output : GpuBuffer,
  grad_input : GpuBuffer,
  bn_scale : GpuBuffer,
  grad_bn_scale : GpuBuffer,
  grad_bn_bias : GpuBuffer,
  bn_save_mean : GpuBuffer,
  bn_save_inv_var : GpuBuffer,
  epsilon : Float,
) -> Int {
  moonbit_batchnorm_backward(
    ctx, cudnn_handle, input, grad_output, grad_input, bn_scale, grad_bn_scale, grad_bn_bias,
    bn_save_mean, bn_save_inv_var, epsilon,
  )
}

///|
pub fn batchnorm_destroy(ctx : BatchNormContext) -> Unit {
  moonbit_batchnorm_destroy(ctx)
}

// === ReLU via cuDNN ===

///|
pub fn cudnn_relu_forward(
  handle : CudnnHandle,
  n_elements : Int64,
  input : GpuBuffer,
  output : GpuBuffer,
) -> Int {
  moonbit_cudnn_relu_forward(handle, n_elements, input, output)
}

///|
pub fn cudnn_relu_backward(
  handle : CudnnHandle,
  n_elements : Int64,
  input : GpuBuffer,
  grad_output : GpuBuffer,
  grad_input : GpuBuffer,
) -> Int {
  moonbit_cudnn_relu_backward(
    handle, n_elements, input, grad_output, grad_input,
  )
}

// === Pooling ===

///|
pub fn pool2d_create(
  n : Int,
  c : Int,
  h : Int,
  w : Int,
  kh : Int,
  kw : Int,
  pad_h : Int,
  pad_w : Int,
  stride_h : Int,
  stride_w : Int,
  pool_type : Int,
) -> Pool2dContext {
  moonbit_pool2d_create(
    n, c, h, w, kh, kw, pad_h, pad_w, stride_h, stride_w, pool_type,
  )
}

///|
pub fn pool2d_forward(
  ctx : Pool2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  output : GpuBuffer,
) -> Int {
  moonbit_pool2d_forward(ctx, cudnn_handle, input, output)
}

///|
pub fn pool2d_backward(
  ctx : Pool2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  grad_output : GpuBuffer,
  grad_input : GpuBuffer,
) -> Int {
  moonbit_pool2d_backward(ctx, cudnn_handle, input, grad_output, grad_input)
}

///|
pub fn pool2d_destroy(ctx : Pool2dContext) -> Unit {
  moonbit_pool2d_destroy(ctx)
}

// === Cache Management ===

///|
pub fn conv2d_cache_clear() -> Unit {
  moonbit_conv2d_cache_clear()
}

///|
pub fn conv2d_cache_size() -> Int {
  moonbit_conv2d_cache_size()
}

///|
pub fn batchnorm_cache_clear() -> Unit {
  moonbit_batchnorm_cache_clear()
}

///|
pub fn batchnorm_cache_size() -> Int {
  moonbit_batchnorm_cache_size()
}

///|
pub fn pool2d_cache_clear() -> Unit {
  moonbit_pool2d_cache_clear()
}

///|
pub fn pool2d_cache_size() -> Int {
  moonbit_pool2d_cache_size()
}

///|
pub fn pool_create(capacity : Int64) -> GpuPool {
  moonbit_pool_create(capacity)
}

///|
pub fn pool_alloc(pool : GpuPool, size : Int64) -> GpuBuffer {
  moonbit_pool_alloc(pool, size)
}

///|
pub fn pool_reset(pool : GpuPool) -> Unit {
  moonbit_pool_reset(pool)
}

///|
pub fn pool_used(pool : GpuPool) -> Int64 {
  moonbit_pool_used(pool)
}

///|
pub fn pool_peak(pool : GpuPool) -> Int64 {
  moonbit_pool_peak(pool)
}

///|
pub fn pool_capacity(pool : GpuPool) -> Int64 {
  moonbit_pool_capacity(pool)
}

// === Transpose and elementwise kernels ===

///|
pub fn transpose_nchw_to_nhwc(
  src : GpuBuffer,
  dst : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int {
  moonbit_cuda_transpose_nchw_to_nhwc(src, dst, n, c, h, w)
}

///|
pub fn transpose_nhwc_to_nchw(
  src : GpuBuffer,
  dst : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int {
  moonbit_cuda_transpose_nhwc_to_nchw(src, dst, n, c, h, w)
}

///|
pub fn bias_add_nchw(
  data : GpuBuffer,
  bias : GpuBuffer,
  n : Int,
  c : Int,
  hw : Int,
) -> Int {
  moonbit_cuda_bias_add_nchw(data, bias, n, c, hw)
}

///|
pub fn elementwise_add(
  a : GpuBuffer,
  b : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int {
  moonbit_cuda_elementwise_add(a, b, out, n)
}

///|
pub fn elementwise_add_into(dst : GpuBuffer, src : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_elementwise_add_into(dst, src, n)
}

// === Reduce Sum Spatial ===

///|
pub fn reduce_sum_spatial(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int {
  moonbit_cuda_reduce_sum_spatial(input, output, n, c, h, w)
}

///|
pub fn reduce_sum_leading(
  input : GpuBuffer,
  output : GpuBuffer,
  output_numel : Int,
  lead_stride : Int,
) -> Int {
  moonbit_cuda_reduce_sum_leading(input, output, output_numel, lead_stride)
}

///|
pub fn reduce_sum_dim(
  input : GpuBuffer,
  output : GpuBuffer,
  output_numel : Int,
  dim_size : Int,
  stride_before : Int,
  stride_after : Int,
) -> Int {
  moonbit_cuda_reduce_sum_dim(
    input, output, output_numel, dim_size, stride_before, stride_after,
  )
}

// === Adaptive Avg Pool2d ===

///|
pub fn adaptive_avg_pool2d_forward(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int {
  moonbit_cuda_adaptive_avg_pool2d_forward(input, output, n, c, h, w)
}

///|
pub fn adaptive_avg_pool2d_backward(
  dy : GpuBuffer,
  d_input : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int {
  moonbit_cuda_adaptive_avg_pool2d_backward(dy, d_input, n, c, h, w)
}

// === Softmax CE ===

///|
pub fn softmax_ce_forward(
  logits : GpuBuffer,
  targets : GpuBuffer,
  loss_output : GpuBuffer,
  softmax_output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int {
  moonbit_cuda_softmax_ce_forward(
    logits, targets, loss_output, softmax_output, batch, num_classes,
  )
}

///|
pub fn softmax_ce_backward(
  logits : GpuBuffer,
  targets : GpuBuffer,
  output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int {
  moonbit_cuda_softmax_ce_backward(logits, targets, output, batch, num_classes)
}

// === Softmax CE with integer labels ===

///|
/// Softmax cross-entropy forward with integer class labels.
/// labels: GpuBuffer of shape [batch], each value is an integer class index stored as float.
pub fn softmax_ce_forward_labels(
  logits : GpuBuffer,
  labels : GpuBuffer,
  loss_output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int {
  moonbit_cuda_softmax_ce_forward_labels(
    logits, labels, loss_output, batch, num_classes,
  )
}

///|
/// Softmax cross-entropy backward with integer class labels.
/// labels: GpuBuffer of shape [batch], each value is an integer class index stored as float.
pub fn softmax_ce_backward_labels(
  logits : GpuBuffer,
  labels : GpuBuffer,
  output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int {
  moonbit_cuda_softmax_ce_backward_labels(
    logits, labels, output, batch, num_classes,
  )
}

// === Matmul ===

///|
pub fn matmul(
  handle : CublasHandle,
  a : GpuBuffer,
  b : GpuBuffer,
  c : GpuBuffer,
  m : Int,
  k : Int,
  n : Int,
  transpose_a : Bool,
  transpose_b : Bool,
  alpha : Float,
  beta : Float,
) -> Int {
  moonbit_cuda_matmul(
    handle,
    a,
    b,
    c,
    m,
    k,
    n,
    if transpose_a {
      1
    } else {
      0
    },
    if transpose_b {
      1
    } else {
      0
    },
    alpha,
    beta,
  )
}

///|
pub fn accumulate_inplace(dst : GpuBuffer, src : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_accumulate_inplace(dst, src, n)
}

// === GPU Optimizer Kernels ===

///|
pub fn sgd_momentum_step(
  param : GpuBuffer,
  grad : GpuBuffer,
  velocity : GpuBuffer,
  n : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int {
  moonbit_cuda_sgd_momentum_step(
    param, grad, velocity, n, lr, momentum, weight_decay, clip_coef,
  )
}

///|
pub fn sgd_step(
  param : GpuBuffer,
  grad : GpuBuffer,
  n : Int,
  lr : Float,
  weight_decay : Float,
) -> Int {
  moonbit_cuda_sgd_step(param, grad, n, lr, weight_decay)
}

///|
pub fn norm_sq(src : GpuBuffer, n : Int, dst : GpuBuffer) -> Int {
  moonbit_cuda_norm_sq(src, n, dst)
}

///|
pub fn multi_tensor_norm_sq(
  tensor_ptrs : FixedArray[GpuBuffer],
  tensor_sizes : FixedArray[Int],
  num_tensors : Int,
  dst : GpuBuffer,
) -> Int {
  moonbit_cuda_multi_tensor_norm_sq(tensor_ptrs, tensor_sizes, num_tensors, dst)
}

///|
/// GPU-only multi-tensor norm_sq — all pointer arrays already on device.
pub fn multi_tensor_norm_sq_gpu_only(
  d_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
  dst : GpuBuffer,
) -> Int {
  moonbit_cuda_multi_tensor_norm_sq_gpu_only(d_ptrs, d_sizes, num_tensors, dst)
}

///|
/// Multi-block-per-tensor norm_sq for high GPU occupancy.
/// d_block_offsets: prefix-sum of ceil(size/256) per tensor.
/// total_blocks: total number of CUDA blocks.
pub fn mt_norm_sq_gpu_only(
  d_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  d_block_offsets : GpuBuffer,
  num_tensors : Int,
  total_blocks : Int,
  dst : GpuBuffer,
) -> Int {
  moonbit_cuda_mt_norm_sq_gpu_only(
    d_ptrs, d_sizes, d_block_offsets, num_tensors, total_blocks, dst,
  )
}

///|
/// Multi-block-per-tensor SGD+momentum for high GPU occupancy.
pub fn mt_sgd_momentum_gpu_only(
  d_param_ptrs : GpuBuffer,
  d_grad_ptrs : GpuBuffer,
  d_vel_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  d_block_offsets : GpuBuffer,
  num_tensors : Int,
  total_blocks : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int {
  moonbit_cuda_mt_sgd_momentum_gpu_only(
    d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes, d_block_offsets, num_tensors,
    total_blocks, lr, momentum, weight_decay, clip_coef,
  )
}

///|
pub fn mt_sgd_momentum_autoclip(
  d_param_ptrs : GpuBuffer,
  d_grad_ptrs : GpuBuffer,
  d_vel_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  d_block_offsets : GpuBuffer,
  num_tensors : Int,
  total_blocks : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  norm_buf : GpuBuffer,
  max_grad_norm : Float,
) -> Int {
  moonbit_cuda_mt_sgd_momentum_autoclip(
    d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes, d_block_offsets, num_tensors,
    total_blocks, lr, momentum, weight_decay, norm_buf, max_grad_norm,
  )
}

///|
pub fn multi_tensor_sgd_momentum_step(
  param_ptrs : FixedArray[GpuBuffer],
  grad_ptrs : FixedArray[GpuBuffer],
  velocity_ptrs : FixedArray[GpuBuffer],
  tensor_sizes : FixedArray[Int],
  num_tensors : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int {
  moonbit_cuda_multi_tensor_sgd_momentum_step(
    param_ptrs, grad_ptrs, velocity_ptrs, tensor_sizes, num_tensors, lr, momentum,
    weight_decay, clip_coef,
  )
}

///|
pub fn multi_tensor_sgd_momentum_step_prealloc(
  param_ptrs : FixedArray[GpuBuffer],
  grad_ptrs : FixedArray[GpuBuffer],
  velocity_ptrs : FixedArray[GpuBuffer],
  tensor_sizes : FixedArray[Int],
  num_tensors : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
  d_param_ptrs : GpuBuffer,
  d_grad_ptrs : GpuBuffer,
  d_vel_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
) -> Int {
  moonbit_cuda_multi_tensor_sgd_momentum_step_prealloc(
    param_ptrs, grad_ptrs, velocity_ptrs, tensor_sizes, num_tensors, lr, momentum,
    weight_decay, clip_coef, d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes,
  )
}

///|
pub fn multi_tensor_sgd_momentum_step_gpu_only(
  d_param_ptrs : GpuBuffer,
  d_grad_ptrs : GpuBuffer,
  d_vel_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int {
  moonbit_cuda_multi_tensor_sgd_momentum_step_gpu_only(
    d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes, num_tensors, lr, momentum, weight_decay,
    clip_coef,
  )
}

///|
pub fn contiguous_sgd_momentum_step(
  params : GpuBuffer,
  grads : GpuBuffer,
  vels : GpuBuffer,
  total_elements : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int {
  moonbit_cuda_contiguous_sgd_momentum_step(
    params, grads, vels, total_elements, lr, momentum, weight_decay, clip_coef,
  )
}

///|
/// Gather scattered tensors into one flat buffer (all GPU-side).
/// d_src_ptrs: GPU array of float* pointers (device memory)
/// d_offsets: GPU array of int offsets (device memory)
/// d_sizes: GPU array of int sizes (device memory)
/// dst: flat output buffer (pre-allocated)
pub fn gather(
  d_src_ptrs : GpuBuffer,
  d_offsets : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
  dst : GpuBuffer,
) -> Int {
  moonbit_cuda_gather(d_src_ptrs, d_offsets, d_sizes, num_tensors, dst)
}

///|
/// Scatter flat buffer back to scattered tensors (all GPU-side).
pub fn scatter(
  src : GpuBuffer,
  d_dst_ptrs : GpuBuffer,
  d_offsets : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
) -> Int {
  moonbit_cuda_scatter(src, d_dst_ptrs, d_offsets, d_sizes, num_tensors)
}

///|
/// Compute sum-of-squares over a flat GPU buffer, accumulate into out.
pub fn contiguous_norm_sq(data : GpuBuffer, n : Int, out : GpuBuffer) -> Int {
  moonbit_cuda_contiguous_norm_sq(data, n, out)
}

///|
/// GPU-side batch gather: gather N samples from a flat GPU buffer by index.
/// src: flat GPU buffer [total_samples * stride]
/// indices: GPU int32 array [batch_size]
/// dst: flat GPU buffer [batch_size * stride]
/// stride: elements per sample (e.g. 3072 for CIFAR images)
pub fn batch_gather(
  src : GpuBuffer,
  indices : GpuBuffer,
  dst : GpuBuffer,
  stride : Int,
  batch_size : Int,
) -> Int {
  moonbit_cuda_batch_gather(src, indices, dst, stride, batch_size)
}

///|
pub fn scale_inplace(data : GpuBuffer, n : Int, scale : Float) -> Int {
  moonbit_cuda_scale_inplace(data, n, scale)
}

///|
pub fn fill(out : GpuBuffer, n : Int, value : Float) -> Int {
  moonbit_cuda_fill(out, n, value)
}

///|
pub fn read_scalar(src : GpuBuffer) -> Float {
  moonbit_cuda_read_scalar(src)
}

///|
pub fn axpy(n : Int, alpha : Float, x : GpuBuffer, y : GpuBuffer) -> Int {
  moonbit_cuda_axpy(n, alpha, x, y)
}

///|
pub fn get_time_ms() -> Double {
  moonbit_get_time_ms()
}

///|
pub fn sync() -> Unit {
  moonbit_cuda_sync()
}

///|
pub fn sub(a : GpuBuffer, b : GpuBuffer, out : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_sub(a, b, out, n)
}

///|
pub fn mul_elem(a : GpuBuffer, b : GpuBuffer, out : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_mul_elem(a, b, out, n)
}

///|
pub fn square(src : GpuBuffer, out : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_square(src, out, n)
}

///|
pub fn sqrt(src : GpuBuffer, out : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_sqrt(src, out, n)
}

///|
pub fn div_elem(a : GpuBuffer, b : GpuBuffer, out : GpuBuffer, n : Int) -> Int {
  moonbit_cuda_div_elem(a, b, out, n)
}

///|
pub fn scale_out(
  src : GpuBuffer,
  out : GpuBuffer,
  n : Int,
  scale : Float,
) -> Int {
  moonbit_cuda_scale_out(src, out, n, scale)
}

///|
pub fn transpose_2d(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int,
  m : Int,
) -> Int {
  moonbit_cuda_transpose_2d(input, output, n, m)
}

// === CUDA Events ===

///|
pub fn event_create() -> CudaEvent {
  moonbit_cuda_event_create()
}

///|
pub fn null_buffer() -> GpuBuffer {
  moonbit_cuda_null_buffer()
}

///|
pub fn event_record(stream : GpuBuffer, event : CudaEvent) -> Int {
  moonbit_cuda_event_record(stream, event)
}

///|
pub fn event_elapsed_ms(start : CudaEvent, end : CudaEvent) -> Float {
  moonbit_cuda_event_elapsed_ms(start, end)
}

///|
pub fn event_synchronize(event : CudaEvent) -> Int {
  moonbit_cuda_event_synchronize(event)
}

///|
pub fn argmax(
  input : GpuBuffer,
  output : GpuBuffer,
  outer : Int,
  inner : Int,
) -> Int {
  moonbit_cuda_argmax(input, output, outer, inner)
}

///|
pub fn mt_adam_autoclip(
  d_param_ptrs : GpuBuffer,
  d_grad_ptrs : GpuBuffer,
  d_m_ptrs : GpuBuffer,
  d_v_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  d_block_offsets : GpuBuffer,
  num_tensors : Int,
  total_blocks : Int,
  lr : Float,
  beta1 : Float,
  beta2 : Float,
  one_minus_beta1 : Float,
  one_minus_beta2 : Float,
  bias_correction1 : Float,
  bias_correction2 : Float,
  eps : Float,
  weight_decay : Float,
  norm_buf : GpuBuffer,
  max_grad_norm : Float,
) -> Int {
  moonbit_cuda_mt_adam_autoclip(
    d_param_ptrs, d_grad_ptrs, d_m_ptrs, d_v_ptrs, d_sizes, d_block_offsets, num_tensors,
    total_blocks, lr, beta1, beta2, one_minus_beta1, one_minus_beta2, bias_correction1,
    bias_correction2, eps, weight_decay, norm_buf, max_grad_norm,
  )
}

///|
pub fn mt_rmsprop_autoclip(
  d_param_ptrs : GpuBuffer,
  d_grad_ptrs : GpuBuffer,
  d_v_ptrs : GpuBuffer,
  d_buf_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  d_block_offsets : GpuBuffer,
  num_tensors : Int,
  total_blocks : Int,
  lr : Float,
  alpha : Float,
  one_minus_alpha : Float,
  eps : Float,
  weight_decay : Float,
  momentum : Float,
  has_momentum : Int,
  norm_buf : GpuBuffer,
  max_grad_norm : Float,
) -> Int {
  moonbit_cuda_mt_rmsprop_autoclip(
    d_param_ptrs, d_grad_ptrs, d_v_ptrs, d_buf_ptrs, d_sizes, d_block_offsets, num_tensors,
    total_blocks, lr, alpha, one_minus_alpha, eps, weight_decay, momentum, has_momentum,
    norm_buf, max_grad_norm,
  )
}
