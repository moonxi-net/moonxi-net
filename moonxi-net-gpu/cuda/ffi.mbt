// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// ffi.mbt — Raw FFI declarations for CUDA/cuBLAS (native only)
// All extern functions map to cuda_stub.c MOONBIT_FFI_EXPORT symbols.

///|
extern "c" fn moonbit_cuda_device_count() -> Int = "moonbit_cuda_device_count"

///|
extern "c" fn moonbit_cuda_check_error() -> Int = "moonbit_cuda_check_error"

///|
extern "c" fn moonbit_cuda_peek_error() -> Int = "moonbit_cuda_peek_error"

///|
extern "c" fn moonbit_cuda_device_name() -> Bytes = "moonbit_cuda_device_name"

///|
extern "c" fn moonbit_cuda_compute_capability_major() -> Int = "moonbit_cuda_compute_capability_major"

///|
extern "c" fn moonbit_cuda_compute_capability_minor() -> Int = "moonbit_cuda_compute_capability_minor"

// === Memory Management ===

///|
extern "c" fn moonbit_cuda_alloc(size : Int64) -> GpuBuffer = "moonbit_cuda_alloc"

// === Memory Transfer ===

///|
#borrow(dst, src)
extern "c" fn moonbit_cuda_copy_h2d(
  dst : GpuBuffer,
  src : FixedArray[Float],
  count : Int64,
) -> Int = "moonbit_cuda_copy_h2d"

///|
#borrow(dst, src)
extern "c" fn moonbit_cuda_raw_h2d(
  dst : GpuBuffer,
  src : FixedArray[Int],
  bytes : Int64,
) -> Int = "moonbit_cuda_raw_h2d"

///|
#borrow(dst, src)
extern "c" fn moonbit_cuda_copy_ptrs_h2d(
  dst : GpuBuffer,
  src : FixedArray[GpuBuffer],
  count : Int,
) -> Int = "moonbit_cuda_copy_ptrs_h2d"

///|
#borrow(dst, src)
extern "c" fn moonbit_cuda_copy_d2h(
  dst : FixedArray[Float],
  src : GpuBuffer,
  count : Int64,
) -> Int = "moonbit_cuda_copy_d2h"

///|
#borrow(dst, src_base)
extern "c" fn moonbit_cuda_copy_h2d_offset(
  dst : GpuBuffer,
  src_base : FixedArray[Float],
  offset_elements : Int,
  count_elements : Int,
) -> Int = "moonbit_cuda_copy_h2d_offset"

///|
#borrow(dst)
extern "c" fn moonbit_cuda_memset_zero(dst : GpuBuffer, count : Int64) -> Int = "moonbit_cuda_memset_zero"

// === Vector Add ===

///|
extern "c" fn moonbit_cuda_vec_add(
  a : GpuBuffer,
  b : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_vec_add"

// === cuBLAS ===

///|
extern "c" fn moonbit_cublas_create() -> CublasHandle = "moonbit_cublas_create"

///|
extern "c" fn moonbit_cublas_destroy(handle : CublasHandle) -> Unit = "moonbit_cublas_destroy"

///|
#borrow(handle, d_A, d_x, d_y)
extern "c" fn moonbit_cublas_sgemv(
  handle : CublasHandle,
  m : Int,
  n : Int,
  alpha : Float,
  d_A : GpuBuffer,
  d_x : GpuBuffer,
  beta : Float,
  d_y : GpuBuffer,
) -> Int = "moonbit_cublas_sgemv"

///|
#borrow(handle, d_A, d_B, d_C)
extern "c" fn moonbit_cublas_sgemm(
  handle : CublasHandle,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  d_A : GpuBuffer,
  d_B : GpuBuffer,
  beta : Float,
  d_C : GpuBuffer,
) -> Int = "moonbit_cublas_sgemm"

// === cuDNN Handle ===

///|
extern "c" fn moonbit_cudnn_create() -> CudnnHandle = "moonbit_cudnn_create"

///|
extern "c" fn moonbit_cudnn_destroy(handle : CudnnHandle) -> Unit = "moonbit_cudnn_destroy"

// === Conv2d ===

///|
extern "c" fn moonbit_conv2d_create(
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
) -> Conv2dContext = "moonbit_conv2d_create"

///|
#borrow(ctx, cudnn_handle, input, weight, bias, output)
extern "c" fn moonbit_conv2d_forward(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  weight : GpuBuffer,
  bias : GpuBuffer,
  output : GpuBuffer,
) -> Int = "moonbit_conv2d_forward"

///|
#borrow(ctx, cudnn_handle, input, weight, bias, output)
extern "c" fn moonbit_conv2d_forward_bias_relu(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  weight : GpuBuffer,
  bias : GpuBuffer,
  output : GpuBuffer,
) -> Int = "moonbit_conv2d_forward_bias_relu"

///|
#borrow(ctx, cudnn_handle, grad_output, weight, grad_input)
extern "c" fn moonbit_conv2d_backward_data(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  grad_output : GpuBuffer,
  weight : GpuBuffer,
  grad_input : GpuBuffer,
) -> Int = "moonbit_conv2d_backward_data"

///|
#borrow(ctx, cudnn_handle, input, grad_output, grad_weight)
extern "c" fn moonbit_conv2d_backward_filter(
  ctx : Conv2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  grad_output : GpuBuffer,
  grad_weight : GpuBuffer,
) -> Int = "moonbit_conv2d_backward_filter"

///|
#borrow(cudnn_handle, grad_output, grad_bias)
extern "c" fn moonbit_conv2d_backward_bias(
  cudnn_handle : CudnnHandle,
  grad_output : GpuBuffer,
  grad_bias : GpuBuffer,
  batch : Int,
  out_c : Int,
  out_h : Int,
  out_w : Int,
) -> Int = "moonbit_conv2d_backward_bias"

///|
#borrow(ctx)
extern "c" fn moonbit_conv2d_destroy(ctx : Conv2dContext) -> Unit = "moonbit_conv2d_destroy"

// === BatchNorm ===

///|
extern "c" fn moonbit_batchnorm_create(
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> BatchNormContext = "moonbit_batchnorm_create"

///|
#borrow(ctx, cudnn_handle, input, output, bn_scale, bn_bias, bn_running_mean, bn_running_var, bn_save_mean, bn_save_inv_var)
extern "c" fn moonbit_batchnorm_forward(
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
  is_training : Int,
) -> Int = "moonbit_batchnorm_forward"

///|
#borrow(input, output, gamma, beta, running_mean, running_var)
extern "c" fn moonbit_cuda_batchnorm_inference(
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
) -> Int = "moonbit_cuda_batchnorm_inference"

///|
extern "c" fn moonbit_batchnorm_backward(
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
) -> Int = "moonbit_batchnorm_backward"

///|
#borrow(ctx)
extern "c" fn moonbit_batchnorm_destroy(ctx : BatchNormContext) -> Unit = "moonbit_batchnorm_destroy"

// === ReLU via cuDNN ===

///|
#borrow(handle, input, output)
extern "c" fn moonbit_cudnn_relu_forward(
  handle : CudnnHandle,
  n_elements : Int64,
  input : GpuBuffer,
  output : GpuBuffer,
) -> Int = "moonbit_cudnn_relu_forward"

///|
#borrow(handle, input, grad_output, grad_input)
extern "c" fn moonbit_cudnn_relu_backward(
  handle : CudnnHandle,
  n_elements : Int64,
  input : GpuBuffer,
  grad_output : GpuBuffer,
  grad_input : GpuBuffer,
) -> Int = "moonbit_cudnn_relu_backward"

// === Pooling ===

///|
extern "c" fn moonbit_pool2d_create(
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
) -> Pool2dContext = "moonbit_pool2d_create"

///|
#borrow(ctx, cudnn_handle, input, output)
extern "c" fn moonbit_pool2d_forward(
  ctx : Pool2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  output : GpuBuffer,
) -> Int = "moonbit_pool2d_forward"

///|
#borrow(ctx, cudnn_handle, input, grad_output, grad_input)
extern "c" fn moonbit_pool2d_backward(
  ctx : Pool2dContext,
  cudnn_handle : CudnnHandle,
  input : GpuBuffer,
  grad_output : GpuBuffer,
  grad_input : GpuBuffer,
) -> Int = "moonbit_pool2d_backward"

///|
#borrow(ctx)
extern "c" fn moonbit_pool2d_destroy(ctx : Pool2dContext) -> Unit = "moonbit_pool2d_destroy"

// === Cache Management ===

///|
extern "c" fn moonbit_conv2d_cache_clear() -> Unit = "moonbit_conv2d_cache_clear"

///|
extern "c" fn moonbit_conv2d_cache_size() -> Int = "moonbit_conv2d_cache_size"

///|
extern "c" fn moonbit_batchnorm_cache_clear() -> Unit = "moonbit_batchnorm_cache_clear"

///|
extern "c" fn moonbit_batchnorm_cache_size() -> Int = "moonbit_batchnorm_cache_size"

///|
extern "c" fn moonbit_pool2d_cache_clear() -> Unit = "moonbit_pool2d_cache_clear"

///|
extern "c" fn moonbit_pool2d_cache_size() -> Int = "moonbit_pool2d_cache_size"

// === GPU Memory Pool ===

///|
extern "c" fn moonbit_pool_create(capacity : Int64) -> GpuPool = "moonbit_pool_create"

///|
extern "c" fn moonbit_pool_alloc(pool : GpuPool, size : Int64) -> GpuBuffer = "moonbit_pool_alloc"

///|
extern "c" fn moonbit_pool_reset(pool : GpuPool) -> Unit = "moonbit_pool_reset"

///|
extern "c" fn moonbit_pool_used(pool : GpuPool) -> Int64 = "moonbit_pool_used"

///|
extern "c" fn moonbit_pool_peak(pool : GpuPool) -> Int64 = "moonbit_pool_peak"

///|
extern "c" fn moonbit_pool_capacity(pool : GpuPool) -> Int64 = "moonbit_pool_capacity"

// === Transpose and elementwise kernels ===

///|
#borrow(src, dst)
extern "c" fn moonbit_cuda_transpose_nchw_to_nhwc(
  src : GpuBuffer,
  dst : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int = "moonbit_cuda_transpose_nchw_to_nhwc"

///|
#borrow(src, dst)
extern "c" fn moonbit_cuda_transpose_nhwc_to_nchw(
  src : GpuBuffer,
  dst : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int = "moonbit_cuda_transpose_nhwc_to_nchw"

///|
#borrow(data, bias)
extern "c" fn moonbit_cuda_bias_add_nchw(
  data : GpuBuffer,
  bias : GpuBuffer,
  n : Int,
  c : Int,
  hw : Int,
) -> Int = "moonbit_cuda_bias_add_nchw"

///|
#borrow(a, b, out)
extern "c" fn moonbit_cuda_elementwise_add(
  a : GpuBuffer,
  b : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_elementwise_add"

///|
#borrow(dst, src)
extern "c" fn moonbit_cuda_elementwise_add_into(
  dst : GpuBuffer,
  src : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_elementwise_add_into"

// === Reduce Sum Spatial ===

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_reduce_sum_spatial(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int = "moonbit_cuda_reduce_sum_spatial"

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_reduce_sum_leading(
  input : GpuBuffer,
  output : GpuBuffer,
  output_numel : Int,
  lead_stride : Int,
) -> Int = "moonbit_cuda_reduce_sum_leading"

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_reduce_sum_dim(
  input : GpuBuffer,
  output : GpuBuffer,
  output_numel : Int,
  dim_size : Int,
  stride_before : Int,
  stride_after : Int,
) -> Int = "moonbit_cuda_reduce_sum_dim"

// === Adaptive Avg Pool2d ===

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_adaptive_avg_pool2d_forward(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int = "moonbit_cuda_adaptive_avg_pool2d_forward"

///|
#borrow(dy, d_input)
extern "c" fn moonbit_cuda_adaptive_avg_pool2d_backward(
  dy : GpuBuffer,
  d_input : GpuBuffer,
  n : Int,
  c : Int,
  h : Int,
  w : Int,
) -> Int = "moonbit_cuda_adaptive_avg_pool2d_backward"

// === Softmax CE ===

///|
#borrow(logits, targets, loss_output, softmax_output)
extern "c" fn moonbit_cuda_softmax_ce_forward(
  logits : GpuBuffer,
  targets : GpuBuffer,
  loss_output : GpuBuffer,
  softmax_output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int = "moonbit_cuda_softmax_ce_forward"

///|
#borrow(logits, targets, output)
extern "c" fn moonbit_cuda_softmax_ce_backward(
  logits : GpuBuffer,
  targets : GpuBuffer,
  output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int = "moonbit_cuda_softmax_ce_backward"

// === Softmax CE with integer labels ===

///|
#borrow(logits, labels, loss_output)
extern "c" fn moonbit_cuda_softmax_ce_forward_labels(
  logits : GpuBuffer,
  labels : GpuBuffer,
  loss_output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int = "moonbit_cuda_softmax_ce_forward_labels"

///|
#borrow(logits, labels, output)
extern "c" fn moonbit_cuda_softmax_ce_backward_labels(
  logits : GpuBuffer,
  labels : GpuBuffer,
  output : GpuBuffer,
  batch : Int,
  num_classes : Int,
) -> Int = "moonbit_cuda_softmax_ce_backward_labels"

// === Matmul ===

///|
#borrow(handle, a, b, c)
extern "c" fn moonbit_cuda_matmul(
  handle : CublasHandle,
  a : GpuBuffer,
  b : GpuBuffer,
  c : GpuBuffer,
  m : Int,
  k : Int,
  n : Int,
  transpose_a : Int,
  transpose_b : Int,
  alpha : Float,
  beta : Float,
) -> Int = "moonbit_cuda_matmul"

///|
#borrow(dst, src)
extern "c" fn moonbit_cuda_accumulate_inplace(
  dst : GpuBuffer,
  src : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_accumulate_inplace"

// === GPU Optimizer Kernels ===

///|
#borrow(param, grad, velocity)
extern "c" fn moonbit_cuda_sgd_momentum_step(
  param : GpuBuffer,
  grad : GpuBuffer,
  velocity : GpuBuffer,
  n : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int = "moonbit_cuda_sgd_momentum_step"

///|
#borrow(param, grad)
extern "c" fn moonbit_cuda_sgd_step(
  param : GpuBuffer,
  grad : GpuBuffer,
  n : Int,
  lr : Float,
  weight_decay : Float,
) -> Int = "moonbit_cuda_sgd_step"

///|
#borrow(src, dst)
extern "c" fn moonbit_cuda_norm_sq(
  src : GpuBuffer,
  n : Int,
  dst : GpuBuffer,
) -> Int = "moonbit_cuda_norm_sq"

///|
#borrow(data)
extern "c" fn moonbit_cuda_scale_inplace(
  data : GpuBuffer,
  n : Int,
  scale : Float,
) -> Int = "moonbit_cuda_scale_inplace"

///|
#borrow(out)
extern "c" fn moonbit_cuda_fill(out : GpuBuffer, n : Int, value : Float) -> Int = "moonbit_cuda_fill"

///|
extern "c" fn moonbit_cuda_read_scalar(src : GpuBuffer) -> Float = "moonbit_cuda_read_scalar"

///|
#borrow(x, y)
extern "c" fn moonbit_cuda_axpy(
  n : Int,
  alpha : Float,
  x : GpuBuffer,
  y : GpuBuffer,
) -> Int = "moonbit_cuda_axpy"

///|
extern "c" fn moonbit_get_time_ms() -> Double = "moonbit_get_time_ms"

///|
extern "c" fn moonbit_cuda_sync() -> Unit = "moonbit_cuda_sync"

// === Elementwise GPU kernels ===

///|
#borrow(a, b, out)
extern "c" fn moonbit_cuda_sub(
  a : GpuBuffer,
  b : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_sub"

///|
#borrow(a, b, out)
extern "c" fn moonbit_cuda_mul_elem(
  a : GpuBuffer,
  b : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_mul_elem"

///|
#borrow(src, out)
extern "c" fn moonbit_cuda_square(
  src : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_square"

///|
#borrow(src, out)
extern "c" fn moonbit_cuda_sqrt(
  src : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_sqrt"

///|
#borrow(a, b, out)
extern "c" fn moonbit_cuda_div_elem(
  a : GpuBuffer,
  b : GpuBuffer,
  out : GpuBuffer,
  n : Int,
) -> Int = "moonbit_cuda_div_elem"

///|
#borrow(src, out)
extern "c" fn moonbit_cuda_scale_out(
  src : GpuBuffer,
  out : GpuBuffer,
  n : Int,
  scale : Float,
) -> Int = "moonbit_cuda_scale_out"

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_transpose_2d(
  input : GpuBuffer,
  output : GpuBuffer,
  n : Int,
  m : Int,
) -> Int = "moonbit_cuda_transpose_2d"

///|
#borrow(ptrs)
pub extern "c" fn moonbit_cuda_test_ptr_array_read(
  ptrs : FixedArray[GpuBuffer],
  n : Int,
) -> Float = "moonbit_cuda_test_ptr_array_read"

///|
#borrow(tensor_ptrs, tensor_sizes, dst)
extern "c" fn moonbit_cuda_multi_tensor_norm_sq(
  tensor_ptrs : FixedArray[GpuBuffer],
  tensor_sizes : FixedArray[Int],
  num_tensors : Int,
  dst : GpuBuffer,
) -> Int = "moonbit_cuda_multi_tensor_norm_sq"

///|
#borrow(d_ptrs, d_sizes, dst)
extern "c" fn moonbit_cuda_multi_tensor_norm_sq_gpu_only(
  d_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
  dst : GpuBuffer,
) -> Int = "moonbit_cuda_multi_tensor_norm_sq_gpu_only"

///|
#borrow(d_ptrs, d_sizes, d_block_offsets, dst)
extern "c" fn moonbit_cuda_mt_norm_sq_gpu_only(
  d_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  d_block_offsets : GpuBuffer,
  num_tensors : Int,
  total_blocks : Int,
  dst : GpuBuffer,
) -> Int = "moonbit_cuda_mt_norm_sq_gpu_only"

///|
#borrow(d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes, d_block_offsets)
extern "c" fn moonbit_cuda_mt_sgd_momentum_gpu_only(
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
) -> Int = "moonbit_cuda_mt_sgd_momentum_gpu_only"

///|
#borrow(d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes, d_block_offsets, norm_buf)
extern "c" fn moonbit_cuda_mt_sgd_momentum_autoclip(
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
) -> Int = "moonbit_cuda_mt_sgd_momentum_autoclip"

///|
#borrow(param_ptrs, grad_ptrs, velocity_ptrs, tensor_sizes)
extern "c" fn moonbit_cuda_multi_tensor_sgd_momentum_step(
  param_ptrs : FixedArray[GpuBuffer],
  grad_ptrs : FixedArray[GpuBuffer],
  velocity_ptrs : FixedArray[GpuBuffer],
  tensor_sizes : FixedArray[Int],
  num_tensors : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int = "moonbit_cuda_multi_tensor_sgd_momentum_step"

///|
#borrow(param_ptrs, grad_ptrs, velocity_ptrs, tensor_sizes, d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes)
extern "c" fn moonbit_cuda_multi_tensor_sgd_momentum_step_prealloc(
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
) -> Int = "moonbit_cuda_multi_tensor_sgd_momentum_step_prealloc"

///|
#borrow(d_param_ptrs, d_grad_ptrs, d_vel_ptrs, d_sizes)
extern "c" fn moonbit_cuda_multi_tensor_sgd_momentum_step_gpu_only(
  d_param_ptrs : GpuBuffer,
  d_grad_ptrs : GpuBuffer,
  d_vel_ptrs : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int = "moonbit_cuda_multi_tensor_sgd_momentum_step_gpu_only"

///|
#borrow(params, grads, vels)
extern "c" fn moonbit_cuda_contiguous_sgd_momentum_step(
  params : GpuBuffer,
  grads : GpuBuffer,
  vels : GpuBuffer,
  total_elements : Int,
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  clip_coef : Float,
) -> Int = "moonbit_cuda_contiguous_sgd_momentum_step"

///|
#borrow(d_src_ptrs, d_offsets, d_sizes, dst)
extern "c" fn moonbit_cuda_gather(
  d_src_ptrs : GpuBuffer,
  d_offsets : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
  dst : GpuBuffer,
) -> Int = "moonbit_cuda_gather"

///|
#borrow(src, d_dst_ptrs, d_offsets, d_sizes)
extern "c" fn moonbit_cuda_scatter(
  src : GpuBuffer,
  d_dst_ptrs : GpuBuffer,
  d_offsets : GpuBuffer,
  d_sizes : GpuBuffer,
  num_tensors : Int,
) -> Int = "moonbit_cuda_scatter"

///|
#borrow(data, out)
extern "c" fn moonbit_cuda_contiguous_norm_sq(
  data : GpuBuffer,
  n : Int,
  out : GpuBuffer,
) -> Int = "moonbit_cuda_contiguous_norm_sq"

///|
#borrow(src, indices, dst)
extern "c" fn moonbit_cuda_batch_gather(
  src : GpuBuffer,
  indices : GpuBuffer,
  dst : GpuBuffer,
  stride : Int,
  batch_size : Int,
) -> Int = "moonbit_cuda_batch_gather"

// === CUDA Events ===

///|
extern "c" fn moonbit_cuda_event_create() -> CudaEvent = "moonbit_cuda_event_create"

///|
extern "c" fn moonbit_cuda_null_buffer() -> GpuBuffer = "moonbit_cuda_null_buffer"

///|
#borrow(event)
extern "c" fn moonbit_cuda_event_record(
  stream : GpuBuffer,
  event : CudaEvent,
) -> Int = "moonbit_cuda_event_record"

///|
#borrow(start, end)
extern "c" fn moonbit_cuda_event_elapsed_ms(
  start : CudaEvent,
  end : CudaEvent,
) -> Float = "moonbit_cuda_event_elapsed_ms"

///|
#borrow(event)
extern "c" fn moonbit_cuda_event_synchronize(event : CudaEvent) -> Int = "moonbit_cuda_event_synchronize"

///|
#borrow(input, output)
extern "c" fn moonbit_cuda_argmax(
  input : GpuBuffer,
  output : GpuBuffer,
  outer : Int,
  inner : Int,
) -> Int = "moonbit_cuda_argmax"

///|
#borrow(d_param_ptrs, d_grad_ptrs, d_m_ptrs, d_v_ptrs, d_sizes, d_block_offsets, norm_buf)
extern "c" fn moonbit_cuda_mt_adam_autoclip(
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
) -> Int = "moonbit_cuda_mt_adam_autoclip"

///|
#borrow(d_param_ptrs, d_grad_ptrs, d_v_ptrs, d_buf_ptrs, d_sizes, d_block_offsets, norm_buf)
extern "c" fn moonbit_cuda_mt_rmsprop_autoclip(
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
) -> Int = "moonbit_cuda_mt_rmsprop_autoclip"
