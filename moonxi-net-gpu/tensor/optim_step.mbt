// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// =============================================================================
// GPU optimizer utilities — ContiguousOptimState, MultiTensorOptimState
// Used by optim_substep_bench for benchmarking different GPU optimization strategies.
// The actual MomentumSGD trait impl lives in optimizer/momentum_sgd.mbt.
// =============================================================================

///|
pub(all) struct ContiguousOptimState {
  // Flat GPU buffers
  flat_params : @cuda.GpuBuffer
  flat_grads : @cuda.GpuBuffer
  flat_vels : @cuda.GpuBuffer
  norm_buf : @cuda.GpuBuffer
  // GPU-side pointer arrays (pre-populated)
  d_param_ptrs : @cuda.GpuBuffer
  d_grad_ptrs : @cuda.GpuBuffer
  d_vel_ptrs : @cuda.GpuBuffer
  d_sizes : @cuda.GpuBuffer
  d_offsets : @cuda.GpuBuffer
  // Metadata
  param_ptrs : FixedArray[@cuda.GpuBuffer]
  vel_ptrs : FixedArray[@cuda.GpuBuffer]
  sizes : FixedArray[Int]
  offsets : FixedArray[Int]
  total_elements : Int
  num_tensors : Int
}

///|
/// Build a ContiguousOptimState from params and velocities.
pub fn contiguous_state_new(
  params : Array[GpuTensor],
  velocities : Array[GpuTensor],
) -> ContiguousOptimState {
  let n = params.length()
  let sizes = FixedArray::makei(n, fn(i) { params[i].numel() })
  let offsets = FixedArray::makei(
    n,
    {
      let mut off = 0
      fn(i) {
        let prev = off
        off = off + sizes[i]
        prev
      }
    },
  )
  let total = {
    let mut s = 0
    for i in 0..<n {
      s = s + sizes[i]
    }
    s
  }

  // Flat buffers
  let flat_params = @cuda.alloc(Int64::from_int(total * 4))
  let flat_grads = @cuda.alloc(Int64::from_int(total * 4))
  let flat_vels = @cuda.alloc(Int64::from_int(total * 4))
  let norm_buf = @cuda.alloc(4L)
  check_cuda(
    @cuda.memset_zero(norm_buf, 1L),
    "memset_zero in contiguous_state_new",
  )

  // Host pointer arrays
  let param_ptrs = FixedArray::makei(n, fn(i) { params[i].data })
  let vel_ptrs = FixedArray::makei(n, fn(i) { velocities[i].data })

  // GPU buffers for pointer arrays and metadata
  let ptrs_bytes = Int64::from_int(n * 8)
  let d_param_ptrs = @cuda.alloc(ptrs_bytes)
  let d_grad_ptrs = @cuda.alloc(ptrs_bytes)
  let d_vel_ptrs = @cuda.alloc(ptrs_bytes)
  let d_sizes = @cuda.alloc(Int64::from_int(n * 4))
  let d_offsets = @cuda.alloc(Int64::from_int(n * 4))

  // Copy pointer arrays to GPU
  check_cuda(
    @cuda.copy_ptrs_h2d(d_param_ptrs, param_ptrs),
    "copy_ptrs_h2d param_ptrs",
  )
  check_cuda(
    @cuda.copy_ptrs_h2d(d_vel_ptrs, vel_ptrs),
    "copy_ptrs_h2d vel_ptrs",
  )

  // Copy int metadata to GPU
  check_cuda(@cuda.raw_h2d(d_sizes, sizes), "raw_h2d sizes")
  check_cuda(@cuda.raw_h2d(d_offsets, offsets), "raw_h2d offsets")

  // Initial gather: params and vels into flat buffers
  check_cuda(
    @cuda.gather(d_param_ptrs, d_offsets, d_sizes, n, flat_params),
    "gather params",
  )
  check_cuda(
    @cuda.gather(d_vel_ptrs, d_offsets, d_sizes, n, flat_vels),
    "gather vels",
  )

  {
    flat_params,
    flat_grads,
    flat_vels,
    norm_buf,
    d_param_ptrs,
    d_grad_ptrs,
    d_vel_ptrs,
    d_sizes,
    d_offsets,
    param_ptrs,
    vel_ptrs,
    sizes,
    offsets,
    total_elements: total,
    num_tensors: n,
  }
}

///|
pub fn contiguous_sgd_step(
  state : ContiguousOptimState,
  grads : Array[GpuTensor],
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  max_grad_norm : Float,
) -> Unit {
  let n = state.num_tensors
  // 1. Copy grad pointers to GPU
  let g_ptrs = FixedArray::makei(n, fn(i) { grads[i].data })
  check_cuda(
    @cuda.copy_ptrs_h2d(state.d_grad_ptrs, g_ptrs),
    "copy_ptrs_h2d grad_ptrs",
  )
  // 2. Gather grads into flat buffer
  check_cuda(
    @cuda.gather(
      state.d_grad_ptrs,
      state.d_offsets,
      state.d_sizes,
      n,
      state.flat_grads,
    ),
    "gather grads",
  )
  // 3. Reset norm, compute flat norm_sq, read
  check_cuda(@cuda.memset_zero(state.norm_buf, 1L), "memset_zero norm")
  check_cuda(
    @cuda.contiguous_norm_sq(
      state.flat_grads,
      state.total_elements,
      state.norm_buf,
    ),
    "contiguous_norm_sq",
  )
  let total_norm = Float::sqrt(@cuda.read_scalar(state.norm_buf))
  let clip_coef = if total_norm > max_grad_norm {
    max_grad_norm / total_norm
  } else {
    Float::from_double(1.0)
  }
  // 4. Flat SGD+momentum
  check_cuda(
    @cuda.contiguous_sgd_momentum_step(
      state.flat_params,
      state.flat_grads,
      state.flat_vels,
      state.total_elements,
      lr,
      momentum,
      weight_decay,
      clip_coef,
    ),
    "contiguous_sgd_momentum_step",
  )
  // 5. Scatter params back to individual tensors
  check_cuda(
    @cuda.scatter(
      state.flat_params,
      state.d_param_ptrs,
      state.d_offsets,
      state.d_sizes,
      n,
    ),
    "scatter params",
  )
}

///|
/// Multi-tensor optimizer state — zero-alloc per step, high GPU occupancy.
/// Pre-allocates GPU pointer arrays for params, velocities, sizes.
/// Only grad pointers are updated each step (one small H2D copy).
/// Uses multi-block-per-tensor kernels for full GPU saturation.
pub(all) struct MultiTensorOptimState {
  // GPU-side pointer arrays (pre-populated, immutable across steps)
  d_param_ptrs : @cuda.GpuBuffer
  d_grad_ptrs : @cuda.GpuBuffer
  d_vel_ptrs : @cuda.GpuBuffer
  d_sizes : @cuda.GpuBuffer
  d_block_offsets : @cuda.GpuBuffer
  // Norm accumulation buffer
  norm_buf : @cuda.GpuBuffer
  // Host-side grad pointer array (reused each step)
  grad_ptrs : FixedArray[@cuda.GpuBuffer]
  // Metadata
  num_tensors : Int
  total_blocks : Int
}

///|
/// Build a MultiTensorOptimState from params and velocities.
/// Pre-computes block offsets for multi-block-per-tensor kernels.
pub fn multi_tensor_state_new(
  params : Array[GpuTensor],
  velocities : Array[GpuTensor],
) -> MultiTensorOptimState {
  let n = params.length()
  let p_ptrs = FixedArray::makei(n, fn(i) { params[i].data })
  let v_ptrs = FixedArray::makei(n, fn(i) { velocities[i].data })
  let sizes = FixedArray::makei(n, fn(i) { params[i].numel() })
  let grad_ptrs = FixedArray::makei(n, fn(_) { params[0].data }) // placeholder

  // Compute block_offsets: prefix-sum of ceil(size/256) per tensor
  let block_offsets = FixedArray::makei(
    n,
    {
      let mut off = 0
      fn(i) {
        let prev = off
        off = off + (sizes[i] + 255) / 256
        prev
      }
    },
  )
  let total_blocks = {
    let mut s = 0
    for i in 0..<n {
      s = s + (sizes[i] + 255) / 256
    }
    s
  }

  let ptrs_bytes = Int64::from_int(n * 8)
  let sizes_bytes = Int64::from_int(n * 4)

  let d_param_ptrs = @cuda.alloc(ptrs_bytes)
  let d_grad_ptrs = @cuda.alloc(ptrs_bytes)
  let d_vel_ptrs = @cuda.alloc(ptrs_bytes)
  let d_sizes = @cuda.alloc(sizes_bytes)
  let d_block_offsets = @cuda.alloc(sizes_bytes)
  let norm_buf = @cuda.alloc(4L)
  check_cuda(
    @cuda.memset_zero(norm_buf, 1L),
    "memset_zero in multi_tensor_state_new",
  )

  // Copy immutable pointer arrays to GPU once
  check_cuda(@cuda.copy_ptrs_h2d(d_param_ptrs, p_ptrs), "copy_ptrs_h2d p_ptrs")
  check_cuda(@cuda.copy_ptrs_h2d(d_vel_ptrs, v_ptrs), "copy_ptrs_h2d v_ptrs")
  check_cuda(@cuda.raw_h2d(d_sizes, sizes), "raw_h2d sizes")
  check_cuda(
    @cuda.raw_h2d(d_block_offsets, block_offsets),
    "raw_h2d block_offsets",
  )

  {
    d_param_ptrs,
    d_grad_ptrs,
    d_vel_ptrs,
    d_sizes,
    d_block_offsets,
    norm_buf,
    grad_ptrs,
    num_tensors: n,
    total_blocks,
  }
}

///|
/// Zero-alloc multi-tensor SGD+momentum step with gradient clipping.
/// Fully async: no D2H sync. Norm computation and clipping happen on GPU.
/// Per-step cost: 1 small H2D copy (grad ptrs) + 2 GPU kernels (norm + momentum step).
/// Uses multi-block-per-tensor kernels for full GPU occupancy (~43K blocks).
pub fn multi_tensor_sgd_step(
  state : MultiTensorOptimState,
  grads : Array[GpuTensor],
  lr : Float,
  momentum : Float,
  weight_decay : Float,
  max_grad_norm : Float,
) -> Unit {
  let n = state.num_tensors
  // 1. Update grad pointers and copy to GPU (one small H2D)
  for i in 0..<n {
    state.grad_ptrs[i] = grads[i].data
  }
  check_cuda(
    @cuda.copy_ptrs_h2d(state.d_grad_ptrs, state.grad_ptrs),
    "copy_ptrs_h2d grad_ptrs",
  )
  // 2. Reset norm, compute multi-tensor norm_sq (multi-block)
  check_cuda(@cuda.memset_zero(state.norm_buf, 1L), "memset_zero norm")
  check_cuda(
    @cuda.mt_norm_sq_gpu_only(
      state.d_grad_ptrs,
      state.d_sizes,
      state.d_block_offsets,
      n,
      state.total_blocks,
      state.norm_buf,
    ),
    "mt_norm_sq_gpu_only",
  )
  // 3. SGD+momentum with autoclip — reads norm_buf on GPU, no D2H sync
  check_cuda(
    @cuda.mt_sgd_momentum_autoclip(
      state.d_param_ptrs,
      state.d_grad_ptrs,
      state.d_vel_ptrs,
      state.d_sizes,
      state.d_block_offsets,
      n,
      state.total_blocks,
      lr,
      momentum,
      weight_decay,
      state.norm_buf,
      max_grad_norm,
    ),
    "mt_sgd_momentum_autoclip",
  )
}
