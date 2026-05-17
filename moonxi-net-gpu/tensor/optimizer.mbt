// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// =============================================================================
// GpuTensor optimizer backend — fused CUDA kernels, zero D2H sync
// =============================================================================

///|
/// Compute prefix-sum block offsets for multi-block-per-tensor kernels.
fn compute_block_offsets(sizes : FixedArray[Int], n : Int) -> FixedArray[Int] {
  FixedArray::makei(
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
}

///|
/// Cached optimizer state — pre-allocated device buffers.
/// Params, velocities, sizes, and offsets never change between steps,
/// so we upload them once and reuse. Only grad pointers change (workspace reset).
priv struct OptCache {
  mut d_p : @cuda.GpuBuffer
  mut d_g : @cuda.GpuBuffer
  mut d_v : @cuda.GpuBuffer
  mut d_sizes : @cuda.GpuBuffer
  mut d_offsets : @cuda.GpuBuffer
  mut norm_buf : @cuda.GpuBuffer
  mut cached_n : Int
  mut total_blocks : Int
}

///|
/// Cached Adam optimizer state — pre-allocated device buffers.
priv struct OptAdamCache {
  mut d_p : @cuda.GpuBuffer
  mut d_g : @cuda.GpuBuffer
  mut d_m : @cuda.GpuBuffer
  mut d_v : @cuda.GpuBuffer
  mut d_sizes : @cuda.GpuBuffer
  mut d_offsets : @cuda.GpuBuffer
  mut norm_buf : @cuda.GpuBuffer
  mut cached_n : Int
  mut total_blocks : Int
}

///|
/// Cached RMSprop optimizer state — pre-allocated device buffers.
priv struct OptRmspropCache {
  mut d_p : @cuda.GpuBuffer
  mut d_g : @cuda.GpuBuffer
  mut d_v : @cuda.GpuBuffer
  mut d_buf : @cuda.GpuBuffer
  mut d_sizes : @cuda.GpuBuffer
  mut d_offsets : @cuda.GpuBuffer
  mut norm_buf : @cuda.GpuBuffer
  mut cached_n : Int
  mut total_blocks : Int
}

///|
let opt_cache : OptCache = {
  d_p: @cuda.null_buffer(),
  d_g: @cuda.null_buffer(),
  d_v: @cuda.null_buffer(),
  d_sizes: @cuda.null_buffer(),
  d_offsets: @cuda.null_buffer(),
  norm_buf: @cuda.null_buffer(),
  cached_n: 0,
  total_blocks: 0,
}

///|
let opt_adam_cache : OptAdamCache = {
  d_p: @cuda.null_buffer(),
  d_g: @cuda.null_buffer(),
  d_m: @cuda.null_buffer(),
  d_v: @cuda.null_buffer(),
  d_sizes: @cuda.null_buffer(),
  d_offsets: @cuda.null_buffer(),
  norm_buf: @cuda.null_buffer(),
  cached_n: 0,
  total_blocks: 0,
}

///|
let opt_rmsprop_cache : OptRmspropCache = {
  d_p: @cuda.null_buffer(),
  d_g: @cuda.null_buffer(),
  d_v: @cuda.null_buffer(),
  d_buf: @cuda.null_buffer(),
  d_sizes: @cuda.null_buffer(),
  d_offsets: @cuda.null_buffer(),
  norm_buf: @cuda.null_buffer(),
  cached_n: 0,
  total_blocks: 0,
}

///|
pub impl @optimizer.BatchSgdMomentumAutoclip for GpuTensor with batch_sgd_momentum_autoclip(
  params : Array[GpuTensor],
  grads : Array[GpuTensor],
  velocities : Array[GpuTensor],
  config : @optimizer.SgdMomentumConfig,
) -> Unit {
  let { lr, momentum, weight_decay, max_grad_norm } = config
  let n = params.length()
  let ptrs_bytes = Int64::from_int(n * 8)
  let sizes_bytes = Int64::from_int(n * 4)

  // Grad pointers change every step (workspace-allocated, reset between batches)
  let g_ptrs = FixedArray::makei(n, fn(i) { grads[i].data })

  if n != opt_cache.cached_n {
    // First call or param count changed — allocate and upload static buffers
    let p_ptrs = FixedArray::makei(n, fn(i) { params[i].data })
    let v_ptrs = FixedArray::makei(n, fn(i) { velocities[i].data })
    let sizes = FixedArray::makei(n, fn(i) { params[i].numel() })
    let block_offsets = compute_block_offsets(sizes, n)
    let mut tb = 0
    for i in 0..<n {
      tb = tb + (sizes[i] + 255) / 256
    }

    opt_cache.d_p = @cuda.alloc(ptrs_bytes)
    opt_cache.d_g = @cuda.alloc(ptrs_bytes)
    opt_cache.d_v = @cuda.alloc(ptrs_bytes)
    opt_cache.d_sizes = @cuda.alloc(sizes_bytes)
    opt_cache.d_offsets = @cuda.alloc(sizes_bytes)
    opt_cache.norm_buf = @cuda.alloc(4L)

    check_cuda(@cuda.copy_ptrs_h2d(opt_cache.d_p, p_ptrs), "copy_ptrs_h2d")
    check_cuda(@cuda.copy_ptrs_h2d(opt_cache.d_v, v_ptrs), "copy_ptrs_h2d")
    check_cuda(@cuda.raw_h2d(opt_cache.d_sizes, sizes), "raw_h2d")
    check_cuda(@cuda.raw_h2d(opt_cache.d_offsets, block_offsets), "raw_h2d")

    opt_cache.cached_n = n
    opt_cache.total_blocks = tb
  }

  // Per-step: only upload grad pointers and reset norm
  check_cuda(@cuda.copy_ptrs_h2d(opt_cache.d_g, g_ptrs), "copy_ptrs_h2d")
  check_cuda(@cuda.memset_zero(opt_cache.norm_buf, 1L), "memset_zero")
  check_cuda(
    @cuda.mt_norm_sq_gpu_only(
      opt_cache.d_g,
      opt_cache.d_sizes,
      opt_cache.d_offsets,
      n,
      opt_cache.total_blocks,
      opt_cache.norm_buf,
    ),
    "mt_norm_sq_gpu_only",
  )
  check_cuda(
    @cuda.mt_sgd_momentum_autoclip(
      opt_cache.d_p,
      opt_cache.d_g,
      opt_cache.d_v,
      opt_cache.d_sizes,
      opt_cache.d_offsets,
      n,
      opt_cache.total_blocks,
      lr,
      momentum,
      weight_decay,
      opt_cache.norm_buf,
      max_grad_norm,
    ),
    "mt_sgd_momentum_autoclip",
  )
}

///|
pub impl @optimizer.BatchAdamAutoclip for GpuTensor with batch_adam_autoclip(
  params : Array[GpuTensor],
  grads : Array[GpuTensor],
  m : Array[GpuTensor],
  v : Array[GpuTensor],
  config : @optimizer.AdamConfig,
) -> Unit {
  let {
    lr,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    bias_correction1,
    bias_correction2,
    eps,
    weight_decay,
    max_grad_norm,
  } = config
  let n = params.length()
  let ptrs_bytes = Int64::from_int(n * 8)
  let sizes_bytes = Int64::from_int(n * 4)

  let g_ptrs = FixedArray::makei(n, fn(i) { grads[i].data })

  if n != opt_adam_cache.cached_n {
    let p_ptrs = FixedArray::makei(n, fn(i) { params[i].data })
    let m_ptrs = FixedArray::makei(n, fn(i) { m[i].data })
    let v_ptrs = FixedArray::makei(n, fn(i) { v[i].data })
    let sizes = FixedArray::makei(n, fn(i) { params[i].numel() })
    let block_offsets = compute_block_offsets(sizes, n)
    let mut tb = 0
    for i in 0..<n {
      tb = tb + (sizes[i] + 255) / 256
    }

    opt_adam_cache.d_p = @cuda.alloc(ptrs_bytes)
    opt_adam_cache.d_g = @cuda.alloc(ptrs_bytes)
    opt_adam_cache.d_m = @cuda.alloc(ptrs_bytes)
    opt_adam_cache.d_v = @cuda.alloc(ptrs_bytes)
    opt_adam_cache.d_sizes = @cuda.alloc(sizes_bytes)
    opt_adam_cache.d_offsets = @cuda.alloc(sizes_bytes)
    opt_adam_cache.norm_buf = @cuda.alloc(4L)

    check_cuda(@cuda.copy_ptrs_h2d(opt_adam_cache.d_p, p_ptrs), "copy_ptrs_h2d")
    check_cuda(@cuda.copy_ptrs_h2d(opt_adam_cache.d_m, m_ptrs), "copy_ptrs_h2d")
    check_cuda(@cuda.copy_ptrs_h2d(opt_adam_cache.d_v, v_ptrs), "copy_ptrs_h2d")
    check_cuda(@cuda.raw_h2d(opt_adam_cache.d_sizes, sizes), "raw_h2d")
    check_cuda(
      @cuda.raw_h2d(opt_adam_cache.d_offsets, block_offsets),
      "raw_h2d",
    )

    opt_adam_cache.cached_n = n
    opt_adam_cache.total_blocks = tb
  }

  check_cuda(@cuda.copy_ptrs_h2d(opt_adam_cache.d_g, g_ptrs), "copy_ptrs_h2d")
  check_cuda(@cuda.memset_zero(opt_adam_cache.norm_buf, 1L), "memset_zero")
  check_cuda(
    @cuda.mt_norm_sq_gpu_only(
      opt_adam_cache.d_g,
      opt_adam_cache.d_sizes,
      opt_adam_cache.d_offsets,
      n,
      opt_adam_cache.total_blocks,
      opt_adam_cache.norm_buf,
    ),
    "mt_norm_sq_gpu_only",
  )
  check_cuda(
    @cuda.mt_adam_autoclip(
      opt_adam_cache.d_p,
      opt_adam_cache.d_g,
      opt_adam_cache.d_m,
      opt_adam_cache.d_v,
      opt_adam_cache.d_sizes,
      opt_adam_cache.d_offsets,
      n,
      opt_adam_cache.total_blocks,
      lr,
      beta1,
      beta2,
      one_minus_beta1,
      one_minus_beta2,
      bias_correction1,
      bias_correction2,
      eps,
      weight_decay,
      opt_adam_cache.norm_buf,
      max_grad_norm,
    ),
    "mt_adam_autoclip",
  )
}

///|
pub impl @optimizer.BatchRmspropAutoclip for GpuTensor with batch_rmsprop_autoclip(
  params : Array[GpuTensor],
  grads : Array[GpuTensor],
  v : Array[GpuTensor],
  buf : Array[GpuTensor],
  config : @optimizer.RmspropConfig,
) -> Unit {
  let {
    lr,
    alpha,
    one_minus_alpha,
    eps,
    weight_decay,
    momentum,
    has_momentum,
    max_grad_norm,
  } = config
  let n = params.length()
  let ptrs_bytes = Int64::from_int(n * 8)
  let sizes_bytes = Int64::from_int(n * 4)

  let g_ptrs = FixedArray::makei(n, fn(i) { grads[i].data })
  let has_momentum_i = if has_momentum { 1 } else { 0 }

  if n != opt_rmsprop_cache.cached_n {
    let p_ptrs = FixedArray::makei(n, fn(i) { params[i].data })
    let v_ptrs = FixedArray::makei(n, fn(i) { v[i].data })
    let buf_ptrs = FixedArray::makei(n, fn(i) { buf[i].data })
    let sizes = FixedArray::makei(n, fn(i) { params[i].numel() })
    let block_offsets = compute_block_offsets(sizes, n)
    let mut tb = 0
    for i in 0..<n {
      tb = tb + (sizes[i] + 255) / 256
    }

    opt_rmsprop_cache.d_p = @cuda.alloc(ptrs_bytes)
    opt_rmsprop_cache.d_g = @cuda.alloc(ptrs_bytes)
    opt_rmsprop_cache.d_v = @cuda.alloc(ptrs_bytes)
    opt_rmsprop_cache.d_buf = @cuda.alloc(ptrs_bytes)
    opt_rmsprop_cache.d_sizes = @cuda.alloc(sizes_bytes)
    opt_rmsprop_cache.d_offsets = @cuda.alloc(sizes_bytes)
    opt_rmsprop_cache.norm_buf = @cuda.alloc(4L)

    check_cuda(
      @cuda.copy_ptrs_h2d(opt_rmsprop_cache.d_p, p_ptrs),
      "copy_ptrs_h2d",
    )
    check_cuda(
      @cuda.copy_ptrs_h2d(opt_rmsprop_cache.d_v, v_ptrs),
      "copy_ptrs_h2d",
    )
    check_cuda(
      @cuda.copy_ptrs_h2d(opt_rmsprop_cache.d_buf, buf_ptrs),
      "copy_ptrs_h2d",
    )
    check_cuda(@cuda.raw_h2d(opt_rmsprop_cache.d_sizes, sizes), "raw_h2d")
    check_cuda(
      @cuda.raw_h2d(opt_rmsprop_cache.d_offsets, block_offsets),
      "raw_h2d",
    )

    opt_rmsprop_cache.cached_n = n
    opt_rmsprop_cache.total_blocks = tb
  }

  check_cuda(
    @cuda.copy_ptrs_h2d(opt_rmsprop_cache.d_g, g_ptrs),
    "copy_ptrs_h2d",
  )
  check_cuda(@cuda.memset_zero(opt_rmsprop_cache.norm_buf, 1L), "memset_zero")
  check_cuda(
    @cuda.mt_norm_sq_gpu_only(
      opt_rmsprop_cache.d_g,
      opt_rmsprop_cache.d_sizes,
      opt_rmsprop_cache.d_offsets,
      n,
      opt_rmsprop_cache.total_blocks,
      opt_rmsprop_cache.norm_buf,
    ),
    "mt_norm_sq_gpu_only",
  )
  check_cuda(
    @cuda.mt_rmsprop_autoclip(
      opt_rmsprop_cache.d_p,
      opt_rmsprop_cache.d_g,
      opt_rmsprop_cache.d_v,
      opt_rmsprop_cache.d_buf,
      opt_rmsprop_cache.d_sizes,
      opt_rmsprop_cache.d_offsets,
      n,
      opt_rmsprop_cache.total_blocks,
      lr,
      alpha,
      one_minus_alpha,
      eps,
      weight_decay,
      momentum,
      has_momentum_i,
      opt_rmsprop_cache.norm_buf,
      max_grad_norm,
    ),
    "mt_rmsprop_autoclip",
  )
}
