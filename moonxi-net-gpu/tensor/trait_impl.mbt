// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Tensor, BlasTensor, ImageTensor implementations for GpuTensor.
/// Wraps gpu_ops functions as trait methods on GpuTensor.
/// Note: Add and Sub impls are in gpu/tensor/gpu_tensor.mbt (to avoid circular deps).
pub impl @ts.Tensor for GpuTensor with dims(gt : GpuTensor) -> FixedArray[Int] {
  gt.shape.shape
}

///|
pub impl @ts.Tensor for GpuTensor with zeros(dims : FixedArray[Int]) -> GpuTensor {
  let shape = dims.iter().collect()
  GpuTensor::zeros(shape)
}

///|
pub impl @ts.Tensor for GpuTensor with zeros_like(gt : GpuTensor) -> GpuTensor {
  GpuTensor::zeros_st(gt.shape)
}

///|
pub impl @ts.Tensor for GpuTensor with from_host(
  data : FixedArray[Float],
  shape : Array[Int],
) -> GpuTensor {
  GpuTensor::from_host(data, shape)
}

///|
pub impl @ts.Tensor for GpuTensor with square(gt : GpuTensor) -> GpuTensor {
  let out = GpuTensor::alloc_st(gt.shape)
  check_cuda(@cuda.square(gt.data, out.data, gt.numel()), "square")
  out
}

///|
pub impl @ts.Tensor for GpuTensor with sqrt(gt : GpuTensor) -> GpuTensor {
  let out = GpuTensor::alloc_st(gt.shape)
  check_cuda(@cuda.sqrt(gt.data, out.data, gt.numel()), "sqrt")
  out
}

///|
pub impl @ts.Tensor for GpuTensor with mean(gt : GpuTensor) -> GpuTensor {
  let host = gt.to_host()
  let sum = host.fold(init=(0.0 : Float), fn(acc, v) { acc + v })
  let n = Float::from_int(gt.numel())
  GpuTensor::_from_host_unchecked(FixedArray::make(1, sum / n), [1])
}

///|
pub impl @ts.Tensor for GpuTensor with scale(gt : GpuTensor, s : Float) -> GpuTensor {
  let out = GpuTensor::alloc_st(gt.shape)
  check_cuda(@cuda.scale_out(gt.data, out.data, gt.numel(), s), "scale_out")
  out
}

///|
pub impl @ts.Tensor for GpuTensor with mul_elem(a : GpuTensor, b : GpuTensor) -> GpuTensor {
  let out = GpuTensor::alloc_st(a.shape)
  check_cuda(@cuda.mul_elem(a.data, b.data, out.data, a.numel()), "mul_elem")
  out
}

///|
pub impl @ts.Tensor for GpuTensor with div_elem(a : GpuTensor, b : GpuTensor) -> GpuTensor {
  let out = GpuTensor::alloc_st(a.shape)
  check_cuda(@cuda.div_elem(a.data, b.data, out.data, a.numel()), "div_elem")
  out
}

///|
pub impl @ts.Tensor for GpuTensor with scalar(v : Float) -> GpuTensor {
  GpuTensor::_from_host_unchecked(FixedArray::make(1, v), [1])
}

///|
pub impl @ts.Tensor for GpuTensor with size(gt : GpuTensor) -> Int {
  gt.numel()
}

///|
pub impl @ts.Tensor for GpuTensor with reduce_sum_to(
  gt : GpuTensor,
  target_dims : FixedArray[Int],
) -> GpuTensor {
  let gd = @ts.ShapeTensor::to_array(gt.shape)
  let xd = gd.length()
  let td = target_dims.length()
  let target_shape = target_dims.iter().collect()
  // Fast path: shapes match exactly — return a view (no reduction needed)
  if xd == td {
    let mut all_match = true
    for i in 0..<xd {
      if gd[i] != target_dims[i] {
        all_match = false
      }
    }
    if all_match {
      return gt._view_unchecked(target_shape)
    }
  }
  let offset = xd - td
  // Step 1: Reduce leading dimensions using GPU kernel
  let mut current_data = gt.data
  let mut current_shape = gd
  if offset > 0 {
    // Compute lead_stride = product of leading dims to reduce
    let mut lead_stride = 1
    for i in 0..<offset {
      lead_stride = lead_stride * gd[i]
    }
    // Output shape is [gd[offset], gd[offset+1], ..., gd[xd-1]]
    let output_shape_arr = FixedArray::makei(xd - offset, fn(i) {
      gd[offset + i]
    })
    let output_shape = output_shape_arr.iter().collect()
    let output_numel = GpuTensor::shape_numel(output_shape)
    let output = GpuTensor::alloc_shape(output_shape)
    check_cuda(
      @cuda.reduce_sum_leading(
        current_data,
        output.data,
        output_numel,
        lead_stride,
      ),
      "reduce_sum_leading",
    )
    current_data = output.data
    current_shape = output_shape
  }
  // Step 2: Reduce remaining dimensions where target has 1
  for dim_idx in 0..<td {
    if target_dims[dim_idx] == 1 && current_shape[dim_idx] != 1 {
      let dim_size = current_shape[dim_idx]
      // Compute strides for current_shape
      let mut stride_after = 1
      for j in (dim_idx + 1)..<td {
        stride_after = stride_after * current_shape[j]
      }
      let stride_before = GpuTensor::shape_numel(current_shape) /
        (dim_size * stride_after)
      // Output shape: same as current_shape but with this dim set to 1
      let output_shape_arr = FixedArray::makei(current_shape.length(), fn(i) {
        if i == dim_idx {
          1
        } else {
          current_shape[i]
        }
      })
      let output_shape = output_shape_arr.iter().collect()
      let output_numel = GpuTensor::shape_numel(output_shape)
      let output = GpuTensor::alloc_shape(output_shape)
      check_cuda(
        @cuda.reduce_sum_dim(
          current_data,
          output.data,
          output_numel,
          dim_size,
          stride_before,
          stride_after,
        ),
        "reduce_sum_dim",
      )
      current_data = output.data
      current_shape = output_shape
    }
  }
  // Create result tensor with target_shape (may be same as current_shape)
  // If shapes match, just return a view
  let mut shapes_match = true
  if current_shape.length() != target_shape.length() {
    shapes_match = false
  } else {
    for i in 0..<current_shape.length() {
      if current_shape[i] != target_shape[i] {
        shapes_match = false
      }
    }
  }
  if shapes_match {
    { data: current_data, shape: @ts.ShapeTensor::from_array(target_shape) }
  } else {
    // This shouldn't happen if the reduction logic is correct, but handle it
    let n = GpuTensor::shape_numel(current_shape)
    let h = FixedArray::make(n, Float::from_double(0.0))
    check_cuda(@cuda.copy_d2h(h, current_data), "copy_d2h in reduce_sum_to")
    GpuTensor::_from_host_unchecked(h, target_shape)
  }
}

///|
pub impl @ts.Tensor for GpuTensor with value(gt : GpuTensor) -> @ts.TensorData {
  let s = gt.shape.shape
  { dims: FixedArray::makei(s.length(), fn(i) { s[i] }), data: gt.to_host() }
}

///|
pub impl @ts.Tensor for GpuTensor with view(
  gt : GpuTensor,
  new_shape : FixedArray[Int],
) -> GpuTensor {
  gt.view(new_shape.iter().collect())
}

///|
pub impl @ts.Tensor for GpuTensor with add(a : GpuTensor, b : GpuTensor) -> GpuTensor {
  gpu_add(a, b)
}

///|
pub impl @ts.Tensor for GpuTensor with add_into(a : GpuTensor, b : GpuTensor) -> Unit {
  a.add_into(b)
}

///|
pub impl @ts.Tensor for GpuTensor with sub(a : GpuTensor, b : GpuTensor) -> GpuTensor {
  let out = GpuTensor::alloc_st(a.shape)
  check_cuda(@cuda.sub(a.data, b.data, out.data, a.numel()), "sub")
  out
}

///|
pub impl @ts.BlasTensor for GpuTensor with matmul(a : GpuTensor, b : GpuTensor) -> GpuTensor {
  let out_shape = @ts.ShapeTensor::matmul(a.shape, b.shape)
  let out = GpuTensor::alloc_st(out_shape)
  let n = a.dim(0)
  let m = a.dim(1)
  let p = b.dim(1)
  let ctx = @cuda.get_context(0)
  check_cuda(
    @cuda.matmul(
      ctx.cublas,
      a.data,
      b.data,
      out.data,
      n,
      m,
      p,
      false,
      false,
      1.0,
      0.0,
    ),
    "matmul",
  )
  out
}

///|
pub impl @ts.BlasTensor for GpuTensor with transpose(gt : GpuTensor) -> GpuTensor {
  let out_shape = @ts.ShapeTensor::transpose(gt.shape)
  let out = GpuTensor::alloc_st(out_shape)
  let n = gt.dim(0)
  let m = gt.dim(1)
  check_cuda(@cuda.transpose_2d(gt.data, out.data, n, m), "transpose_2d")
  out
}

///|
/// ImageTensor impl for GpuTensor
pub impl @ts.ImageTensor for GpuTensor with relu(gt : GpuTensor) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  gpu_relu(gt, ctx.cudnn)
}

///|
pub impl @ts.ImageTensor for GpuTensor with conv2d(
  input : GpuTensor,
  weight : GpuTensor,
  bias : GpuTensor,
  stride : Int,
  padding : Int,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  gpu_conv2d(input, weight, bias, ctx.cudnn, stride, padding)
}

///|
pub impl @ts.ImageTensor for GpuTensor with maxpool2d(
  gt : GpuTensor,
  kernel_size : Int,
  stride : Int,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  gpu_maxpool2d(gt, kernel_size, stride, 1, ctx.cudnn)
}

///|
pub impl @ts.ImageTensor for GpuTensor with adaptive_avg_pool2d(
  gt : GpuTensor,
  output_size : Int,
) -> GpuTensor {
  // Note: gpu_adaptive_avg_pool2d doesn't take ctx, output_size is ignored (always 1x1)
  ignore(output_size)
  gpu_adaptive_avg_pool2d(gt)
}

///|
pub impl @ts.ImageTensor for GpuTensor with batchnorm_training(
  input : GpuTensor,
  gamma : GpuTensor,
  beta : GpuTensor,
  running_mean : GpuTensor,
  running_var : GpuTensor,
  momentum : Float,
  eps : Float,
) -> (GpuTensor, GpuTensor, GpuTensor) {
  let ctx = @cuda.get_context(0)
  gpu_batchnorm_training(
    input,
    gamma,
    beta,
    running_mean,
    running_var,
    momentum,
    eps,
    ctx.cudnn,
  )
}

///|
pub impl @ts.ImageTensor for GpuTensor with batchnorm_inference(
  input : GpuTensor,
  gamma : GpuTensor,
  beta : GpuTensor,
  running_mean : GpuTensor,
  running_var : GpuTensor,
  eps : Float,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  gpu_batchnorm_inference(
    input,
    gamma,
    beta,
    running_mean,
    running_var,
    eps,
    ctx.cudnn,
  )
}

///|
pub impl @ts.ImageTensor for GpuTensor with softmax_cross_entropy(
  logits : GpuTensor,
  targets : GpuTensor,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  gpu_softmax_cross_entropy(logits, targets, ctx.cublas)
}

///|
pub impl @ts.ImageTensor for GpuTensor with cross_entropy_with_labels(
  logits : GpuTensor,
  labels : GpuTensor,
  num_classes : Int,
) -> GpuTensor {
  gpu_cross_entropy_with_labels(logits, labels, num_classes)
}

// =============================================================================
// ImageBackwardOps implementations for GpuTensor
// =============================================================================

///|
pub impl @ts.ImageBackwardOps for GpuTensor with conv2d_backward_data(
  grad_output : GpuTensor,
  weight : GpuTensor,
  input : GpuTensor,
  stride : Int,
  padding : Int,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  let n = input.dim(0)
  let c_in = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let c_out = weight.dim(0)
  let kh = weight.dim(2)
  let kw = weight.dim(3)
  let cudnn_ctx = @cuda.conv2d_create(
    n, c_in, h, w, c_out, kh, kw, padding, padding, stride, stride, 1, 1,
  )
  let grad_input = GpuTensor::alloc_st(input.shape)
  check_cuda(
    @cuda.conv2d_backward_data(
      cudnn_ctx,
      ctx.cudnn,
      grad_output.data,
      weight.data,
      grad_input.data,
    ),
    "conv2d_backward_data",
  )
  grad_input
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with conv2d_backward_weight(
  grad_output : GpuTensor,
  input : GpuTensor,
  weight : GpuTensor,
  stride : Int,
  padding : Int,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  let n = input.dim(0)
  let c_in = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let c_out = weight.dim(0)
  let kh = weight.dim(2)
  let kw = weight.dim(3)
  let cudnn_ctx = @cuda.conv2d_create(
    n, c_in, h, w, c_out, kh, kw, padding, padding, stride, stride, 1, 1,
  )
  let grad_weight = GpuTensor::alloc_st(weight.shape)
  check_cuda(
    @cuda.conv2d_backward_filter(
      cudnn_ctx,
      ctx.cudnn,
      input.data,
      grad_output.data,
      grad_weight.data,
    ),
    "conv2d_backward_filter",
  )
  grad_weight
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with conv2d_backward_bias(
  grad_output : GpuTensor,
) -> GpuTensor {
  let bias_shape = @ts.ImageBackwardOps::conv2d_backward_bias(grad_output.shape)
  let grad_bias = GpuTensor::zeros_st(bias_shape)
  let ctx = @cuda.get_context(0)
  let batch = grad_output.dim(0)
  let c_out = grad_output.dim(1)
  let oh = grad_output.dim(2)
  let ow = grad_output.dim(3)
  check_cuda(
    @cuda.conv2d_backward_bias(
      ctx.cudnn,
      grad_output.data,
      grad_bias.data,
      batch,
      c_out,
      oh,
      ow,
    ),
    "conv2d_backward_bias",
  )
  grad_bias
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with relu_backward(
  grad_output : GpuTensor,
  input : GpuTensor,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  let numel = Int64::from_int(input.numel())
  let grad_input = GpuTensor::alloc_st(input.shape)
  check_cuda(
    @cuda.cudnn_relu_backward(
      ctx.cudnn,
      numel,
      input.data,
      grad_output.data,
      grad_input.data,
    ),
    "cudnn_relu_backward",
  )
  grad_input
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with batchnorm_backward(
  grad_output : GpuTensor,
  input : GpuTensor,
  gamma : GpuTensor,
  save_mean : GpuTensor,
  save_inv_var : GpuTensor,
  eps : Float,
) -> (GpuTensor, GpuTensor, GpuTensor) {
  let (gi_shape, gg_shape, gb_shape) = @ts.ImageBackwardOps::batchnorm_backward(
    grad_output.shape,
    input.shape,
    gamma.shape,
    save_mean.shape,
    save_inv_var.shape,
    eps,
  )
  let ctx = @cuda.get_context(0)
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let bn_ctx = @cuda.batchnorm_create(n, c, h, w)
  let grad_input = GpuTensor::alloc_st(gi_shape)
  let grad_gamma = GpuTensor::zeros_st(gg_shape)
  let grad_beta = GpuTensor::zeros_st(gb_shape)
  check_cuda(
    @cuda.batchnorm_backward(
      bn_ctx,
      ctx.cudnn,
      input.data,
      grad_output.data,
      grad_input.data,
      gamma.data,
      grad_gamma.data,
      grad_beta.data,
      save_mean.data,
      save_inv_var.data,
      eps,
    ),
    "batchnorm_backward",
  )
  (grad_input, grad_gamma, grad_beta)
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with maxpool2d_backward(
  grad_output : GpuTensor,
  input : GpuTensor,
  kernel_size : Int,
  stride : Int,
) -> GpuTensor {
  let ctx = @cuda.get_context(0)
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let pool_ctx = @cuda.pool2d_create(
    n, c, h, w, kernel_size, kernel_size, 1, 1, stride, stride, 0,
  )
  let grad_input = GpuTensor::alloc_st(input.shape)
  check_cuda(
    @cuda.pool2d_backward(
      pool_ctx,
      ctx.cudnn,
      input.data,
      grad_output.data,
      grad_input.data,
    ),
    "pool2d_backward",
  )
  grad_input
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with adaptive_avg_pool2d_backward(
  grad_output : GpuTensor,
  input : GpuTensor,
) -> GpuTensor {
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let grad_input = GpuTensor::alloc_st(input.shape)
  check_cuda(
    @cuda.adaptive_avg_pool2d_backward(
      grad_output.data,
      grad_input.data,
      n,
      c,
      h,
      w,
    ),
    "adaptive_avg_pool2d_backward",
  )
  grad_input
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with softmax_ce_backward(
  logits : GpuTensor,
  targets : GpuTensor,
) -> GpuTensor {
  let batch = logits.dim(0)
  let num_classes = logits.dim(1)
  let grad_logits = GpuTensor::alloc_st(logits.shape)
  check_cuda(
    @cuda.softmax_ce_backward(
      logits.data,
      targets.data,
      grad_logits.data,
      batch,
      num_classes,
    ),
    "softmax_ce_backward",
  )
  grad_logits
}

///|
pub impl @ts.ImageBackwardOps for GpuTensor with softmax_ce_backward_labels(
  logits : GpuTensor,
  labels : GpuTensor,
  num_classes : Int,
) -> GpuTensor {
  gpu_softmax_ce_backward_labels(logits, labels, num_classes)
}

///|
pub impl @ts.Tensor for GpuTensor with broadcast_to(
  gt : GpuTensor,
  target_shape : FixedArray[Int],
) -> GpuTensor {
  let out_shape = @ts.ShapeTensor::broadcast_to(gt.shape, target_shape)
  let n = @ts.ShapeTensor::size(out_shape)
  let out = GpuTensor::alloc_st(out_shape)
  let scalar_val = @cuda.read_scalar(gt.data)
  check_cuda(@cuda.fill(out.data, n, scalar_val), "fill in broadcast_to")
  out
}
