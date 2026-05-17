// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn gpu_avg_pool2d(
  input : GpuTensor,
  kernel_size : Int,
  stride : Int,
  padding : Int,
  cudnn : @cuda.CudnnHandle,
) -> GpuTensor {
  let output = GpuTensor::alloc_st(
    @ts.ShapeTensor::pool2d(input.shape, kernel_size, stride, padding),
  )
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let ctx = @cuda.pool2d_create(
    n, c, h, w, kernel_size, kernel_size, padding, padding, stride, stride, 1,
  )
  check_cuda(
    @cuda.pool2d_forward(ctx, cudnn, input.data, output.data),
    "pool2d_forward (avg)",
  )
  output
}

///|
pub fn gpu_maxpool2d(
  input : GpuTensor,
  kernel_size : Int,
  stride : Int,
  padding : Int,
  cudnn : @cuda.CudnnHandle,
) -> GpuTensor {
  let output = GpuTensor::alloc_st(
    @ts.ShapeTensor::pool2d(input.shape, kernel_size, stride, padding),
  )
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let ctx = @cuda.pool2d_create(
    n, c, h, w, kernel_size, kernel_size, padding, padding, stride, stride, 0,
  )
  check_cuda(
    @cuda.pool2d_forward(ctx, cudnn, input.data, output.data),
    "pool2d_forward (max)",
  )
  output
}

///|
pub fn gpu_adaptive_avg_pool2d(input : GpuTensor) -> GpuTensor {
  let output = GpuTensor::alloc_st(
    @ts.ShapeTensor::adaptive_avg_pool2d(input.shape, 1),
  )
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  check_cuda(
    @cuda.adaptive_avg_pool2d_forward(input.data, output.data, n, c, h, w),
    "adaptive_avg_pool2d_forward",
  )
  output
}
