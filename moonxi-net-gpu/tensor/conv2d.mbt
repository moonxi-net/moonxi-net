// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn gpu_conv2d(
  input : GpuTensor,
  weight : GpuTensor,
  bias : GpuTensor,
  cudnn : @cuda.CudnnHandle,
  stride : Int,
  padding : Int,
) -> GpuTensor {
  let output = GpuTensor::alloc_st(
    @ts.ShapeTensor::conv2d(
      input.shape,
      weight.shape,
      bias.shape,
      stride,
      padding,
    ),
  )
  let n = input.dim(0)
  let c_in = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let c_out = weight.dim(0)
  let kh = weight.dim(2)
  let kw = weight.dim(3)
  let ctx = @cuda.conv2d_create(
    n, c_in, h, w, c_out, kh, kw, padding, padding, stride, stride, 1, 1,
  )
  check_cuda(
    @cuda.conv2d_forward(
      ctx,
      cudnn,
      input.data,
      weight.data,
      bias.data,
      output.data,
    ),
    "conv2d_forward",
  )
  output
}
