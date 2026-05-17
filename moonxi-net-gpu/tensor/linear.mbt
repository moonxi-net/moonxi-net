// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn gpu_linear(
  input : GpuTensor,
  weight : GpuTensor,
  bias : GpuTensor,
  cublas : @cuda.CublasHandle,
) -> GpuTensor {
  let output = GpuTensor::alloc_st(
    @ts.ShapeTensor::linear(input.shape, weight.shape),
  )
  let batch = input.dim(0)
  let in_f = input.dim(1)
  let out_f = weight.dim(0)
  check_cuda(
    @cuda.matmul(
      cublas,
      input.data,
      weight.data,
      output.data,
      batch,
      in_f,
      out_f,
      false,
      true,
      1.0,
      0.0,
    ),
    "matmul in gpu_linear",
  )
  check_cuda(
    @cuda.bias_add_nchw(output.data, bias.data, batch, out_f, 1),
    "bias_add_nchw in gpu_linear",
  )
  output
}
