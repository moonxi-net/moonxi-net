// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn gpu_relu(input : GpuTensor, cudnn : @cuda.CudnnHandle) -> GpuTensor {
  let output = GpuTensor::alloc_st(input.shape)
  check_cuda(
    @cuda.cudnn_relu_forward(
      cudnn,
      Int64::from_int(input.numel()),
      input.data,
      output.data,
    ),
    "cudnn_relu_forward",
  )
  output
}
