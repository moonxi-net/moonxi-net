// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn gpu_batchnorm_training(
  input : GpuTensor,
  gamma : GpuTensor,
  beta : GpuTensor,
  running_mean : GpuTensor,
  running_var : GpuTensor,
  momentum : Float,
  epsilon : Float,
  cudnn : @cuda.CudnnHandle,
) -> (GpuTensor, GpuTensor, GpuTensor) {
  let (out_shape, sm_shape, siv_shape) = @ts.ShapeTensor::batchnorm_training(
    input.shape,
    gamma.shape,
    beta.shape,
    running_mean.shape,
    running_var.shape,
    momentum,
    epsilon,
  )
  let output = GpuTensor::alloc_st(out_shape)
  let save_mean = GpuTensor::zeros_st(sm_shape)
  let save_inv_var = GpuTensor::zeros_st(siv_shape)
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  let ctx = @cuda.batchnorm_create(n, c, h, w)
  check_cuda(
    @cuda.batchnorm_forward(
      ctx,
      cudnn,
      input.data,
      output.data,
      gamma.data,
      beta.data,
      running_mean.data,
      running_var.data,
      save_mean.data,
      save_inv_var.data,
      momentum,
      epsilon,
      true,
    ),
    "batchnorm_forward (training)",
  )
  (output, save_mean, save_inv_var)
}

///|
pub fn gpu_batchnorm_inference(
  input : GpuTensor,
  gamma : GpuTensor,
  beta : GpuTensor,
  running_mean : GpuTensor,
  running_var : GpuTensor,
  epsilon : Float,
  _cudnn : @cuda.CudnnHandle,
) -> GpuTensor {
  let output = GpuTensor::alloc_st(
    @ts.ShapeTensor::batchnorm_inference(
      input.shape,
      gamma.shape,
      beta.shape,
      running_mean.shape,
      running_var.shape,
      epsilon,
    ),
  )
  let n = input.dim(0)
  let c = input.dim(1)
  let h = input.dim(2)
  let w = input.dim(3)
  check_cuda(
    @cuda.cuda_batchnorm_inference(
      input.data,
      output.data,
      gamma.data,
      beta.data,
      running_mean.data,
      running_var.data,
      epsilon,
      n,
      c,
      h * w,
    ),
    "cuda_batchnorm_inference",
  )
  output
}
