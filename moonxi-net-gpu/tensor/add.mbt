// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub fn gpu_add(a : GpuTensor, b : GpuTensor) -> GpuTensor {
  let output = GpuTensor::alloc_st(a.shape)
  check_cuda(
    @cuda.elementwise_add(a.data, b.data, output.data, a.numel()),
    "elementwise_add",
  )
  output
}

///|
pub fn GpuTensor::add_into(self : GpuTensor, other : GpuTensor) -> Unit {
  check_cuda(
    @cuda.elementwise_add_into(self.data, other.data, self.numel()),
    "elementwise_add_into",
  )
}
