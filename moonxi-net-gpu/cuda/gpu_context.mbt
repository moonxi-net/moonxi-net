// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Device-indexed GpuContext registry.
/// Allows GPU operations to retrieve the appropriate cuBLAS/cuDNN handles by device ID.
pub(all) struct GpuContext {
  cublas : CublasHandle
  cudnn : CudnnHandle
}

///|
let gpu_contexts : Ref[FixedArray[GpuContext?]] = Ref(FixedArray::make(8, None))

///|
/// Register a GpuContext for a device ID.
pub fn register_context(device_id : Int, ctx : GpuContext) -> Unit {
  gpu_contexts.val[device_id] = Some(ctx)
}

///|
/// Initialize CUDA: create cuBLAS and cuDNN handles, register context for device.
pub fn gpu_init(device_id : Int) -> Unit {
  let cublas = cublas_create()
  let cudnn = cudnn_create()
  register_context(device_id, { cublas, cudnn })
}

///|
/// Retrieve the GpuContext for a device ID.
/// Aborts if no context is registered for that device.
/// Note: uses abort() instead of raise because this package cannot import
/// the root error types (would create a circular dependency with gpu/tensor).
pub fn get_context(device_id : Int) -> GpuContext {
  match gpu_contexts.val[device_id] {
    Some(ctx) => ctx
    None => abort("GpuContext not registered for device \{device_id}")
  }
}
