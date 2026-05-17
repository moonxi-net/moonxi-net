// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Opaque handle to GPU device memory. Maps to `void*` in C.
/// Returned by `alloc`, consumed by all GPU kernels.
/// MoonBit does NOT manage its lifecycle — caller must free via `cuda_free_device`.
#external
pub type GpuBuffer

///|
/// Opaque handle to cuBLAS context. Maps to `void*` in C.
#external
pub type CublasHandle

///|
/// Opaque handle to cuDNN context. Maps to `void*` in C.
#external
pub type CudnnHandle

///|
/// Opaque handle to a CUDA memory pool. Maps to `void*` in C.
#external
pub type GpuPool

///|
/// Opaque handle to cuDNN Conv2D descriptors + workspace. Maps to `void*` in C.
#external
pub type Conv2dContext

///|
/// Opaque handle to cuDNN BatchNorm descriptors. Maps to `void*` in C.
#external
pub type BatchNormContext

///|
/// Opaque handle to cuDNN Pooling descriptors. Maps to `void*` in C.
#external
pub type Pool2dContext

///|
/// Opaque handle to a CUDA event. Maps to `cudaEvent_t` in C.
#external
pub type CudaEvent
