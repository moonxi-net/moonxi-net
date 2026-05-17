// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub(all) struct GpuTensor {
  data : @cuda.GpuBuffer
  shape : @ts.ShapeTensor
}

///|
/// Check a CUDA return code and abort if non-zero.
fn check_cuda(rc : Int, msg : String) -> Unit {
  if rc != 0 {
    abort("\{msg} (rc=\{rc})")
  }
}

///|
/// Computed property: total number of elements (product of shape dimensions).
pub fn GpuTensor::numel(self : GpuTensor) -> Int {
  @ts.ShapeTensor::size(self.shape)
}

///|
/// Computed property: row-major strides derived from shape.
pub fn GpuTensor::stride(self : GpuTensor) -> Array[Int] {
  GpuTensor::compute_stride(@ts.ShapeTensor::to_array(self.shape))
}

///|
pub fn GpuTensor::dim(self : GpuTensor, i : Int) -> Int {
  @ts.ShapeTensor::dim(self.shape, i)
}

///|
pub fn GpuTensor::ndim(self : GpuTensor) -> Int {
  @ts.ShapeTensor::ndim(self.shape)
}

///|
priv struct GpuPoolState {
  mut pool : @cuda.GpuPool?
  mut use_pool : Bool
}

///|
let gpu_pool_state : GpuPoolState = GpuPoolState::{
  pool: None,
  use_pool: false,
}

///|
let gpu_inference_mode : Ref[Bool] = Ref(false)

///|
pub fn gpu_set_inference(mode : Bool) -> Unit {
  gpu_inference_mode.val = mode
}

///|
pub fn gpu_is_inference() -> Bool {
  gpu_inference_mode.val
}

///|
/// Execute a callback in inference mode, guaranteeing restoration of the previous mode.
/// This prevents inference mode from leaking if the callback throws or returns early.
pub fn[T] gpu_with_inference(inference : Bool, cb : () -> T) -> T {
  let prev = gpu_inference_mode.val
  gpu_inference_mode.val = inference
  let result = cb()
  gpu_inference_mode.val = prev
  result
}

///|
pub fn gpu_enable_pool(capacity : Int64) -> Unit {
  gpu_pool_state.pool = Some(@cuda.pool_create(capacity))
  gpu_pool_state.use_pool = true
}

///|
pub fn gpu_workspace_reset() -> Unit {
  if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(p) => @cuda.pool_reset(p)
      None => ()
    }
  }
}

///|
pub fn gpu_pool_peak() -> Int64 {
  if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(p) => @cuda.pool_peak(p)
      None => 0L
    }
  } else {
    0L
  }
}

///|
pub fn gpu_pool_used() -> Int64 {
  if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(p) => @cuda.pool_used(p)
      None => 0L
    }
  } else {
    0L
  }
}

///|
pub fn GpuTensor::compute_stride(shape : Array[Int]) -> Array[Int] {
  let n = shape.length()
  if n == 0 {
    return []
  }
  let stride = Array::make(n, 0)
  stride[n - 1] = 1
  if n > 1 {
    let mut i = n - 2
    while i >= 0 {
      stride[i] = stride[i + 1] * shape[i + 1]
      i = i - 1
    }
  }
  stride
}

///|
/// Compute total number of elements for a given shape.
pub fn GpuTensor::shape_numel(shape : Array[Int]) -> Int {
  let mut r = 1
  for i in 0..<shape.length() {
    r = r * shape[i]
  }
  r
}

///|
pub fn GpuTensor::zeros(shape : Array[Int]) -> GpuTensor {
  GpuTensor::zeros_st(@ts.ShapeTensor::from_array(shape))
}

///|
pub fn GpuTensor::zeros_st(shape : @ts.ShapeTensor) -> GpuTensor {
  let n = @ts.ShapeTensor::size(shape)
  let bytes = Int64::from_int(n * 4) // 4 bytes per float
  let ptr = if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(p) => @cuda.pool_alloc(p, bytes)
      None => @cuda.alloc(bytes)
    }
  } else {
    @cuda.alloc(bytes)
  }
  check_cuda(
    @cuda.memset_zero(ptr, Int64::from_int(n)),
    "memset_zero in alloc_shape",
  )
  { data: ptr, shape }
}

///|

///|
/// Internal helper: copy host data to GPU without shape validation.
/// Used by non-raise methods that know the shape is correct (e.g. mean).
fn GpuTensor::_from_host_unchecked(
  data : FixedArray[Float],
  shape : Array[Int],
) -> GpuTensor {
  let n = data.length()
  let bytes = Int64::from_int(n * 4)
  let ptr = if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(p) => @cuda.pool_alloc(p, bytes)
      None => @cuda.alloc(bytes)
    }
  } else {
    @cuda.alloc(bytes)
  }
  check_cuda(@cuda.copy_h2d(ptr, data), "copy_h2d in _from_host_unchecked")
  { data: ptr, shape: @ts.ShapeTensor::from_array(shape) }
}

///|
pub fn GpuTensor::from_host(
  data : FixedArray[Float],
  shape : Array[Int],
) -> GpuTensor {
  let n = GpuTensor::shape_numel(shape)
  if data.length() != n {
    abort("from_host: data length \{data.length()} != expected \{n}")
  }
  let bytes = Int64::from_int(n * 4)
  let ptr = if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(p) => @cuda.pool_alloc(p, bytes)
      None => @cuda.alloc(bytes)
    }
  } else {
    @cuda.alloc(bytes)
  }
  check_cuda(@cuda.copy_h2d(ptr, data), "copy_h2d in from_host")
  { data: ptr, shape: @ts.ShapeTensor::from_array(shape) }
}

///|
pub fn GpuTensor::to_host(self : GpuTensor) -> FixedArray[Float] {
  let result = FixedArray::make(self.numel(), Float::from_double(0.0))
  check_cuda(@cuda.copy_d2h(result, self.data), "copy_d2h in to_host")
  result
}

///|
/// Extract a scalar value from a single-element GPU tensor.
/// Equivalent to PyTorch's tensor.item().
pub fn GpuTensor::item(self : GpuTensor) -> Float {
  @cuda.read_scalar(self.data)
}

///|
/// Compute argmax along the last dimension of a 2D tensor.
/// Input shape: [outer, inner], output shape: [outer].
/// Each output element is the index of the maximum value along the inner dim,
/// stored as a float (use .to_int() to convert).
pub fn GpuTensor::argmax(self : GpuTensor) -> GpuTensor {
  let ndim = self.ndim()
  if ndim != 2 {
    abort("argmax: expected 2D tensor, got \{ndim}D")
  }
  let outer = self.dim(0)
  let inner = self.dim(1)
  let output = GpuTensor::zeros_st(@ts.ShapeTensor::from_array([outer]))
  check_cuda(@cuda.argmax(self.data, output.data, outer, inner), "argmax")
  output
}

///|
/// Internal helper: create a view without shape validation.
/// Used by non-raise methods that know the shape is correct.
fn GpuTensor::_view_unchecked(
  self : GpuTensor,
  new_shape : Array[Int],
) -> GpuTensor {
  { data: self.data, shape: @ts.ShapeTensor::from_array(new_shape) }
}

///|
/// Create a view of this tensor with a different shape.
/// Shares the same GPU data pointer — preserves the underlying data.
pub fn GpuTensor::view(self : GpuTensor, new_shape : Array[Int]) -> GpuTensor {
  let new_numel = GpuTensor::shape_numel(new_shape)
  if new_numel != self.numel() {
    abort(
      "view: shape mismatch, original has \{self.numel()} elements but new shape has \{new_numel} elements",
    )
  }
  { data: self.data, shape: @ts.ShapeTensor::from_array(new_shape) }
}

///|
pub fn GpuTensor::alloc_shape(shape : Array[Int]) -> GpuTensor {
  GpuTensor::alloc_st(@ts.ShapeTensor::from_array(shape))
}

///|
pub fn GpuTensor::alloc_st(shape : @ts.ShapeTensor) -> GpuTensor {
  let n = @ts.ShapeTensor::size(shape)
  let bytes = Int64::from_int(n * 4)
  let ptr = if gpu_pool_state.use_pool {
    match gpu_pool_state.pool {
      Some(p) => @cuda.pool_alloc(p, bytes)
      None => @cuda.alloc(bytes)
    }
  } else {
    @cuda.alloc(bytes)
  }
  { data: ptr, shape }
}
