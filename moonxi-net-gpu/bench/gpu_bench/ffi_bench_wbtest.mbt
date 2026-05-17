// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
test (b : @bench.T) {
  // Pre-allocate reusable buffers
  let buf1 = @cuda.alloc(4L)

  let a1 = @gt.GpuTensor::zeros([1])
  let b1 = @gt.GpuTensor::zeros([1])

  let a1k = @gt.GpuTensor::zeros([1000])
  let b1k = @gt.GpuTensor::zeros([1000])
  let buf1k = @cuda.alloc(4000L)

  let a1m = @gt.GpuTensor::zeros([1_000_000])
  let b1m = @gt.GpuTensor::zeros([1_000_000])
  let buf1m = @cuda.alloc(4_000_000L)

  let conv_in = @gt.GpuTensor::zeros([1, 1, 1, 1])
  let conv_w = @gt.GpuTensor::zeros([1, 1, 1, 1])
  let conv_b = @gt.GpuTensor::zeros([1])
  let conv_ctx = @cuda.get_context(0)

  let mut _counter = 0
  let tape_ops : Array[() -> Unit] = Array::new()
  for _ in 0..<122 {
    tape_ops.push(fn() { _counter += 1 })
  }

  let tape_ffi_ops : Array[() -> Unit] = Array::new()
  for _ in 0..<122 {
    tape_ffi_ops.push(fn() { let _ = @cuda.memset_zero(buf1, 1L) })
  }

  b.bench(name="0.  cuda_sync (baseline)", fn() { @cuda.sync() })

  b.bench(name="1.  cuda_peek_error (FFI no-op)", fn() {
    let _ = @cuda.peek_error()
    @cuda.sync()
  })

  b.bench(name="2.  cuda_alloc(4 bytes)", fn() {
    let _ = @cuda.alloc(4L)
    @cuda.sync()
  })

  b.bench(name="3.  cuda_memset_zero(1 elem)", fn() {
    let _ = @cuda.memset_zero(buf1, 1L)
    @cuda.sync()
  })

  b.bench(name="4a. elementwise_add FFI(1 elem)", fn() {
    let _ = @cuda.elementwise_add(a1.data, b1.data, buf1, 1)
    @cuda.sync()
  })

  b.bench(name="4b. elementwise_add FFI(1K elem)", fn() {
    let _ = @cuda.elementwise_add(a1k.data, b1k.data, buf1k, 1000)
    @cuda.sync()
  })

  b.bench(name="4c. elementwise_add FFI(1M elem)", fn() {
    let _ = @cuda.elementwise_add(a1m.data, b1m.data, buf1m, 1_000_000)
    @cuda.sync()
  })

  b.bench(name="5. conv2d fwd raw FFI(1x1x1x1)", fn() {
    let cudnn_ctx = @cuda.conv2d_create(1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1)
    let out = @gt.GpuTensor::alloc_shape([1, 1, 1, 1])
    let _ = @cuda.conv2d_forward(
      cudnn_ctx,
      conv_ctx.cudnn,
      conv_in.data,
      conv_w.data,
      conv_b.data,
      out.data,
    )
    @cuda.sync()
  })

  b.bench(name="6.  122 closures (pure MoonBit, no FFI)", fn() {
    for i in 0..<122 {
      tape_ops[i]()
    }
  })

  b.bench(name="7.  122 closures + memset_zero FFI", fn() {
    for i in 0..<122 {
      tape_ffi_ops[i]()
    }
    @cuda.sync()
  })
}
