// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
test (b : @bench.T) {
  // Prime conv2d_create cache for layer1 3x3
  ignore(@cuda.conv2d_create(128, 64, 16, 16, 64, 3, 3, 1, 1, 1, 1, 1, 1))

  b.bench(name="conv2d_create layer1 3x3 (no pool)", fn() {
    ignore(@cuda.conv2d_create(128, 64, 16, 16, 64, 3, 3, 1, 1, 1, 1, 1, 1))
    @cuda.sync()
  })

  // Prime conv2d_create cache for stem 7x7
  ignore(@cuda.conv2d_create(128, 3, 32, 32, 64, 7, 7, 3, 3, 2, 2, 1, 1))

  b.bench(name="conv2d_create stem 7x7 (no pool)", fn() {
    ignore(@cuda.conv2d_create(128, 3, 32, 32, 64, 7, 7, 3, 3, 2, 2, 1, 1))
    @cuda.sync()
  })

  b.bench(name="alloc_shape([128,64,16,16]) (no pool)", fn() {
    ignore(@gt.GpuTensor::alloc_shape([128, 64, 16, 16]))
    @cuda.sync()
  })

  b.bench(name="alloc_shape([64]) (no pool)", fn() {
    ignore(@gt.GpuTensor::alloc_shape([64]))
    @cuda.sync()
  })
}
