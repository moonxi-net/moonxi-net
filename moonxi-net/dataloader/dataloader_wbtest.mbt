// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Helper: create a simple DataLoader for testing.
/// get(i) returns a scalar tensor with value float(i) for both input and target.
/// batch gathers multiple scalars into a 1d tensor.
fn make_test_loader(num_samples : Int) -> DataLoader[@nparray.NpArray] {
  {
    num_samples,
    get: fn(i : Int) -> (@nparray.NpArray, @nparray.NpArray) {
      let v = @nparray.NpArray::from1d([Float::from_int(i)])
      (v, v)
    },
    batch: fn(
      indices : ArrayView[Int],
    ) -> (@nparray.NpArray, @nparray.NpArray) {
      let data = indices.map(fn(i) { Float::from_int(i) })
      let t = @nparray.NpArray::from1d(data)
      (t, t)
    },
  }
}

///|
fn float_near(a : Float, b : Float, eps : Float) -> Bool {
  let d = a - b
  d < eps && d > 0.0 - eps
}

///|
test "DataLoader::num_samples returns correct value" {
  let loader = make_test_loader(10)
  assert_eq(loader.num_samples(), 10)
}

///|
test "DataLoader::batches yields correct number when evenly divisible" {
  let loader = make_test_loader(10)
  let batch_size = 2
  let mut count = 0
  for _ in loader.batches(batch_size) {
    count += 1
  }
  assert_eq(count, 5) // 10 / 2 = 5 batches
}

///|
test "DataLoader::batches yields correct number with remainder" {
  let loader = make_test_loader(10)
  let batch_size = 3
  let mut count = 0
  for _ in loader.batches(batch_size) {
    count += 1
  }
  assert_eq(count, 4) // 10 / 3 = 3 full + 1 partial
}

///|
test "DataLoader::batches last batch has correct size (partial)" {
  let loader = make_test_loader(7)
  let batch_size = 3
  let batch_sizes : Array[Int] = []
  for batch in loader.batches(batch_size) {
    let (input, _) = batch
    batch_sizes.push(input.dims()[0])
  }
  assert_eq(batch_sizes.length(), 3) // 2 full + 1 partial
  assert_eq(batch_sizes[0], 3)
  assert_eq(batch_sizes[1], 3)
  assert_eq(batch_sizes[2], 1) // last batch: 7 % 3 = 1
}

///|
test "DataLoader::batches without shuffle preserves order" {
  let loader = make_test_loader(6)
  let all_data : Array[Float] = []
  for batch in loader.batches(3, shuffle=false) {
    let (input, _) = batch
    for i in 0..<input.dims()[0] {
      all_data.push(input.data[i])
    }
  }
  assert_eq(all_data.length(), 6)
  for i in 0..<6 {
    assert_true(float_near(all_data[i], Float::from_int(i), 0.0001))
  }
}

///|
test "DataLoader::batches with shuffle yields same total elements" {
  let loader = make_test_loader(10)
  let batch_size = 3
  let mut total = 0
  for batch in loader.batches(batch_size, shuffle=true) {
    let (input, _) = batch
    total += input.dims()[0]
  }
  assert_eq(total, 10)
}

///|
test "DataLoader::batches single element per batch" {
  let loader = make_test_loader(4)
  let mut count = 0
  for _ in loader.batches(1) {
    count += 1
  }
  assert_eq(count, 4)
}

///|
test "DataLoader::batches when batch_size equals num_samples" {
  let loader = make_test_loader(5)
  let mut count = 0
  for _ in loader.batches(5) {
    count += 1
  }
  assert_eq(count, 1)
}
