// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|

///|
/// Helper: create a dummy Cifar10Example with given label and pixel pattern.
fn make_dummy_example(label : Int, pixel_val : Float) -> Cifar10Example {
  let pixels = FixedArray::makei(3072, fn(_) { pixel_val })
  Cifar10Example::{ pixels, label }
}

///|
test "one_hot encoding correctness" {
  // Create a single dummy example
  let ex = make_dummy_example(3, 0.5)
  // Convert to tensors with count=1
  let view = [ex].view()
  let (images, labels) = cifar10_to_tensors(view, 0, 1)
  let _ = images // unused
  // Label 3 should be one-hot as [0,0,0,1,0,0,0,0,0,0]
  assert_eq(labels[0], 0.0)
  assert_eq(labels[1], 0.0)
  assert_eq(labels[2], 0.0)
  assert_eq(labels[3], 1.0)
  assert_eq(labels[4], 0.0)
  assert_eq(labels[5], 0.0)
  assert_eq(labels[6], 0.0)
  assert_eq(labels[7], 0.0)
  assert_eq(labels[8], 0.0)
  assert_eq(labels[9], 0.0)
}

///|
test "cifar10_to_tensors shape" {
  // Create 3 dummy examples with different labels
  let ex0 = make_dummy_example(0, 0.1)
  let ex1 = make_dummy_example(5, 0.2)
  let ex2 = make_dummy_example(9, 0.3)
  let examples = [ex0, ex1, ex2]
  let view = examples.view()
  // Convert all 3
  let (images, labels) = cifar10_to_tensors(view, 0, 3)
  // images: [batch * 3 * 32 * 32] = [3 * 3072] = 9216
  assert_eq(images.length(), 9216)
  // labels: [batch * 10] = [3 * 10] = 30
  assert_eq(labels.length(), 30)
  // Verify label 0 is one-hot at index 0
  assert_eq(labels[0], 1.0)
  assert_eq(labels[1], 0.0)
  // Verify label 5 is one-hot at index 5 (offset 1)
  assert_eq(labels[11], 0.0)
  assert_eq(labels[15], 1.0)
  // Verify label 9 is one-hot at index 9 (offset 2)
  assert_eq(labels[29], 1.0)
  // Verify pixel values are preserved
  assert_eq(images[0], 0.1)
  assert_eq(images[3072], 0.2)
  assert_eq(images[6144], 0.3)
}

///|
test "cifar10_to_tensors partial slice" {
  // Create 5 dummy examples
  let examples = Array::makei(5, fn(i) {
    make_dummy_example(i, Float::from_int(i + 1) * Float::from_double(0.1))
  })
  let view = examples.view()
  // Extract only 2 examples starting at index 1 (labels 1 and 2)
  let (images, labels) = cifar10_to_tensors(view, 1, 2)
  // Verify shape
  assert_eq(images.length(), 6144)
  assert_eq(labels.length(), 20)
  // Verify example 1 (label=1): one-hot at batch_index*10 + label = 0*10 + 1 = 1
  assert_eq(labels[1], 1.0)
  // Verify example 2 (label=2): one-hot at batch_index*10 + label = 1*10 + 2 = 12
  assert_eq(labels[12], 1.0)
  // Verify pixel values
  assert_eq(images[0], 0.2)
}

///|
test "cifar10_labels length and content" {
  assert_eq(cifar10_labels.length(), 10)
  assert_eq(cifar10_labels[0], "airplane")
  assert_eq(cifar10_labels[9], "truck")
}
