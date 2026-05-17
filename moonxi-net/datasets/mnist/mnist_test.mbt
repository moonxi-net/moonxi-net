// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn make_dummy_mnist(label : Int, pixel_val : Float) -> MnistExample {
  let pixels = FixedArray::makei(784, fn(_) { pixel_val })
  MnistExample::{ pixels, label }
}

///|
test "mnist_to_tensors one_hot encoding" {
  let ex = make_dummy_mnist(3, 0.5)
  let view = [ex].view()
  let (images, labels) = mnist_to_tensors(view, 0, 1)
  let _ = images
  assert_eq(labels[3], 1.0)
  assert_eq(labels[0], 0.0)
}

///|
test "mnist_to_tensors shape" {
  let ex0 = make_dummy_mnist(0, 0.1)
  let ex1 = make_dummy_mnist(5, 0.2)
  let ex2 = make_dummy_mnist(9, 0.3)
  let examples = [ex0, ex1, ex2]
  let view = examples.view()
  let (images, labels) = mnist_to_tensors(view, 0, 3)
  assert_eq(images.length(), 2352) // 3 * 784
  assert_eq(labels.length(), 30) // 3 * 10
  assert_eq(labels[0], 1.0)
  assert_eq(labels[15], 1.0) // batch 1, label 5
  assert_eq(labels[29], 1.0) // batch 2, label 9
}

///|
test "mnist_to_tensors_with_labels" {
  let ex0 = make_dummy_mnist(3, 0.5)
  let ex1 = make_dummy_mnist(7, 0.8)
  let examples = [ex0, ex1]
  let view = examples.view()
  let (images, labels) = mnist_to_tensors_with_labels(view, 0, 2)
  assert_eq(images.length(), 1568) // 2 * 784
  assert_eq(labels.length(), 2)
  assert_eq(labels[0], 3.0)
  assert_eq(labels[1], 7.0)
}

///|
test "mnist_labels length" {
  assert_eq(mnist_labels.length(), 10)
}
