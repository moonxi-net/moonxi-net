// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// A single MNIST example.
/// pixels: [784] (28x28 flattened), normalized to [0, 1]
/// label: 0-9
pub(all) struct MnistExample {
  pixels : FixedArray[Float]
  label : Int
}

///|
/// Read a big-endian 32-bit integer from bytes at given offset.
fn be32(bytes : Bytes, offset : Int) -> Int {
  (bytes[offset].to_int() << 24) |
  (bytes[offset + 1].to_int() << 16) |
  (bytes[offset + 2].to_int() << 8) |
  bytes[offset + 3].to_int()
}

///|
/// Parse MNIST IDX image file header.
/// Returns (count, rows, cols).
fn parse_idx_image_header(bytes : Bytes) -> (Int, Int, Int) {
  (be32(bytes, 4), be32(bytes, 8), be32(bytes, 12))
}

///|
/// Load MNIST pixel data from IDX file.
/// Returns array of examples with normalized pixels.
pub fn load_mnist_images(
  image_path : String,
  label_path : String,
) -> Array[MnistExample] raise {
  let img_bytes = @fs.read_file_to_bytes(image_path)
  let lbl_bytes = @fs.read_file_to_bytes(label_path)
  let (count, rows, cols) = parse_idx_image_header(img_bytes)
  let pixel_count = rows * cols // 784 for MNIST

  let result = Array::makei(count, fn(i) {
    let label = lbl_bytes[8 + i].to_int()
    let pixels = FixedArray::makei(pixel_count, fn(j) {
      let pixel = img_bytes[16 + i * pixel_count + j].to_int()
      Float::from_int(pixel) / Float::from_double(255.0)
    })
    MnistExample::{ pixels, label }
  })
  result
}

///|
/// Convert MNIST examples to flat tensors for training.
/// Returns (images_data [count * 784], labels_data [count * 10] one-hot)
pub fn mnist_to_tensors(
  examples : ArrayView[MnistExample],
  start : Int,
  count : Int,
) -> (FixedArray[Float], FixedArray[Float]) {
  let features = 784
  let num_classes = 10
  let images = FixedArray::make(count * features, Float::from_double(0.0))
  let labels = FixedArray::make(count * num_classes, Float::from_double(0.0))
  for i in 0..<count {
    let ex = examples[start + i]
    for j in 0..<features {
      images[i * features + j] = ex.pixels[j]
    }
    for k in 0..<num_classes {
      labels[i * num_classes + k] = if k == ex.label {
        Float::from_double(1.0)
      } else {
        Float::from_double(0.0)
      }
    }
  }
  (images, labels)
}

///|
/// Convert MNIST examples to flat tensors with integer class labels.
/// Returns (images_data [count * 784], labels_data [count] integer labels as float)
pub fn mnist_to_tensors_with_labels(
  examples : ArrayView[MnistExample],
  start : Int,
  count : Int,
) -> (FixedArray[Float], FixedArray[Float]) {
  let features = 784
  let images = FixedArray::make(count * features, Float::from_double(0.0))
  let labels = FixedArray::make(count, Float::from_double(0.0))
  for i in 0..<count {
    let ex = examples[start + i]
    for j in 0..<features {
      images[i * features + j] = ex.pixels[j]
    }
    labels[i] = Float::from_int(ex.label)
  }
  (images, labels)
}

///|
/// Create train + test DataLoaders for MNIST with integer class labels.
/// Generic over any Tensor type that implements `from_host`.
pub fn[T : @tensor.Tensor] mnist_loaders(
  data_dir : String,
) -> (@dl.DataLoader[T], @dl.DataLoader[T]) raise {
  let train_img_path = if data_dir == "" {
    "train-images-idx3-ubyte"
  } else {
    "\{data_dir}/train-images-idx3-ubyte"
  }
  let train_lbl_path = if data_dir == "" {
    "train-labels-idx1-ubyte"
  } else {
    "\{data_dir}/train-labels-idx1-ubyte"
  }
  let test_img_path = if data_dir == "" {
    "t10k-images-idx3-ubyte"
  } else {
    "\{data_dir}/t10k-images-idx3-ubyte"
  }
  let test_lbl_path = if data_dir == "" {
    "t10k-labels-idx1-ubyte"
  } else {
    "\{data_dir}/t10k-labels-idx1-ubyte"
  }

  let train_examples = load_mnist_images(train_img_path, train_lbl_path)
  let test_examples = load_mnist_images(test_img_path, test_lbl_path)
  let num_train = train_examples.length()
  let num_test = test_examples.length()

  let (train_images, train_labels) = mnist_to_tensors_with_labels(
    train_examples.view(),
    0,
    num_train,
  )
  let (test_images, test_labels) = mnist_to_tensors_with_labels(
    test_examples.view(),
    0,
    num_test,
  )

  (
    _mnist_loader(train_images, train_labels, num_train),
    _mnist_loader(test_images, test_labels, num_test),
  )
}

///|
/// Internal: create a single MNIST DataLoader from pre-loaded tensors.
fn[T : @tensor.Tensor] _mnist_loader(
  images : FixedArray[Float],
  labels : FixedArray[Float],
  num_samples : Int,
) -> @dl.DataLoader[T] {
  let features = 784
  @dl.DataLoader::{
    num_samples,
    get: fn(idx) {
      let h_img = FixedArray::make(features, Float::from_double(0.0))
      let h_lbl = FixedArray::make(1, Float::from_double(0.0))
      for k in 0..<features {
        h_img[k] = images[idx * features + k]
      }
      h_lbl[0] = labels[idx]
      let img = T::from_host(h_img, [1, 784])
      let lbl = T::from_host(h_lbl, [1])
      (img, lbl)
    },
    batch: fn(indices) {
      let count = indices.length()
      let h_img = FixedArray::make(count * features, Float::from_double(0.0))
      let h_lbl = FixedArray::make(count, Float::from_double(0.0))
      for j in 0..<count {
        let idx = indices[j]
        for k in 0..<features {
          h_img[j * features + k] = images[idx * features + k]
        }
        h_lbl[j] = labels[idx]
      }
      let img = T::from_host(h_img, [count, 784])
      let lbl = T::from_host(h_lbl, [count])
      (img, lbl)
    },
  }
}

///|
/// Label names for MNIST digit classes
pub let mnist_labels : Array[String] = [
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]
