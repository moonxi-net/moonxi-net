// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// A single CIFAR-10 example.
/// pixels: [3, 32, 32] in NCHW format, normalized to [0, 1]
/// label: 0-9
pub(all) struct Cifar10Example {
  pixels : FixedArray[Float]
  label : Int
}

///|
/// Load one CIFAR-10 batch file (10000 images).
/// Returns array of 10000 examples.
pub fn load_cifar10_batch(path : String) -> Array[Cifar10Example] raise {
  let bytes = @fs.read_file_to_bytes(path) catch {
    _ => raise @tensor.IoError("Failed to read file: \{path}")
  }
  let len = bytes.length()
  let expected = 10000 * 3073
  if len < expected {
    raise @tensor.IoError(
      "CIFAR-10 batch file too small: \{path}, expected \{expected} bytes, got \{len}",
    )
  } else {
    let n = 10000
    let result = Array::make(n, Cifar10Example::{
      pixels: FixedArray::make(3072, Float::from_double(0.0)),
      label: 0,
    })
    for i in 0..<n {
      let offset = i * 3073
      let label_byte = bytes[offset]
      let label = label_byte.to_int()
      let pixels = FixedArray::makei(3072, fn(j) {
        let pixel = bytes[offset + 1 + j].to_int()
        Float::from_int(pixel) / Float::from_double(255.0)
      })
      result[i] = Cifar10Example::{ pixels, label }
    }
    result
  }
}

///|
/// Load all 5 training batches from a directory.
pub fn load_cifar10_train(dir : String) -> Array[Cifar10Example] raise {
  let all : Array[Cifar10Example] = []
  for batch_id in 1..<=5 {
    let filename = "data_batch_\{batch_id}.bin"
    let path = if dir == "" { filename } else { "\{dir}/\{filename}" }
    let batch = load_cifar10_batch(path)
    for ex in batch {
      all.push(ex)
    }
  }
  all
}

///|
/// Load test batch from a directory.
pub fn load_cifar10_test(dir : String) -> Array[Cifar10Example] raise {
  let filename = "test_batch.bin"
  let path = if dir == "" { filename } else { "\{dir}/\{filename}" }
  load_cifar10_batch(path)
}

///|
/// Convert a slice of examples to flat tensors for GPU upload.
/// Returns (images_data, labels_data) where:
///   images_data: [count * 3 * 32 * 32] pixels normalized to [0,1]
///   labels_data: [count * 10] one-hot encoded
pub fn cifar10_to_tensors(
  examples : ArrayView[Cifar10Example],
  start : Int,
  count : Int,
) -> (FixedArray[Float], FixedArray[Float]) {
  let images = FixedArray::make(count * 3 * 32 * 32, Float::from_double(0.0))
  let labels = FixedArray::make(count * 10, Float::from_double(0.0))
  for i in 0..<count {
    let ex = examples[start + i]
    for j in 0..<3072 {
      images[i * 3072 + j] = ex.pixels[j]
    }
    for k in 0..<10 {
      labels[i * 10 + k] = if k == ex.label {
        Float::from_double(1.0)
      } else {
        Float::from_double(0.0)
      }
    }
  }
  (images, labels)
}

///|
/// Convert a slice of examples to flat tensors with integer class labels.
/// Returns (images_data, labels_data) where:
///   images_data: [count * 3 * 32 * 32] pixels normalized to [0,1]
///   labels_data: [count] integer class indices stored as float (for GPU upload)
pub fn cifar10_to_tensors_with_labels(
  examples : ArrayView[Cifar10Example],
  start : Int,
  count : Int,
) -> (FixedArray[Float], FixedArray[Float]) {
  let images = FixedArray::make(count * 3 * 32 * 32, Float::from_double(0.0))
  let labels = FixedArray::make(count, Float::from_double(0.0))
  for i in 0..<count {
    let ex = examples[start + i]
    for j in 0..<3072 {
      images[i * 3072 + j] = ex.pixels[j]
    }
    labels[i] = Float::from_int(ex.label)
  }
  (images, labels)
}

///|
/// Label names for CIFAR-10 classes
pub let cifar10_labels : Array[String] = [
  "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
  "truck",
]

///|
/// Create train + test DataLoaders for CIFAR-10 with integer class labels.
/// Generic over any Tensor type that implements `from_host`.
/// Usage: `let (train_loader, test_loader) = @data.cifar10_loaders("data/cifar-10-batches-bin")`
pub fn[T : @tensor.Tensor] cifar10_loaders(
  data_dir : String,
) -> (@dl.DataLoader[T], @dl.DataLoader[T]) raise {
  let train_examples = load_cifar10_train(data_dir)
  let test_examples = load_cifar10_test(data_dir)
  let num_train = train_examples.length()
  let num_test = test_examples.length()
  let (train_images, train_labels) = cifar10_to_tensors_with_labels(
    train_examples.view(),
    0,
    num_train,
  )
  let (test_images, test_labels) = cifar10_to_tensors_with_labels(
    test_examples.view(),
    0,
    num_test,
  )
  (
    _cifar10_loader(train_images, train_labels, num_train),
    _cifar10_loader(test_images, test_labels, num_test),
  )
}

///|
/// Internal: create a single CIFAR-10 DataLoader from pre-loaded tensors.
fn[T : @tensor.Tensor] _cifar10_loader(
  images : FixedArray[Float],
  labels : FixedArray[Float],
  num_samples : Int,
) -> @dl.DataLoader[T] {
  let image_size = 3 * 32 * 32
  @dl.DataLoader::{
    num_samples,
    get: fn(idx) {
      let h_img = FixedArray::make(image_size, Float::from_double(0.0))
      let h_lbl = FixedArray::make(1, Float::from_double(0.0))
      for k in 0..<image_size {
        h_img[k] = images[idx * image_size + k]
      }
      h_lbl[0] = labels[idx]
      let img = T::from_host(h_img, [1, 3, 32, 32])
      let lbl = T::from_host(h_lbl, [1])
      (img, lbl)
    },
    batch: fn(indices) {
      let count = indices.length()
      let h_img = FixedArray::make(count * image_size, Float::from_double(0.0))
      let h_lbl = FixedArray::make(count, Float::from_double(0.0))
      for j in 0..<count {
        let idx = indices[j]
        for k in 0..<image_size {
          h_img[j * image_size + k] = images[idx * image_size + k]
        }
        h_lbl[j] = labels[idx]
      }
      let img = T::from_host(h_img, [count, 3, 32, 32])
      let lbl = T::from_host(h_lbl, [count])
      (img, lbl)
    },
  }
}
