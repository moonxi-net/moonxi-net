// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Generic DataLoader. Closures capture dataset-specific gather + transfer logic.
///
/// `T` is the tensor type (e.g. `NpArray` for CPU, `GpuTensor` for GPU).
/// Each DataLoader provides:
/// - `num_samples`: total number of samples in the dataset
/// - `get`: retrieve a single sample by index → `(input, target)`
/// - `batch`: gather multiple samples by indices → `(input_batch, target_batch)`
///
/// Use `batches()` to iterate over mini-batches with optional shuffling.
pub(all) struct DataLoader[T] {
  num_samples : Int
  /// Get a single sample by index → (input, target)
  get : (Int) -> (T, T)
  /// Gather multiple samples by indices → (input_batch, target_batch)
  batch : (ArrayView[Int]) -> (T, T)
}

///|
pub fn[T] DataLoader::num_samples(self : DataLoader[T]) -> Int {
  self.num_samples
}

///|
/// Iterate over batches: shuffle indices, split into chunks, call self.batch.
/// Yields (input_batch, target_batch) for each chunk.
pub fn[T] DataLoader::batches(
  self : DataLoader[T],
  batch_size : Int,
  shuffle? : Bool = false,
) -> Iter[(T, T)] {
  let indices = Array::makei(self.num_samples, fn(i) { i })
  let ordered = if shuffle { _fisher_yates_shuffle(indices) } else { indices }
  let full = self.num_samples / batch_size
  let rem = self.num_samples % batch_size
  let total = full + (if rem > 0 { 1 } else { 0 })
  let mut cursor = 0
  Iter::new(fn() {
    if cursor >= total {
      None
    } else {
      let count = if cursor < full { batch_size } else { rem }
      let start = cursor * batch_size
      let chunk = ordered[start:start + count]
      let result = (self.batch)(chunk)
      cursor = cursor + 1
      Some(result)
    }
  })
}

///|
fn _fisher_yates_shuffle(arr : Array[Int]) -> Array[Int] {
  let result = arr.copy()
  let rng = @random.Rand::new()
  let mut i = result.length() - 1
  while i > 0 {
    let j = rng.int(limit=i + 1)
    let tmp = result[i]
    result[i] = result[j]
    result[j] = tmp
    i -= 1
  }
  result
}
