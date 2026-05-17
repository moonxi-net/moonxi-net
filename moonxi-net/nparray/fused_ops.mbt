// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Fused linear forward: output = relu(input @ weight^T + bias)
/// Combines matmul + bias add + relu into a single allocation.
/// Returns (relu_output, pre_relu) where pre_relu is needed for backward.
///
/// input:  shape [batch, in_features]
/// weight: shape [out_features, in_features]
/// bias:   shape [out_features]
pub fn fused_linear_relu(
  input~ : NpArray,
  weight~ : NpArray,
  bias~ : NpArray,
) -> (NpArray, NpArray) {
  let n = input.shape.shape[0]
  let m = input.shape.shape[1]
  let p = weight.shape.shape[0]
  let out_shape = @tensor.ShapeTensor::from_array([n, p])
  let pre_relu_data : FixedArray[Float] = FixedArray::make(n * p, (0.0 : Float))
  let output_data : FixedArray[Float] = FixedArray::make(n * p, (0.0 : Float))
  // Fused matmul + bias add + relu in one pass over each output element
  for i in 0..<n {
    let a_base = i * m
    let c_base = i * p
    for j in 0..<p {
      let mut sum = bias.data[j]
      let w_base = j * m
      for k in 0..<m {
        sum = sum + input.data[a_base + k] * weight.data[w_base + k]
      }
      pre_relu_data[c_base + j] = sum
      output_data[c_base + j] = if sum > (0.0 : Float) {
        sum
      } else {
        (0.0 : Float)
      }
    }
  }
  (
    { shape: out_shape, data: output_data },
    { shape: out_shape, data: pre_relu_data },
  )
}

///|
/// Fused linear forward (no relu): output = input @ weight^T + bias
/// Used for the final layer of MLP.
///
/// input:  shape [batch, in_features]
/// weight: shape [out_features, in_features]
/// bias:   shape [out_features]
pub fn fused_linear(
  input~ : NpArray,
  weight~ : NpArray,
  bias~ : NpArray,
) -> NpArray {
  let n = input.shape.shape[0]
  let m = input.shape.shape[1]
  let p = weight.shape.shape[0]
  let out_shape = @tensor.ShapeTensor::from_array([n, p])
  let output_data : FixedArray[Float] = FixedArray::make(n * p, (0.0 : Float))
  // Fused matmul + bias add in one pass
  for i in 0..<n {
    let a_base = i * m
    let c_base = i * p
    for j in 0..<p {
      let mut sum = bias.data[j]
      let w_base = j * m
      for k in 0..<m {
        sum = sum + input.data[a_base + k] * weight.data[w_base + k]
      }
      output_data[c_base + j] = sum
    }
  }
  { shape: out_shape, data: output_data }
}
