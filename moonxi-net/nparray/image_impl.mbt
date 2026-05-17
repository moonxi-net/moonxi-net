// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub impl @tensor.ImageTensor for NpArray with conv2d(
  _input : NpArray,
  _weight : NpArray,
  _bias : NpArray,
  _stride : Int,
  _padding : Int,
) -> NpArray {
  abort("NpArray: conv2d not supported on CPU backend")
}

///|
pub impl @tensor.ImageTensor for NpArray with relu(x : NpArray) -> NpArray {
  let n = x.data.length()
  {
    shape: _copy_shape(x.shape),
    data: FixedArray::makei(n, fn(i) {
      if x.data[i] > 0.0 {
        x.data[i]
      } else {
        0.0
      }
    }),
  }
}

///|
pub impl @tensor.ImageTensor for NpArray with maxpool2d(
  _x : NpArray,
  _kernel_size : Int,
  _stride : Int,
) -> NpArray {
  abort("NpArray: maxpool2d not supported on CPU backend")
}

///|
pub impl @tensor.ImageTensor for NpArray with adaptive_avg_pool2d(
  _x : NpArray,
  _output_size : Int,
) -> NpArray {
  abort("NpArray: adaptive_avg_pool2d not supported on CPU backend")
}

///|
pub impl @tensor.ImageTensor for NpArray with batchnorm_training(
  _input : NpArray,
  _gamma : NpArray,
  _beta : NpArray,
  _running_mean : NpArray,
  _running_var : NpArray,
  _momentum : Float,
  _eps : Float,
) -> (NpArray, NpArray, NpArray) {
  abort("NpArray: batchnorm_training not supported on CPU backend")
}

///|
pub impl @tensor.ImageTensor for NpArray with batchnorm_inference(
  _input : NpArray,
  _gamma : NpArray,
  _beta : NpArray,
  _running_mean : NpArray,
  _running_var : NpArray,
  _eps : Float,
) -> NpArray {
  abort("NpArray: batchnorm_inference not supported on CPU backend")
}

///|
pub impl @tensor.ImageTensor for NpArray with softmax_cross_entropy(
  _logits : NpArray,
  _targets : NpArray,
) -> NpArray {
  abort(
    "NpArray: softmax_cross_entropy not supported — use cross_entropy_with_labels",
  )
}

///|
pub impl @tensor.ImageTensor for NpArray with cross_entropy_with_labels(
  logits : NpArray,
  labels : NpArray,
  num_classes : Int,
) -> NpArray {
  let batch = logits.shape.shape[0]
  let mut total_loss : Float = 0.0
  for i in 0..<batch {
    let label = labels.data[i].to_int()
    let mut max_val : Float = logits.data[i * num_classes]
    for j in 1..<num_classes {
      if logits.data[i * num_classes + j] > max_val {
        max_val = logits.data[i * num_classes + j]
      }
    }
    let mut sum_exp : Float = 0.0
    for j in 0..<num_classes {
      sum_exp = sum_exp +
        Float::from_double(
          @math.exp((logits.data[i * num_classes + j] - max_val).to_double()),
        )
    }
    let log_sum_exp = Float::from_double(@math.ln(sum_exp.to_double()))
    let log_prob = logits.data[i * num_classes + label] - max_val - log_sum_exp
    total_loss = total_loss - log_prob
  }
  let loss_val = total_loss / Float::from_int(batch)
  {
    shape: @tensor.ShapeTensor::from_array([1, 1]),
    data: FixedArray::make(1, loss_val),
  }
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with conv2d_backward_data(
  _grad_output : NpArray,
  _weight : NpArray,
  _input : NpArray,
  _stride : Int,
  _padding : Int,
) -> NpArray {
  abort("NpArray: conv2d_backward_data not supported on CPU backend")
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with conv2d_backward_weight(
  _grad_output : NpArray,
  _input : NpArray,
  _weight : NpArray,
  _stride : Int,
  _padding : Int,
) -> NpArray {
  abort("NpArray: conv2d_backward_weight not supported on CPU backend")
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with conv2d_backward_bias(
  _grad_output : NpArray,
) -> NpArray {
  abort("NpArray: conv2d_backward_bias not supported on CPU backend")
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with relu_backward(
  grad_output : NpArray,
  input : NpArray,
) -> NpArray {
  let n = input.data.length()
  let data = FixedArray::makei(n, fn(i) {
    if input.data[i] > (0.0 : Float) {
      grad_output.data[i]
    } else {
      (0.0 : Float)
    }
  })
  { shape: _copy_shape(input.shape), data }
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with batchnorm_backward(
  _grad_output : NpArray,
  _input : NpArray,
  _gamma : NpArray,
  _save_mean : NpArray,
  _save_inv_var : NpArray,
  _eps : Float,
) -> (NpArray, NpArray, NpArray) {
  abort("NpArray: batchnorm_backward not supported on CPU backend")
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with maxpool2d_backward(
  _grad_output : NpArray,
  _input : NpArray,
  _kernel_size : Int,
  _stride : Int,
) -> NpArray {
  abort("NpArray: maxpool2d_backward not supported on CPU backend")
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with adaptive_avg_pool2d_backward(
  _grad_output : NpArray,
  _input : NpArray,
) -> NpArray {
  abort("NpArray: adaptive_avg_pool2d_backward not supported on CPU backend")
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with softmax_ce_backward(
  _logits : NpArray,
  _targets : NpArray,
) -> NpArray {
  abort(
    "NpArray: softmax_ce_backward not supported — use softmax_ce_backward_labels",
  )
}

///|
pub impl @tensor.ImageBackwardOps for NpArray with softmax_ce_backward_labels(
  logits : NpArray,
  labels : NpArray,
  num_classes : Int,
) -> NpArray {
  let batch = labels.data.length()
  let result = FixedArray::make(batch * num_classes, (0.0 : Float))
  for i in 0..<batch {
    let label = labels.data[i].to_int()
    let mut max_val : Float = logits.data[i * num_classes]
    for j in 1..<num_classes {
      if logits.data[i * num_classes + j] > max_val {
        max_val = logits.data[i * num_classes + j]
      }
    }
    let mut sum_exp : Float = 0.0
    for j in 0..<num_classes {
      sum_exp = sum_exp +
        Float::from_double(
          @math.exp((logits.data[i * num_classes + j] - max_val).to_double()),
        )
    }
    for j in 0..<num_classes {
      let softmax_val = Float::from_double(
          @math.exp((logits.data[i * num_classes + j] - max_val).to_double()),
        ) /
        sum_exp
      let target : Float = if j == label {
        (1.0 : Float)
      } else {
        (0.0 : Float)
      }
      result[i * num_classes + j] = (softmax_val - target) /
        Float::from_int(batch)
    }
  }
  { shape: @tensor.ShapeTensor::from_array([batch, num_classes]), data: result }
}
