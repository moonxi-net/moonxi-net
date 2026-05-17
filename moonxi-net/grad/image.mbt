// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// =============================================================================
// ImageTensor impl for Grad[T]
// =============================================================================

///|
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with conv2d(
  input : Grad[T],
  weight : Grad[T],
  bias : Grad[T],
  stride : Int,
  padding : Int,
) -> Grad[T] {
  let result = T::conv2d(input.value, weight.value, bias.value, stride, padding)
  let has_grad = input.grad is Some(_) ||
    weight.grad is Some(_) ||
    bias.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_weight = weight.value
  let saved_input = input.value
  record_accumulate(input, g, og => {
    T::conv2d_backward_data(og, saved_weight, saved_input, stride, padding)
  })
  let saved_input2 = input.value
  let saved_weight2 = weight.value
  record_accumulate(weight, g, og => {
    T::conv2d_backward_weight(og, saved_input2, saved_weight2, stride, padding)
  })
  record_accumulate(bias, g, og => T::conv2d_backward_bias(og))
  g
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with relu(x : Grad[T]) -> Grad[T] {
  let result = T::relu(x.value)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_x = x.value
  record_accumulate(x, g, og => T::relu_backward(og, saved_x))
  g
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with maxpool2d(x : Grad[T], kernel_size : Int, stride : Int) -> Grad[T] {
  let result = T::maxpool2d(x.value, kernel_size, stride)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_x = x.value
  record_accumulate(x, g, og => {
    T::maxpool2d_backward(og, saved_x, kernel_size, stride)
  })
  g
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with adaptive_avg_pool2d(x : Grad[T], output_size : Int) -> Grad[T] {
  let result = T::adaptive_avg_pool2d(x.value, output_size)
  let has_grad = x.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_x = x.value
  record_accumulate(x, g, og => T::adaptive_avg_pool2d_backward(og, saved_x))
  g
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with batchnorm_training(
  input : Grad[T],
  gamma : Grad[T],
  beta : Grad[T],
  running_mean : Grad[T],
  running_var : Grad[T],
  momentum : Float,
  eps : Float,
) -> (Grad[T], Grad[T], Grad[T]) {
  let (out, save_mean, save_inv_var) = T::batchnorm_training(
    input.value,
    gamma.value,
    beta.value,
    running_mean.value,
    running_var.value,
    momentum,
    eps,
  )
  let has_grad = input.grad is Some(_) ||
    gamma.grad is Some(_) ||
    beta.grad is Some(_)
  let g : Grad[T] = {
    value: out,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_input = input.value
  let saved_gamma = gamma.value
  let saved_mean = save_mean
  let saved_inv_var = save_inv_var
  let saved_eps = eps
  // Shared cache: compute batchnorm_backward once, not three times.
  // FixedArray is a reference type — mutations are visible to all closures.
  let bn_cache : FixedArray[(T, T, T)?] = FixedArray::make(1, None)
  record_accumulate(input, g, og => {
    if bn_cache[0] is None {
      bn_cache[0] = Some(
        T::batchnorm_backward(
          og, saved_input, saved_gamma, saved_mean, saved_inv_var, saved_eps,
        ),
      )
    }
    match bn_cache[0] {
      Some((di, _, _)) => di
      None => og // unreachable but needed for type
    }
  })
  record_accumulate(gamma, g, og => {
    if bn_cache[0] is None {
      bn_cache[0] = Some(
        T::batchnorm_backward(
          og, saved_input, saved_gamma, saved_mean, saved_inv_var, saved_eps,
        ),
      )
    }
    match bn_cache[0] {
      Some((_, dg, _)) => dg
      None => og
    }
  })
  record_accumulate(beta, g, og => {
    if bn_cache[0] is None {
      bn_cache[0] = Some(
        T::batchnorm_backward(
          og, saved_input, saved_gamma, saved_mean, saved_inv_var, saved_eps,
        ),
      )
    }
    match bn_cache[0] {
      Some((_, _, db)) => db
      None => og
    }
  })
  (g, no_grad(save_mean), no_grad(save_inv_var))
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with batchnorm_inference(
  input : Grad[T],
  gamma : Grad[T],
  beta : Grad[T],
  running_mean : Grad[T],
  running_var : Grad[T],
  eps : Float,
) -> Grad[T] {
  let result = T::batchnorm_inference(
    input.value,
    gamma.value,
    beta.value,
    running_mean.value,
    running_var.value,
    eps,
  )
  no_grad(result)
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with softmax_cross_entropy(logits : Grad[T], targets : Grad[T]) -> Grad[T] {
  let result = T::softmax_cross_entropy(logits.value, targets.value)
  let has_grad = logits.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_logits = logits.value
  let saved_targets = targets.value
  record_accumulate(logits, g, og => {
    let raw = T::softmax_ce_backward(saved_logits, saved_targets)
    let og_scalar = og.value().data[0]
    raw.scale(og_scalar)
  })
  g
}

///|
/// Autograd wrapper for cross_entropy_with_labels.
/// labels has no gradient — only logits gets gradients.
pub impl[T : @tensor.Tensor + @tensor.ImageTensor + @tensor.ImageBackwardOps] @tensor.ImageTensor for Grad[
  T,
] with cross_entropy_with_labels(
  logits : Grad[T],
  labels : Grad[T],
  num_classes : Int,
) -> Grad[T] {
  let result = T::cross_entropy_with_labels(
    logits.value,
    labels.value,
    num_classes,
  )
  let has_grad = logits.grad is Some(_)
  let g : Grad[T] = {
    value: result,
    grad: if has_grad {
      Some(None)
    } else {
      None
    },
  }
  let saved_logits = logits.value
  let saved_labels = labels.value
  record_accumulate(logits, g, og => {
    let raw = T::softmax_ce_backward_labels(
      saved_logits, saved_labels, num_classes,
    )
    let og_scalar = og.value().data[0]
    raw.scale(og_scalar)
  })
  g
}

// =============================================================================
// ImageBackwardOps impl for Grad[T] — dead code in autograd flow,
// exist only to satisfy the trait. Use raw T backward ops in record closures.
// =============================================================================

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with conv2d_backward_data(
  grad_output : Grad[T],
  weight : Grad[T],
  input : Grad[T],
  stride : Int,
  padding : Int,
) -> Grad[T] {
  let g = match grad_output.grad {
    Some(Some(g)) => g
    Some(None) | None => T::zeros_like(grad_output.value)
  }
  no_grad(
    T::conv2d_backward_data(g, weight.value, input.value, stride, padding),
  )
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with conv2d_backward_weight(
  grad_output : Grad[T],
  input : Grad[T],
  weight : Grad[T],
  stride : Int,
  padding : Int,
) -> Grad[T] {
  let g = match grad_output.grad {
    Some(Some(g)) => g
    Some(None) | None => T::zeros_like(grad_output.value)
  }
  no_grad(
    T::conv2d_backward_weight(g, input.value, weight.value, stride, padding),
  )
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with conv2d_backward_bias(grad_output : Grad[T]) -> Grad[T] {
  let g = match grad_output.grad {
    Some(Some(g)) => g
    Some(None) | None => T::zeros_like(grad_output.value)
  }
  no_grad(T::conv2d_backward_bias(g))
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with relu_backward(grad_output : Grad[T], input : Grad[T]) -> Grad[T] {
  let g = match grad_output.grad {
    Some(Some(g)) => g
    Some(None) | None => T::zeros_like(grad_output.value)
  }
  no_grad(T::relu_backward(g, input.value))
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with maxpool2d_backward(
  grad_output : Grad[T],
  input : Grad[T],
  kernel_size : Int,
  stride : Int,
) -> Grad[T] {
  let g = match grad_output.grad {
    Some(Some(g)) => g
    Some(None) | None => T::zeros_like(grad_output.value)
  }
  no_grad(T::maxpool2d_backward(g, input.value, kernel_size, stride))
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with adaptive_avg_pool2d_backward(grad_output : Grad[T], input : Grad[T]) -> Grad[
  T,
] {
  let g = match grad_output.grad {
    Some(Some(g)) => g
    Some(None) | None => T::zeros_like(grad_output.value)
  }
  no_grad(T::adaptive_avg_pool2d_backward(g, input.value))
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with batchnorm_backward(
  grad_output : Grad[T],
  input : Grad[T],
  gamma : Grad[T],
  save_mean : Grad[T],
  save_inv_var : Grad[T],
  eps : Float,
) -> (Grad[T], Grad[T], Grad[T]) {
  let g = match grad_output.grad {
    Some(Some(g)) => g
    Some(None) | None => T::zeros_like(grad_output.value)
  }
  let (d_input, d_gamma, d_beta) = T::batchnorm_backward(
    g,
    input.value,
    gamma.value,
    save_mean.value,
    save_inv_var.value,
    eps,
  )
  (no_grad(d_input), no_grad(d_gamma), no_grad(d_beta))
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with softmax_ce_backward(logits : Grad[T], targets : Grad[T]) -> Grad[T] {
  no_grad(T::softmax_ce_backward(logits.value, targets.value))
}

///|
pub impl[T : @tensor.Tensor + @tensor.ImageBackwardOps] @tensor.ImageBackwardOps for Grad[
  T,
] with softmax_ce_backward_labels(
  logits : Grad[T],
  labels : Grad[T],
  num_classes : Int,
) -> Grad[T] {
  no_grad(
    T::softmax_ce_backward_labels(logits.value, labels.value, num_classes),
  )
}
