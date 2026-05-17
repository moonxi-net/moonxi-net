// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
pub(all) struct EventProfiler {
  mut times : Map[String, (Float, Int)]
}

///|
let global_profiler : EventProfiler = { times: {} }

///|
pub fn EventProfiler::new() -> EventProfiler {
  { times: {} }
}

///|
pub fn EventProfiler::global() -> EventProfiler {
  global_profiler
}

///|
pub fn EventProfiler::reset(self : EventProfiler) -> Unit {
  self.times = {}
}

///|
pub fn EventProfiler::record(
  self : EventProfiler,
  op : String,
  ms : Float,
) -> Unit {
  match self.times.get(op) {
    Some((total, count)) => self.times[op] = (total + ms, count + 1)
    None => self.times[op] = (ms, 1)
  }
}

///|
pub fn EventProfiler::print(self : EventProfiler) -> Unit {
  let sb = StringBuilder::new()
  sb.write_char('\n')
  sb.write_view("=== CUDA Event Profiler Results ===")
  sb.write_char('\n')
  let ops = []
  for k, _ in self.times {
    ops.push(k)
  }
  ops.sort()
  for op in ops {
    match self.times.get(op) {
      Some((total, count)) =>
        sb.write_view(
          "\n\{op}: \{total} ms total, \{count} calls, avg: \{total / Float::from_int(count)} ms",
        )
      None => ()
    }
  }
  sb.write_char('\n')
  println(sb.to_string())
}

///|
fn time_op(op : String, f : () -> @gt.GpuTensor) -> @gt.GpuTensor {
  let profiler = EventProfiler::global()
  let start = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), start)
  let result = f()
  let end = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), end)
  let _ = @cuda.event_synchronize(end)
  let ms = @cuda.event_elapsed_ms(start, end)
  profiler.record(op, ms)
  result
}

///|
fn time_op_unit(op : String, f : () -> Unit) -> Unit {
  let profiler = EventProfiler::global()
  let start = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), start)
  f()
  let end = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), end)
  let _ = @cuda.event_synchronize(end)
  let ms = @cuda.event_elapsed_ms(start, end)
  profiler.record(op, ms)
}

///|
pub(all) struct CudaEventTensor {
  inner : @gt.GpuTensor
}

///|
pub impl @tensor.Tensor for CudaEventTensor with dims(self : CudaEventTensor) -> FixedArray[
  Int,
] {
  @gt.GpuTensor::dims(self.inner)
}

///|
pub impl @tensor.Tensor for CudaEventTensor with zeros(dims : FixedArray[Int]) -> CudaEventTensor {
  { inner: @gt.GpuTensor::zeros(dims.iter().collect()) }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with zeros_like(
  self : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("zeros_like", fn() { @gt.GpuTensor::zeros_like(self.inner) }),
  }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with from_host(
  data : FixedArray[Float],
  shape : Array[Int],
) -> CudaEventTensor {
  { inner: @gt.GpuTensor::from_host(data, shape) }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with square(self : CudaEventTensor) -> CudaEventTensor {
  { inner: time_op("square", fn() { @gt.GpuTensor::square(self.inner) }) }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with sqrt(self : CudaEventTensor) -> CudaEventTensor {
  { inner: time_op("sqrt", fn() { @gt.GpuTensor::sqrt(self.inner) }) }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with mean(self : CudaEventTensor) -> CudaEventTensor {
  { inner: time_op("mean", fn() { @gt.GpuTensor::mean(self.inner) }) }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with scale(
  self : CudaEventTensor,
  s : Float,
) -> CudaEventTensor {
  { inner: time_op("scale", fn() { @gt.GpuTensor::scale(self.inner, s) }) }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with mul_elem(
  self : CudaEventTensor,
  other : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("mul_elem", fn() {
      @gt.GpuTensor::mul_elem(self.inner, other.inner)
    }),
  }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with div_elem(
  self : CudaEventTensor,
  other : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("div_elem", fn() {
      @gt.GpuTensor::div_elem(self.inner, other.inner)
    }),
  }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with scalar(v : Float) -> CudaEventTensor {
  { inner: @gt.GpuTensor::scalar(v) }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with size(self : CudaEventTensor) -> Int {
  @gt.GpuTensor::size(self.inner)
}

///|
pub impl @tensor.Tensor for CudaEventTensor with reduce_sum_to(
  self : CudaEventTensor,
  target_dims : FixedArray[Int],
) -> CudaEventTensor {
  {
    inner: time_op("reduce_sum_to", fn() {
      @gt.GpuTensor::reduce_sum_to(self.inner, target_dims)
    }),
  }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with value(self : CudaEventTensor) -> @tensor.TensorData {
  @gt.GpuTensor::value(self.inner)
}

///|
pub impl @tensor.Tensor for CudaEventTensor with view(
  self : CudaEventTensor,
  new_shape : FixedArray[Int],
) -> CudaEventTensor {
  {
    inner: time_op("view", fn() {
      @gt.GpuTensor::view(self.inner, new_shape.iter().collect())
    }),
  }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with add(
  self : CudaEventTensor,
  other : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("add", fn() { @gt.GpuTensor::add(self.inner, other.inner) }),
  }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with add_into(
  self : CudaEventTensor,
  other : CudaEventTensor,
) -> Unit {
  time_op_unit("add_into", fn() {
    @gt.GpuTensor::add_into(self.inner, other.inner)
  })
}

///|
pub impl @tensor.Tensor for CudaEventTensor with broadcast_to(
  self : CudaEventTensor,
  target_shape : FixedArray[Int],
) -> CudaEventTensor {
  {
    inner: time_op("broadcast_to", fn() {
      @gt.GpuTensor::broadcast_to(self.inner, target_shape)
    }),
  }
}

///|
pub impl @tensor.Tensor for CudaEventTensor with sub(
  self : CudaEventTensor,
  other : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("sub", fn() { @gt.GpuTensor::sub(self.inner, other.inner) }),
  }
}

///|
pub impl @tensor.BlasTensor for CudaEventTensor with matmul(
  self : CudaEventTensor,
  other : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("matmul", fn() {
      @gt.GpuTensor::matmul(self.inner, other.inner)
    }),
  }
}

///|
pub impl @tensor.BlasTensor for CudaEventTensor with transpose(
  self : CudaEventTensor,
) -> CudaEventTensor {
  { inner: time_op("transpose", fn() { @gt.GpuTensor::transpose(self.inner) }) }
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with conv2d(
  self : CudaEventTensor,
  weight : CudaEventTensor,
  bias : CudaEventTensor,
  stride : Int,
  padding : Int,
) -> CudaEventTensor {
  {
    inner: time_op("conv2d", fn() {
      @gt.GpuTensor::conv2d(
        self.inner,
        weight.inner,
        bias.inner,
        stride,
        padding,
      )
    }),
  }
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with relu(
  self : CudaEventTensor,
) -> CudaEventTensor {
  { inner: time_op("relu", fn() { @gt.GpuTensor::relu(self.inner) }) }
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with maxpool2d(
  self : CudaEventTensor,
  kernel_size : Int,
  stride : Int,
) -> CudaEventTensor {
  {
    inner: time_op("maxpool2d", fn() {
      @gt.GpuTensor::maxpool2d(self.inner, kernel_size, stride)
    }),
  }
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with adaptive_avg_pool2d(
  self : CudaEventTensor,
  output_size : Int,
) -> CudaEventTensor {
  {
    inner: time_op("adaptive_avg_pool2d", fn() {
      @gt.GpuTensor::adaptive_avg_pool2d(self.inner, output_size)
    }),
  }
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with batchnorm_training(
  self : CudaEventTensor,
  gamma : CudaEventTensor,
  beta : CudaEventTensor,
  running_mean : CudaEventTensor,
  running_var : CudaEventTensor,
  momentum : Float,
  eps : Float,
) -> (CudaEventTensor, CudaEventTensor, CudaEventTensor) {
  let profiler = EventProfiler::global()
  let start = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), start)
  let (out, save_mean, save_inv_var) = @gt.GpuTensor::batchnorm_training(
    self.inner,
    gamma.inner,
    beta.inner,
    running_mean.inner,
    running_var.inner,
    momentum,
    eps,
  )
  let end = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), end)
  let _ = @cuda.event_synchronize(end)
  let ms = @cuda.event_elapsed_ms(start, end)
  profiler.record("batchnorm_training", ms)
  ({ inner: out }, { inner: save_mean }, { inner: save_inv_var })
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with batchnorm_inference(
  self : CudaEventTensor,
  gamma : CudaEventTensor,
  beta : CudaEventTensor,
  running_mean : CudaEventTensor,
  running_var : CudaEventTensor,
  eps : Float,
) -> CudaEventTensor {
  {
    inner: time_op("batchnorm_inference", fn() {
      @gt.GpuTensor::batchnorm_inference(
        self.inner,
        gamma.inner,
        beta.inner,
        running_mean.inner,
        running_var.inner,
        eps,
      )
    }),
  }
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with softmax_cross_entropy(
  self : CudaEventTensor,
  targets : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("softmax_cross_entropy", fn() {
      @gt.GpuTensor::softmax_cross_entropy(self.inner, targets.inner)
    }),
  }
}

///|
pub impl @tensor.ImageTensor for CudaEventTensor with cross_entropy_with_labels(
  self : CudaEventTensor,
  labels : CudaEventTensor,
  num_classes : Int,
) -> CudaEventTensor {
  {
    inner: time_op("cross_entropy_with_labels", fn() {
      @gt.GpuTensor::cross_entropy_with_labels(
        self.inner,
        labels.inner,
        num_classes,
      )
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with conv2d_backward_data(
  self : CudaEventTensor,
  weight : CudaEventTensor,
  input : CudaEventTensor,
  stride : Int,
  padding : Int,
) -> CudaEventTensor {
  {
    inner: time_op("conv2d_bwd_data", fn() {
      @gt.GpuTensor::conv2d_backward_data(
        self.inner,
        weight.inner,
        input.inner,
        stride,
        padding,
      )
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with conv2d_backward_weight(
  self : CudaEventTensor,
  input : CudaEventTensor,
  weight : CudaEventTensor,
  stride : Int,
  padding : Int,
) -> CudaEventTensor {
  {
    inner: time_op("conv2d_bwd_weight", fn() {
      @gt.GpuTensor::conv2d_backward_weight(
        self.inner,
        input.inner,
        weight.inner,
        stride,
        padding,
      )
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with conv2d_backward_bias(
  self : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("conv2d_bwd_bias", fn() {
      @gt.GpuTensor::conv2d_backward_bias(self.inner)
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with relu_backward(
  self : CudaEventTensor,
  input : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("relu_backward", fn() {
      @gt.GpuTensor::relu_backward(self.inner, input.inner)
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with batchnorm_backward(
  self : CudaEventTensor,
  input : CudaEventTensor,
  gamma : CudaEventTensor,
  save_mean : CudaEventTensor,
  save_inv_var : CudaEventTensor,
  eps : Float,
) -> (CudaEventTensor, CudaEventTensor, CudaEventTensor) {
  let profiler = EventProfiler::global()
  let start = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), start)
  let (grad_input, grad_gamma, grad_beta) = @gt.GpuTensor::batchnorm_backward(
    self.inner,
    input.inner,
    gamma.inner,
    save_mean.inner,
    save_inv_var.inner,
    eps,
  )
  let end = @cuda.event_create()
  let _ = @cuda.event_record(@cuda.null_buffer(), end)
  let _ = @cuda.event_synchronize(end)
  let ms = @cuda.event_elapsed_ms(start, end)
  profiler.record("batchnorm_backward", ms)
  ({ inner: grad_input }, { inner: grad_gamma }, { inner: grad_beta })
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with maxpool2d_backward(
  self : CudaEventTensor,
  input : CudaEventTensor,
  kernel_size : Int,
  stride : Int,
) -> CudaEventTensor {
  {
    inner: time_op("maxpool2d_backward", fn() {
      @gt.GpuTensor::maxpool2d_backward(
        self.inner,
        input.inner,
        kernel_size,
        stride,
      )
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with adaptive_avg_pool2d_backward(
  self : CudaEventTensor,
  input : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("adaptive_avg_pool2d_bwd", fn() {
      @gt.GpuTensor::adaptive_avg_pool2d_backward(self.inner, input.inner)
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with softmax_ce_backward(
  self : CudaEventTensor,
  targets : CudaEventTensor,
) -> CudaEventTensor {
  {
    inner: time_op("softmax_ce_backward", fn() {
      @gt.GpuTensor::softmax_ce_backward(self.inner, targets.inner)
    }),
  }
}

///|
pub impl @tensor.ImageBackwardOps for CudaEventTensor with softmax_ce_backward_labels(
  self : CudaEventTensor,
  labels : CudaEventTensor,
  num_classes : Int,
) -> CudaEventTensor {
  {
    inner: time_op("softmax_ce_backward_labels", fn() {
      @gt.GpuTensor::softmax_ce_backward_labels(
        self.inner,
        labels.inner,
        num_classes,
      )
    }),
  }
}
