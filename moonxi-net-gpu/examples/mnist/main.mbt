// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn[
  T : @tensor.Tensor + @tensor.BlasTensor + @tensor.ImageTensor + @tensor.ImageBackwardOps + @optimizer.BatchSgdMomentumAutoclip,
] mnist_setup(
  _dummy : T,
) -> (@train.Experiment[T], @dl.DataLoader[T], @dl.DataLoader[T]) raise {
  let (train_loader, test_loader) = @data.mnist_loaders("data")
  (
    @train.Experiment::new(
      model=@model.init_mlp().as_module(),
      loss=@loss.cross_entropy_loss(10),
      optim=fn(ps : Array[@grad.Grad[T]]) {
        @optimizer.momentum_sgd(ps, 0.9, 0.0001, 35.0)
      },
      scheduler=@optimizer.StepLR::new(0.01, 5, 0.5),
    ),
    train_loader,
    test_loader,
  )
}

///|
fn main raise {
  @log_console.initialize()
  let m = @argparse.Command(
    "mnist",
    options=[
      @argparse.OptionArg("epochs", short='e', default_values=["10"]),
      @argparse.OptionArg("batch-size", short='b', default_values=["64"]),
    ],
    flags=[@argparse.FlagArg("gpu")],
  ).parse()
  let epochs = @string.parse_int(m.values["epochs"][0])
  let batch_size = @string.parse_int(m.values["batch-size"][0])
  if m.flags.contains("gpu") && m.flags["gpu"] {
    let (ex, tr, te) = mnist_setup(@gt.GpuTensor::scalar(0.0))
    @gpu_train.run_gpu(ex, tr, epochs~, batch_size~, eval_loader=te)
  } else {
    let (ex, tr, te) = mnist_setup(@nparray.NpArray::scalar(0.0))
    @train.run_cpu(ex, tr, epochs~, batch_size~, eval_loader=te)
  }
}
