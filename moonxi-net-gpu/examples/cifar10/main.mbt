// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
fn main raise {
  @log_console.initialize()
  let cmd = @argparse.Command("cifar10", options=[
    @argparse.OptionArg(
      "epochs",
      short='e',
      default_values=["20"],
      about="Number of training epochs",
    ),
    @argparse.OptionArg(
      "optimizer",
      short='o',
      default_values=["momentum_sgd"],
      about="Optimizer: momentum_sgd, adam, rmsprop",
    ),
  ])
  let matches = cmd.parse()
  let num_epochs = @string.parse_int(matches.values["epochs"][0])
  let optim_name = matches.values["optimizer"][0]
  let (train_loader, test_loader) = @data.cifar10_loaders(
    "data/cifar-10-batches-bin",
  )
  let (optim, default_lr) = @optimizer.make_optimizer(optim_name)
  let experiment = @train.Experiment::new(
    model=@model.init_resnet18(10).as_module(),
    loss=@loss.cross_entropy_loss(10),
    optim~,
    scheduler=@optimizer.StepLR::new(default_lr, 20, 0.1),
  )
  @gpu_train.run_gpu(
    experiment,
    train_loader,
    epochs=num_epochs,
    eval_loader=test_loader,
  )
  @log.info("CIFAR-10 integration test passed")
}
