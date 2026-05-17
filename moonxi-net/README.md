# moonxi-net

A deep learning training framework built with [MoonBit](https://www.moonbitlang.com/), featuring tape-based autograd and a PyTorch-like API on the **CPU backend** (`NpArray`). Write your model once using the `Tensor` trait, and it works with both CPU and GPU backends via tagless-final style.

## Installation

```bash
moon add chnlkw/moonxi-net
```

> **Note:** This package works with the standard MoonBit toolchain — no CUDA or GPU required.

## Quick Example

A minimal linear regression: learn `y = 3x + 1` from data.

```mbt
///|
fn[T : @tensor.Tensor + @tensor.BlasTensor] train_linear() -> Array[T] {
  let w = @grad.no_grad(T::zeros([1, 1]))
  let b = @grad.no_grad(T::zeros([1, 1]))
  let x = @grad.no_grad(
    T::from_host(FixedArray::makei(5, i => Float::from_int(i + 1)), [5, 1]),
  )
  let y = @grad.no_grad(
    T::from_host(FixedArray::makei(5, i => Float::from_int((i + 1) * 3 + 1)), [
      5, 1,
    ]),
  )
  let params : Array[@grad.Grad[T]] = [w, b]
  for _epoch in 0..<=500 {
    @grad.clear_tape()
    for p in params { p.grad = Some(None) }
    let loss = x.matmul(w).add(b).sub(y).square().mean()
    loss.backward()
    for p in params {
      match p.grad {
        Some(Some(g)) => p.value = p.value.sub(g.scale(0.001))
        _ => ()
      }
    }
  }
  params.map(g => g.value)
}
```

```bash
moon run moonxi-net/examples/linear --target native
# Learned: w=3.00, b=1.00
```

## Usage

### Import in `moon.pkg`

```json
{
  "import": [
    "chnlkw/moonxi-net" @tensor,
    "chnlkw/moonxi-net/nparray" @nparray,
    "chnlkw/moonxi-net/grad" @grad,
    "chnlkw/moonxi-net/model" @model,
    "chnlkw/moonxi-net/optimizer" @optimizer,
    "chnlkw/moonxi-net/loss" @loss,
    "chnlkw/moonxi-net/train" @train,
    "chnlkw/moonxi-net/dataloader" @dl
  ]
}
```

### Core Packages

| Package | Alias | Description |
|---|---|---|
| `moonxi-net` | `@tensor` | Core traits: `Tensor`, `BlasTensor`, `ImageTensor`, `ImageBackwardOps` |
| `moonxi-net/nparray` | `@nparray` | CPU tensor backend (`NpArray` implements all tensor traits) |
| `moonxi-net/grad` | `@grad` | Tape-based autograd engine (`Grad[T]`, `backward()`, `clear_tape()`) |
| `moonxi-net/model` | `@model` | Neural network layers: `Linear[T]`, `Conv2d[T]`, `ResNet18[T]`, `MLP[T]` |
| `moonxi-net/optimizer` | `@optimizer` | Optimizers: Momentum SGD, Adam, RMSprop (with gradient clipping) |
| `moonxi-net/loss` | `@loss` | Loss functions: cross-entropy, MSE |
| `moonxi-net/train` | `@train` | Training loop utilities, `Experiment` config, `run_cpu` |
| `moonxi-net/dataloader` | `@dl` | DataLoader with Fisher-Yates shuffle and mini-batch iteration |
| `moonxi-net/datasets/mnist` | — | MNIST dataset loader |
| `moonxi-net/datasets/cifar10` | — | CIFAR-10 dataset loader |

### Basic Training Workflow

```mbt
///|
fn main {
  // Create tensors using NpArray (CPU backend)
  let w = @grad.no_grad(@nparray.NpArray::zeros([10, 10]))
  let b = @grad.no_grad(@nparray.NpArray::zeros([10]))

  // Build computation graph, compute loss, backprop
  @grad.clear_tape()
  let loss = /* ... your forward pass ... */
  loss.backward()

  // Access gradients
  match w.grad {
    Some(Some(g)) => w.value = w.value.sub(g.scale(0.01))
    _ => ()
  }
}
```

## Build & Test

```bash
# From this directory
moon test --target native

# Build
moon build --target native
```

## License

[MIT](../LICENSE)
