# moonxi-net

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/moonxi-net/moonxi-net)
[![CI](https://github.com/moonxi-net/moonxi-net/actions/workflows/stable-check.yml/badge.svg)](https://github.com/moonxi-net/moonxi-net/actions/workflows/stable-check.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Made with MoonBit](https://img.shields.io/badge/made%20with-MoonBit-blueviolet)](https://www.moonbitlang.com/)

A deep learning training framework built with [MoonBit](https://www.moonbitlang.com/) + CUDA/cuDNN, featuring a tape-based autograd engine and PyTorch-like API. Uses **tagless final** style to abstract over CPU (`NpArray`) and GPU (`GpuTensor`) backends — write your model once, run on either. Supports CUDA/cuBLAS/cuDNN acceleration.

Trained a ResNet-18 on CIFAR-10 to **69.6% test accuracy** (20 epochs, momentum SGD, no data augmentation) — comparable to PyTorch's 70.6%, but **32% faster per epoch**. Run `bash scripts/run_cifar10_training.sh` for a side-by-side comparison; results go to `reports/`.

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| [MoonBit (CUDA fork)](https://github.com/chnlkw/moon/tree/support_cuda) | latest | Compiler & toolchain (CUDA-patched fork) |
| CUDA Toolkit | 12.8+ | nvcc, cuBLAS, CUDA Runtime |
| cuDNN | 9.21+ | Conv2d, BatchNorm, Pooling |
| C compiler (GCC/Clang/MSVC) | - | Native stub compilation |
| NVIDIA GPU | Compute Capability 7.5+ | GPU training |

> **Note:** CPU-only packages (`nparray`, `grad`, `model`, `optimizer`, `train`, `loss`, `dataloader`, `utils`, `datasets/*`) work without CUDA/cuDNN. GPU packages (`gpu/*`) require CUDA toolkit and an NVIDIA GPU.

## Quick Example

A minimal linear regression: learn `y = 3x + 1` from data ([source](moonxi-net/examples/linear/main.mbt)).

```mbt
///|
fn[T : @tensor.Tensor + @tensor.BlasTensor] train_linear() -> Array[T] {
  let w = @grad.no_grad(T::zeros([1, 1]))
  let b = @grad.no_grad(T::zeros([1, 1]))
  let x = @grad.no_grad(T::from_host(/* ... */, [5, 1]))
  let y = @grad.no_grad(T::from_host(/* ... */, [5, 1]))
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
# CPU
moon run moonxi-net/examples/linear --target native
# GPU
moon run moonxi-net-gpu/examples/linear --target native --release
# Learned: w=3.00, b=1.00
```

## Standalone Tutorial

A **single-file, zero-dependency** implementation that builds a trainable linear model from scratch — no framework, no imports. Implements `Tensor`, `NpArray`, `Linear[T]`, and tape-based autograd (`Grad[T]`) in ~350 lines to learn `y = 3x + 1`.

([source](moonxi-net/examples/tutorial/main.mbt))

```bash
moon run moonxi-net/examples/tutorial --target native
```

## Features

- **Tape-based autograd** with CPU (NpArray) and GPU (GpuTensor) backends
- **ResNet-18** and **MLP** model definitions
- **Optimizers**: Momentum SGD, Adam, RMSprop (with gradient clipping)
- **Loss functions**: Cross-entropy, MSE
- **DataLoader** with Fisher-Yates shuffle and mini-batch iteration
- **PyTorch-like API**: `Experiment` config → `run_cpu` / `run_gpu`
- **GPU acceleration**: CUDA/cuBLAS/cuDNN via dynamic FFI loading

## Quick Start

### Install MoonBit

This project requires a [CUDA-patched fork of MoonBit](https://github.com/chnlkw/moon/tree/support_cuda). Build and install it with:

```bash
git clone https://github.com/chnlkw/moon.git -b support_cuda
cd moon
cargo build --release
cp target/release/moon ~/.bin/moon/
```

> The standard MoonBit toolchain does not support CUDA compilation. You must use the fork above.

### Build & Test

> **Note:** Running `moon test` or `moon build` for the full workspace requires the CUDA-patched MoonBit, because benchmark and integration test packages depend on GPU packages. If you only have the standard MoonBit, you can test individual CPU packages (see below).

```bash
# From workspace root
moon update

# Format code & update interfaces
moon info && moon fmt

# Run all tests (requires CUDA-patched MoonBit + CUDA toolkit)
moon test --target native

# Build all
moon build --target native
```

**Testing CPU packages with standard MoonBit** (no GPU required):

```bash
# From the moonxi-net/ directory
cd moonxi-net
moon test --target native
```

### Run Examples

**MNIST MLP training** (CPU or GPU):

```bash
bash scripts/download_mnist.sh                              # Download dataset

moon run moonxi-net-gpu/examples/mnist --target native --release                    # CPU (NpArray) backend — same binary, different backend
moon run moonxi-net-gpu/examples/mnist --target native --release -- --gpu           # GPU (GpuTensor) backend — just add --gpu
moon run moonxi-net-gpu/examples/mnist --target native --release -- -e 20 -b 128    # Custom epochs & batch size
```

**CIFAR-10 ResNet-18 training** (GPU only):

```bash
bash scripts/download_cifar10.sh                            # Download dataset

moon run moonxi-net-gpu/examples/cifar10 --target native --release                   # 20 epochs, momentum SGD
moon run moonxi-net-gpu/examples/cifar10 --target native --release -- -e 10          # Custom epochs
moon run moonxi-net-gpu/examples/cifar10 --target native --release -- -o adam        # Use Adam optimizer
```

### Training Comparison Scripts

Run MoonBit vs PyTorch side-by-side with logging, CSV output, SVG plots, and summary reports:

```bash
# MNIST: MoonBit GPU + PyTorch (10 epochs default)
bash scripts/run_mnist_training.sh
bash scripts/run_mnist_training.sh --epochs 20 --batch-size 128

# CIFAR-10: MoonBit GPU + PyTorch (20 epochs default)
bash scripts/run_cifar10_training.sh
bash scripts/run_cifar10_training.sh --epochs 10
bash scripts/run_cifar10_training.sh --no-build    # Skip build, just run
```

Results are written to `reports/`:
- `*.log` — Full training output with timestamps
- `*.csv` — Per-epoch metrics (loss, accuracy, time)
- `*.svg` — Comparison plots (loss, accuracy, epoch time)
- `*_summary.md` — Markdown summary with comparison tables

## License

[MIT](LICENSE)
