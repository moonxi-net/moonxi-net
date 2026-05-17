# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-13

### Added

- Tape-based autograd engine with CPU (NpArray) and GPU (GpuTensor) backends
- ResNet-18 model with CIFAR-10 training (62.9% test accuracy, 10 epochs)
- MLP model for MNIST classification (CPU and GPU)
- GPU backend: CUDA/cuBLAS/cuDNN FFI with dynamic loading
- Optimizers: SGD (with gradient clipping), Momentum SGD, Adam, RMSprop
- StepLR learning rate scheduler
- Loss functions: cross-entropy (one-hot and label modes), MSE
- Generic DataLoader with Fisher-Yates shuffle and batch iteration
- PyTorch-like API (`Experiment`, `run_cpu`, `run_gpu`)
- Benchmark suite for GPU kernels and optimizer comparison
- MNIST CPU training support with data loading
- CIFAR-10 binary dataset loader

[0.1.0]: https://github.com/moonxi-net/moonxi-net/releases/tag/v0.1.0
