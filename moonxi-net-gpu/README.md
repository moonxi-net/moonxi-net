# moonxi-net-gpu

GPU backend for [moonxi-net](../moonxi-net): CUDA/cuDNN-accelerated tensor operations and training. Implements the `Tensor`, `BlasTensor`, `ImageTensor`, and `ImageBackwardOps` traits via `GpuTensor`, so models written against `moonxi-net` traits run on GPU without code changes.

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| [MoonBit (CUDA fork)](https://github.com/chnlkw/moon/tree/support_cuda) | latest | Compiler with CUDA compilation support |
| CUDA Toolkit | 12.8+ | nvcc, cuBLAS, CUDA Runtime |
| cuDNN | 9.21+ | Conv2d, BatchNorm, Pooling |
| NVIDIA GPU | Compute Capability 7.5+ | GPU training |

> **Note:** The standard MoonBit toolchain does **not** support CUDA compilation. You must use the [CUDA-patched fork](https://github.com/chnlkw/moon/tree/support_cuda).

## Installation

```bash
moon add chnlkw/moonxi-net-gpu
```

This will also pull in `moonxi-net` (the CPU backend) as a transitive dependency.

## Quick Example

The same linear regression as the CPU version, running on GPU:

```mbt
///|
fn main {
  // Initialize CUDA context
  @cuda.init()

  // Use GpuTensor instead of NpArray
  let params : Array[@gt.GpuTensor] = train_linear()
  // ...same generic train_linear() function, no code changes needed

  @cuda.shutdown()
}
```

```bash
moon run moonxi-net-gpu/examples/linear --target native --release
# Learned: w=3.00, b=1.00
```

## Usage

### Import in `moon.pkg`

```json
{
  "import": [
    "chnlkw/moonxi-net" @tensor,
    "chnlkw/moonxi-net-gpu/cuda" @cuda,
    "chnlkw/moonxi-net-gpu/tensor" @gt,
    "chnlkw/moonxi-net-gpu/train" @gpu_train,
    "chnlkw/moonxi-net-gpu/event_tensor" @et
  ]
}
```

### Core Packages

| Package | Alias | Description |
|---|---|---|
| `moonxi-net-gpu/cuda` | `@cuda` | CUDA runtime: `init()`, `shutdown()`, device management |
| `moonxi-net-gpu/tensor` | `@gt` | GPU tensor backend (`GpuTensor` implements all tensor traits) |
| `moonxi-net-gpu/train` | `@gpu_train` | GPU training loop, `run_gpu` |
| `moonxi-net-gpu/event_tensor` | `@et` | Event-based tensor with profiling/timing support |

### GPU Training Workflow

```mbt
///|
fn main {
  // Initialize CUDA
  @cuda.init()

  // Create tensors using GpuTensor (GPU backend)
  let w = @grad.no_grad(@gt.GpuTensor::zeros([10, 10]))
  let b = @grad.no_grad(@gt.GpuTensor::zeros([10]))

  // Same autograd API as CPU
  @grad.clear_tape()
  let loss = /* ... your forward pass ... */
  loss.backward()

  // Access gradients
  match w.grad {
    Some(Some(g)) => w.value = w.value.sub(g.scale(0.01))
    _ => ()
  }

  @cuda.shutdown()
}
```

### MNIST Training

```bash
bash scripts/download_mnist.sh
moon run moonxi-net-gpu/examples/mnist --target native --release                    # CPU backend
moon run moonxi-net-gpu/examples/mnist --target native --release -- --gpu           # GPU backend
moon run moonxi-net-gpu/examples/mnist --target native --release -- -e 20 -b 128    # Custom config
```

### CIFAR-10 ResNet-18 Training

```bash
bash scripts/download_cifar10.sh
moon run moonxi-net-gpu/examples/cifar10 --target native --release                   # 20 epochs, momentum SGD
moon run moonxi-net-gpu/examples/cifar10 --target native --release -- -e 10          # Custom epochs
moon run moonxi-net-gpu/examples/cifar10 --target native --release -- -o adam        # Use Adam optimizer
```

## Build & Test

```bash
# Requires CUDA-patched MoonBit + CUDA toolkit
moon test --target native
moon build --target native
```

## License

[MIT](../LICENSE)
