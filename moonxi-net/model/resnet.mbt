// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

// =============================================================================
// Initialization
// =============================================================================

///|
let _init_rng : @random.Rand = @random.Rand::chacha8()

///|
/// Box-Muller transform for generating normally-distributed random floats.
fn box_muller() -> Float {
  let u1 = Float::from_double(_init_rng.double())
  let u2 = Float::from_double(_init_rng.double())
  Float::sqrt(
    -Float::from_double(2.0) * Float::from_double(@math.ln(u1.to_double())),
  ) *
  Float::from_double(@math.cos(2.0 * @math.PI * u2.to_double()))
}

///|
/// Kaiming He initialization for conv weights (fan_in).
fn[T : @tensor.Tensor] kaiming_init(shape : Array[Int]) -> T {
  let fan_in = shape[1] * shape[2] * shape[3]
  let std = Float::sqrt(Float::from_double(2.0) / Float::from_int(fan_in))
  let numel = shape[0] * shape[1] * shape[2] * shape[3]
  let data = FixedArray::makei(numel, fn(_i) { box_muller() * std })
  T::from_host(data, shape)
}

///|
/// Initialize ResNet-18 using Kaiming He initialization (generic).
pub fn[T : @tensor.Tensor] init_resnet18(num_classes : Int) -> ResNet18[T] {
  // Helper: create a conv with Kaiming He init, bias=zeros
  fn make_conv(out_ch : Int, in_ch : Int, kh : Int, kw : Int) -> Conv2d[T] {
    let shape = [out_ch, in_ch, kh, kw]
    Conv2d::{ weight: kaiming_init(shape), bias: T::zeros([out_ch]) }
  }

  // Helper: create BN params (gamma=1, beta=0, running_mean=0, running_var=1)
  fn make_bn(channels : Int) -> BatchNorm2d[T] {
    let ones = FixedArray::make(channels, Float::from_double(1.0))
    let zeros = FixedArray::make(channels, Float::from_double(0.0))
    BatchNorm2d::{
      gamma: T::from_host(ones, [channels]),
      beta: T::from_host(zeros, [channels]),
      running_mean: T::from_host(zeros.copy(), [channels]),
      running_var: T::from_host(ones.copy(), [channels]),
    }
  }

  // Helper: create a BasicBlock
  fn make_block(in_c : Int, out_c : Int, stride : Int) -> BasicBlock[T] {
    let has_shortcut = in_c != out_c || stride != 1
    BasicBlock::{
      conv1: make_conv(out_c, in_c, 3, 3),
      bn1: make_bn(out_c),
      conv2: make_conv(out_c, out_c, 3, 3),
      bn2: make_bn(out_c),
      shortcut: if has_shortcut {
        Some(make_conv(out_c, in_c, 1, 1))
      } else {
        None
      },
      shortcut_bn: if has_shortcut {
        Some(make_bn(out_c))
      } else {
        None
      },
    }
  }

  // Helper: create a layer with 2 BasicBlocks
  fn make_layer(in_c : Int, out_c : Int, first_stride : Int) -> ResNetLayer[T] {
    let blocks : Array[BasicBlock[T]] = []
    blocks.push(make_block(in_c, out_c, first_stride))
    blocks.push(make_block(out_c, out_c, 1))
    ResNetLayer::{ blocks, }
  }

  // Stem: Conv7x7(3, 64, stride=2, pad=3) + BN
  let stem_conv = make_conv(64, 3, 7, 7)
  let stem_bn = make_bn(64)

  // 4 layers, each with 2 BasicBlocks (ResNet-18)
  let layer1 = make_layer(64, 64, 1)
  let layer2 = make_layer(64, 128, 2)
  let layer3 = make_layer(128, 256, 2)
  let layer4 = make_layer(256, 512, 2)

  // Head: Xavier init
  let fan_in = 512
  let fan_out = num_classes
  let std_fc = Float::sqrt(
    Float::from_double(2.0) / Float::from_int(fan_in + fan_out),
  )
  let n = num_classes * 512
  let head_data = FixedArray::makei(n, fn(_i) { box_muller() * std_fc })
  let head_weight = T::from_host(head_data, [num_classes, 512])
  let head_bias = T::zeros([num_classes])

  ResNet18::{
    stem_conv,
    stem_bn,
    layer1,
    layer2,
    layer3,
    layer4,
    head_weight,
    head_bias,
  }
}
