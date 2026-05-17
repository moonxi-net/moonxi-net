# OptimStep 重构：函数签名对比

---

## 当前架构

### trait OptimStep（现有）

```moonbit
trait OptimStep {
  // 单参数
  norm_buf_new() -> Self
  norm_buf_reset(Self) -> Unit
  norm_sq_accum(Self, buf: Self) -> Unit
  norm_buf_read(Self) -> Float
  sgd_momentum(Self, grad: Self, vel: Self, lr: Float, mom: Float, wd: Float, clip: Float) -> (Self, Self)

  // 批量（已有，但 step() 没用）
  batch_norm_sq_accum(tensors: Array[Self], buf: Self) -> Unit
  batch_sgd_momentum(params: Array[Self], grads: Array[Self], vels: Array[Self], lr: Float, mom: Float, wd: Float, clip: Float) -> Unit
}
```

### MomentumSGD[T]（现有）

```moonbit
struct MomentumSGD[T] {
  velocities: Array[T]
  norm_buf: T
  lr: Float; momentum: Float; weight_decay: Float; max_grad_norm: Float
  mut fast_step: (Array[Grad[T]], Array[T], Float, Float, Float, Float) -> Bool
}

// 构造
fn[T: Tensor + OptimStep] MomentumSGD::new(
  params: Array[Grad[T]],
  lr: Float, momentum: Float, weight_decay: Float, max_grad_norm: Float,
) -> MomentumSGD[T]

// 安装后门
fn[T: Tensor + OptimStep] MomentumSGD::set_fast_step(
  self: MomentumSGD[T],
  f: (Array[Grad[T]], Array[T], Float, Float, Float, Float) -> Bool,
) -> Unit

// 执行一步
fn[T: Tensor + OptimStep] MomentumSGD::step(
  self: MomentumSGD[T],
  params: Array[Grad[T]],
) -> Unit
```

### GPU 专属函数（gpu/cuda 包）

```moonbit
struct MultiTensorOptimState {
  d_param_ptrs: GpuBuffer    // GPU 端参数指针数组
  d_grad_ptrs: GpuBuffer     // GPU 端梯度指针数组
  d_vel_ptrs: GpuBuffer      // GPU 端速度指针数组
  d_sizes: GpuBuffer         // 每个参数的元素数
  d_block_offsets: GpuBuffer // CUDA block 偏移
  norm_buf: GpuBuffer        // norm 累加缓冲
  grad_ptrs: FixedArray[GpuBuffer]  // host 端梯度指针（每步复用）
  num_tensors: Int
  total_blocks: Int
}

fn multi_tensor_state_new(
  params: Array[GpuTensor],
  velocities: Array[GpuTensor],
) -> MultiTensorOptimState

fn multi_tensor_sgd_step(
  state: MultiTensorOptimState,
  grads: Array[GpuTensor],
  lr: Float, momentum: Float, weight_decay: Float, max_grad_norm: Float,
) -> Unit
```

### 调用链（当前）

```
main.mbt:
  ┌──────────────────────────────────────────────────────────────────┐
  │ let model = init_resnet18[GpuTensor](10)                         │
  │   → ResNet18[GpuTensor]                                          │
  │   → model.params() : Array[GpuTensor]                            │
  │                                                                  │
  │ let optim = MomentumSGD::new(model.params(), 0.01, ...)          │
  │   → MomentumSGD[GpuTensor]                                       │
  │                                                                  │
  │ // 手动提取参数值                                                  │
  │ let param_vals = model.params().map(fn(p: Grad[GpuTensor])       │
  │   → GpuTensor { p.value })                                        │
  │   : Array[GpuTensor]                                             │
  │                                                                  │
  │ // 手动复制速度                                                    │
  │ let vel_vals = optim.velocities.copy()                            │
  │   : Array[GpuTensor]                                             │
  │                                                                  │
  │ // 创建 GPU 状态                                                   │
  │ let mt_state = multi_tensor_state_new(param_vals, vel_vals)       │
  │   : MultiTensorOptimState                                        │
  │                                                                  │
  │ // 安装 30 行回调                                                  │
  │ optim.set_fast_step(fn(params, velocities, lr, mom, wd, mgn) {   │
  │   let grad_vals = /* 提取梯度或用 velocity 填充 */                  │
  │   multi_tensor_sgd_step(mt_state, grad_vals, lr, mom, wd, mgn)   │
  │   for i ... params[i].grad = None                                 │
  │   true                                                            │
  │ })                                                                │
  └──────────────────────────────────────────────────────────────────┘

trainer.step(input, target):
  └→ Trainer::step → loss.backward() → optim.step(params)
       └→ MomentumSGD::step:
            if fast_step(params, velocities, ...) { return }
                 ↓
            multi_tensor_sgd_step(mt_state, ...)
```

---

## 提议架构

### trait OptimStep（新增一个方法）

```moonbit
trait OptimStep {
  // ... 现有方法不变 ...

  // 新增：融合 norm + clip + sgd，backend 自行决定实现
  batch_sgd_momentum_autoclip(
    params: Array[Self],
    grads: Array[Self],
    velocities: Array[Self],
    lr: Float,
    momentum: Float,
    weight_decay: Float,
    max_grad_norm: Float,
  ) -> Unit
}
```

### GpuTensor 实现（gpu/cuda 包）

```moonbit
impl OptimStep for GpuTensor with batch_sgd_momentum_autoclip(
  params: Array[GpuTensor],
  grads: Array[GpuTensor],
  velocities: Array[GpuTensor],
  lr: Float, momentum: Float, weight_decay: Float, max_grad_norm: Float,
) -> Unit {
  let n = params.length()
  let p_ptrs = FixedArray::makei(n, fn(i) { params[i].data })
  let g_ptrs = FixedArray::makei(n, fn(i) { grads[i].data })
  let v_ptrs = FixedArray::makei(n, fn(i) { velocities[i].data })
  let sizes  = FixedArray::makei(n, fn(i) { params[i].numel() })
  let offsets = compute_block_offsets(sizes)
  let total_blocks = ...
  let norm_buf = alloc(4L)

  // 2 个 GPU kernel，零 D2H sync
  mt_norm_sq_gpu_only(g_ptrs, sizes, offsets, n, total_blocks, norm_buf)
  mt_sgd_momentum_autoclip(p_ptrs, g_ptrs, v_ptrs, sizes, offsets, n,
                           total_blocks, lr, momentum, weight_decay,
                           norm_buf, max_grad_norm)
}
```

### MomentumSGD::step() 重写（optimizer 包）

```moonbit
fn[T: Tensor + OptimStep] MomentumSGD::step(
  self: MomentumSGD[T],
  params: Array[Grad[T]],
) -> Unit {
  // 收集梯度
  let mut has_grad = false
  let param_vals : Array[T] = []
  let grad_vals : Array[T] = []
  for i in 0..<params.length() {
    param_vals.push(params[i].value)
    match params[i].grad {
      Some(Some(g)) => { has_grad = true; grad_vals.push(g) }
      Some(None) | None => grad_vals.push(self.velocities[i])
    }
  }
  if !has_grad { return }

  // 一行 trait 调用
  T::batch_sgd_momentum_autoclip(
    param_vals, grad_vals, self.velocities,
    self.lr, self.momentum, self.weight_decay, self.max_grad_norm,
  )

  // 清理梯度
  for i in 0..<params.length() {
    params[i].grad = None
  }
}
```

### 调用链（提议）

```
main.mbt:
  ┌──────────────────────────────────────────────────────────────────┐
  │ let model = init_resnet18[GpuTensor](10)                         │
  │ let optim = MomentumSGD::new(model.params(), 0.01, ...)          │
  │                                                                  │
  │ // 完了。没有额外设置。                                             │
  └──────────────────────────────────────────────────────────────────┘

trainer.step(input, target):
  └→ Trainer::step → loss.backward() → optim.step(params)
       └→ MomentumSGD::step:
            collect grads → T::batch_sgd_momentum_autoclip(...)
                                  ↓ trait dispatch
                            ┌─────┴──────┐
                            │ GpuTensor  │ → fused GPU kernels (零 sync)
                            │ CPU Tensor │ → 循环 sgd_momentum
                            └────────────┘
```

---

## 对比

```
当前:
  main.mbt:  optim.new() + 30 行 GPU 代码 + set_fast_step()
  step():    if fast_step(...) { return } else { per-param loop }
  后果:      fast_step 后门 + 调用方耦合 GPU

提议:
  main.mbt:  optim.new()   ← 就这一行
  step():    T::batch_sgd_momentum_autoclip(...)   ← trait 调用
  后果:      无 fast_step + 调用方零 GPU 代码

代价:
  GpuTensor impl 每步分配 3 个 FixedArray (~60 × 8 bytes each)
  + 3 次 H2D copy (~480 bytes each)
  ≈ 1.4 KB H2D / step，预计 < 0.1ms
```
