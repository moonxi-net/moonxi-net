"""
Profile per-phase timing for PyTorch CIFAR-10 ResNet-18 (eager and torch.compile).

Usage:
  uv run profile_cifar10.py              # eager + compile
  uv run profile_cifar10.py --eager      # eager only
  uv run profile_cifar10.py --compile    # compile only
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cifar10_common import create_model_and_optimizer, load_cifar10_bin


def profile_phases(model, loader, optimizer, device, n_batches=10):
    model.train()
    phase_times = {}
    batch_count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        def measure(name, fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            phase_times.setdefault(name, 0.0)
            phase_times[name] += start.elapsed_time(end)
            return result

        logits = measure("forward", lambda: model(images))
        loss = measure("loss", lambda: F.cross_entropy(logits, labels))
        measure("zero_grad", lambda: optimizer.zero_grad())
        measure("backward", lambda: loss.backward())
        measure("grad_clip", lambda: torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=35.0
        ))
        measure("optimizer", lambda: optimizer.step())
        measure("loss_item", lambda: loss.item())

        batch_count += 1
        if batch_count >= n_batches:
            break

    print(f"\n  Per-phase timing ({n_batches} batches):")
    total = sum(phase_times.values())
    for name, ms in sorted(phase_times.items(), key=lambda x: -x[1]):
        pct = ms / total * 100
        avg = ms / n_batches
        print(f"    {name:12s}: {avg:6.2f} ms/batch ({pct:5.1f}%)")
    print(f"    {'TOTAL':12s}: {total / n_batches:6.2f} ms/batch")
    return phase_times


def run_profile(label, compile_mode, train_imgs, train_lbls, device, n_batches=10):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    model, optimizer = create_model_and_optimizer(device)

    if compile_mode:
        print(f"  Compiling with torch.compile(mode='{compile_mode}')...")
        model = torch.compile(model, mode=compile_mode)
        dummy = torch.randn(2, 3, 32, 32, device=device)
        _ = model(dummy)
        torch.cuda.synchronize()

    loader = DataLoader(
        TensorDataset(train_imgs, train_lbls),
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    print("  Warmup...")
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        optimizer.step()
        if i >= 10:
            break
    torch.cuda.synchronize()

    print(f"  Profiling {n_batches} batches...")
    profile_phases(model, loader, optimizer, device, n_batches=n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    run_eager = args.eager or (not args.compile)
    run_compile = args.compile or (not args.eager)

    device = torch.device("cuda")
    print(f"Device: {device}, PyTorch {torch.__version__}")

    train_imgs, train_lbls, _, _ = load_cifar10_bin()
    print(f"CIFAR-10: {len(train_imgs)} images")

    if run_eager:
        run_profile("PyTorch Eager", None, train_imgs, train_lbls, device)

    if run_compile:
        run_profile("PyTorch torch.compile", "default", train_imgs, train_lbls, device)


if __name__ == "__main__":
    main()
