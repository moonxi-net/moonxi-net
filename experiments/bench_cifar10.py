"""
Benchmark PyTorch CIFAR-10 ResNet-18: eager vs torch.compile.

Usage:
  uv run bench_cifar10.py              # eager + compile
  uv run bench_cifar10.py --eager      # eager only
  uv run bench_cifar10.py --compile    # compile only
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from cifar10_common import (
    create_model_and_optimizer,
    evaluate,
    load_cifar10_bin,
    train_step,
)


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    epoch_start = time.perf_counter()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
        num_batches = len(loader)
        if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == num_batches:
            print(
                f"epoch={epoch + 1}/{num_epochs} batch={batch_idx + 1}/{num_batches} loss={loss.item():.2f}"
            )
    epoch_time = time.perf_counter() - epoch_start
    return total_loss / total_samples, total_correct / total_samples, epoch_time


def evaluate_timed(model, images, labels, batch_size=256):
    model.eval()
    t0 = time.perf_counter()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            logits = model(images[i : i + batch_size])
            correct += (logits.argmax(1) == labels[i : i + batch_size]).sum().item()
            total += len(labels[i : i + batch_size])
    elapsed = (time.perf_counter() - t0) * 1000
    model.train()
    return correct / total, elapsed


def profile_phases(model, loader, optimizer, device, n_batches=10):
    model.train()
    phase_times = {}
    batch_count = 0
    for images_cpu, labels_cpu in loader:
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

        images = measure("data_h2d", lambda: images_cpu.to(device))
        labels = labels_cpu.to(device)

        logits = measure("forward", lambda: model(images))
        loss = measure("loss", lambda: torch.nn.functional.cross_entropy(logits, labels))
        measure("zero_grad", lambda: optimizer.zero_grad())
        measure("backward", lambda: loss.backward())
        measure("grad_clip", lambda: torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=35.0
        ))
        measure("optimizer", lambda: optimizer.step())

        batch_count += 1
        if batch_count >= n_batches:
            break

    print(f"\n  Per-phase timing ({n_batches} batches):")
    total = sum(phase_times.values())
    for name, ms in sorted(phase_times.items(), key=lambda x: -x[1]):
        print(f"    {name:12s}: {ms / n_batches:6.2f} ms/batch ({ms / total * 100:5.1f}%)")
    print(f"    {'TOTAL':12s}: {total / n_batches:6.2f} ms/batch")
    return phase_times


def run_benchmark(label, backend_or_mode, train_imgs, train_lbls, test_imgs, test_lbls, device, num_epochs):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    model, optimizer = create_model_and_optimizer(device)

    if backend_or_mode:
        backend, mode = backend_or_mode
        print(f"  torch.compile(backend='{backend}', mode='{mode}')...")
        model = torch.compile(model, backend=backend, mode=mode)
        try:
            dummy = torch.randn(2, 3, 32, 32, device=device)
            _ = model(dummy)
            torch.cuda.synchronize()
            print("  Compile succeeded")
        except Exception as e:
            print(f"  Compile failed: {type(e).__name__}: {e}")
            print("  Falling back to eager")
            model, optimizer = create_model_and_optimizer(device)

    loader = DataLoader(
        TensorDataset(train_imgs, train_lbls),
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    train_imgs_dev = train_imgs.to(device)
    train_lbls_dev = train_lbls.to(device)
    test_imgs_dev = test_imgs.to(device)
    test_lbls_dev = test_lbls.to(device)

    for epoch in range(num_epochs):
        avg_loss, train_acc, epoch_time = train_epoch(
            model, loader, optimizer, device, epoch
        )
        test_acc, test_eval_ms = evaluate_timed(model, test_imgs_dev, test_lbls_dev)
        train_acc_eval, train_eval_ms = evaluate_timed(model, train_imgs_dev, train_lbls_dev)
        avg_batch_ms = epoch_time / len(loader) * 1000
        train_eval_s = train_eval_ms / 1000
        test_eval_s = test_eval_ms / 1000
        print(
            f"Epoch {epoch + 1}/{num_epochs} train loss={avg_loss:.2f} "
            f"acc={train_acc_eval * 100:.1f}% time={epoch_time:.1f}s+{train_eval_s:.1f}s batch={avg_batch_ms:.1f}ms"
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs} test  acc={test_acc * 100:.1f}% time={test_eval_s:.2f}s"
        )

    profile_phases(model, loader, optimizer, device, n_batches=10)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true", help="Run eager mode")
    parser.add_argument("--compile", action="store_true", help="Run torch.compile mode")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_eager = args.eager or (not args.compile)
    run_compile = args.compile or (not args.eager)

    device = torch.device("cuda")
    print(f"Device: {device}, PyTorch {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name()}")

    train_imgs, train_lbls, test_imgs, test_lbls = load_cifar10_bin()
    print(f"CIFAR-10: {len(train_imgs)} train, {len(test_imgs)} test")

    if run_eager:
        run_benchmark(
            "PyTorch Eager",
            backend_or_mode=None,
            train_imgs=train_imgs,
            train_lbls=train_lbls,
            test_imgs=test_imgs,
            test_lbls=test_lbls,
            device=device,
            num_epochs=args.epochs,
        )

    if run_compile:
        run_benchmark(
            "PyTorch torch.compile (inductor)",
            ("inductor", "default"),
            train_imgs=train_imgs,
            train_lbls=train_lbls,
            test_imgs=test_imgs,
            test_lbls=test_lbls,
            device=device,
            num_epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
