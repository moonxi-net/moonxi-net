"""
ResNet-18 CIFAR-10 Training with PyTorch.

Replicates the exact algorithm from the MoonBit codebase:
  - ResNet-18: stem conv7x7(stride=2,pad=3) → BN → ReLU → MaxPool(3,stride=2,pad=1)
    → 4 layers of 2 BasicBlocks → AdaptiveAvgPool → Linear
  - SGD with momentum=0.9, weight_decay=0.0001, max_grad_norm=35.0
  - StepLR: step=20, gamma=0.1, initial lr=0.01
  - Softmax cross-entropy loss with one-hot labels
  - BN momentum=0.1, eps=1e-5
  - Kaiming He init for conv, Xavier init for FC head
  - Pixel normalization to [0,1], no additional augmentation
  - batch_size=128

Usage:
  uv run train_cifar10.py
"""

import argparse
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cifar10_common import ResNet18


# ---------------------------------------------------------------------------
# CIFAR-10 binary dataset loader (reads the same data/ files as MoonBit code)
# ---------------------------------------------------------------------------


class CIFAR10BinaryDataset(Dataset):
    """Load CIFAR-10 from the binary format used by the MoonBit codebase."""

    def __init__(self, data_dir: str, train: bool = True):
        self.images = []
        self.labels = []
        if train:
            for batch_id in range(1, 6):
                path = os.path.join(data_dir, f"data_batch_{batch_id}.bin")
                imgs, lbls = self._load_batch(path)
                self.images.append(imgs)
                self.labels.append(lbls)
        else:
            path = os.path.join(data_dir, "test_batch.bin")
            imgs, lbls = self._load_batch(path)
            self.images.append(imgs)
            self.labels.append(lbls)

        self.images = torch.cat(self.images, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    @staticmethod
    def _load_batch(path: str):
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="bytes")

        raw_images = data[b"data"]
        raw_labels = data[b"labels"]

        images = torch.tensor(raw_images, dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
        labels = torch.tensor(raw_labels, dtype=torch.long)
        return images, labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CIFAR10RawBinaryDataset(Dataset):
    """Load CIFAR-10 from raw binary files (data_batch_*.bin / test_batch.bin).

    This matches the MoonBit read_file approach exactly: each record is
    1 byte label + 3072 bytes pixels (3072 = 3*32*32).
    """

    def __init__(self, data_dir: str, train: bool = True):
        self.images = []
        self.labels = []

        if train:
            files = [f"data_batch_{i}.bin" for i in range(1, 6)]
        else:
            files = ["test_batch.bin"]

        for fname in files:
            path = os.path.join(data_dir, fname)
            imgs, lbls = self._load_raw_batch(path)
            self.images.append(imgs)
            self.labels.append(lbls)

        self.images = torch.cat(self.images, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    @staticmethod
    def _load_raw_batch(path: str):
        with open(path, "rb") as f:
            raw = f.read()

        n = len(raw) // 3073
        images = torch.zeros(n, 3, 32, 32, dtype=torch.float32)
        labels = torch.zeros(n, dtype=torch.long)

        for i in range(n):
            offset = i * 3073
            labels[i] = raw[offset]
            pixels = torch.ByteTensor(
                list(raw[offset + 1 : offset + 3073])
            ).reshape(3, 32, 32)
            images[i] = pixels.float() / 255.0

        return images, labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_optimizer(name, params):
    """Return (optimizer, lr, scheduler) matching MoonBit's make_optimizer.

    All optimizers use StepLR(step=20, gamma=0.1) and max_grad_norm=35.0.
    """
    lr_decay_step = 20
    lr_decay_gamma = 0.1

    if name == "sgd":
        lr = 0.01
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=0.0001,
        )
    elif name == "adam":
        lr = 0.001
        optimizer = torch.optim.Adam(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001,
        )
    elif name == "rmsprop":
        lr = 0.001
        optimizer = torch.optim.RMSprop(
            params, lr=lr, alpha=0.99, eps=1e-8, weight_decay=0.0001, momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name!r}")

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma,
    )
    return optimizer, lr, scheduler


def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            torch.cuda.nvtx.range_push("eval_forward")
            logits = model(images)
            torch.cuda.nvtx.range_pop()
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += images.size(0)

    return total_correct / total_samples


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batch_ms = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        t_batch = time.perf_counter()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        torch.cuda.nvtx.range_push(f"fwd_e{epoch}_b{batch_idx}")
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"bwd_e{epoch}_b{batch_idx}")
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"optim_e{epoch}_b{batch_idx}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        optimizer.step()
        torch.cuda.nvtx.range_pop()

        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += images.size(0)
        total_batch_ms += (time.perf_counter() - t_batch) * 1000.0

        if batch_idx % 100 == 0:
            print(f"  e{epoch} b{batch_idx}: loss={loss.item():.4f}")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    avg_batch_ms = total_batch_ms / (batch_idx + 1)
    return avg_loss, accuracy, avg_batch_ms


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 Training")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-o", "--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam", "rmsprop"], help="Optimizer (default: sgd)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "cifar-10-batches-bin")
    num_epochs = args.epochs
    optimizer_name = args.optimizer
    batch_size = 128
    lr_decay_step = 20
    lr_decay_gamma = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        train_dataset = CIFAR10RawBinaryDataset(data_dir, train=True)
        test_dataset = CIFAR10RawBinaryDataset(data_dir, train=False)
        print("Loaded CIFAR-10 from raw binary files")
    except Exception:
        try:
            train_dataset = CIFAR10BinaryDataset(data_dir, train=True)
            test_dataset = CIFAR10BinaryDataset(data_dir, train=False)
            print("Loaded CIFAR-10 from pickle binary files")
        except Exception:
            import torchvision
            import torchvision.transforms as transforms
            print("Downloading CIFAR-10 via torchvision...")
            transform = transforms.ToTensor()
            train_dataset = torchvision.datasets.CIFAR10(
                root="data", train=True, download=True, transform=transform,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root="data", train=False, download=True, transform=transform,
            )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
    )

    num_train = len(train_dataset)
    num_test = len(test_dataset)
    print(f"CIFAR-10: {num_train} train, {num_test} test images loaded")

    model = ResNet18(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18 | params={total_params:,}")

    optimizer, lr, scheduler = make_optimizer(optimizer_name, model.parameters())

    print(f"Optimizer: {optimizer_name} | batch={batch_size} lr={lr} "
          f"step={lr_decay_step} gamma={lr_decay_gamma}")
    print(f"  {optimizer}")

    print(f"\n=== CIFAR-10 ResNet-18 Training ({num_epochs} epochs) ===\n")

    for epoch in range(num_epochs):
        torch.cuda.nvtx.range_push(f"epoch_{epoch}")
        t0 = time.perf_counter()
        avg_loss, train_acc, avg_batch_ms = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        epoch_train_s = time.perf_counter() - t0

        torch.cuda.nvtx.range_push(f"train_eval_{epoch}")
        t_eval = time.perf_counter()
        train_acc_eval = evaluate(model, train_loader, device)
        train_eval_s = time.perf_counter() - t_eval
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"test_eval_{epoch}")
        t_test = time.perf_counter()
        test_acc = evaluate(model, test_loader, device)
        test_eval_s = time.perf_counter() - t_test
        torch.cuda.nvtx.range_pop()

        total_s = time.perf_counter() - t0
        scheduler.step()
        torch.cuda.nvtx.range_pop()

        print(f"Epoch {epoch + 1}/{num_epochs} loss={avg_loss:.4f} "
              f"train_acc={train_acc * 100:.1f}% test_acc={test_acc * 100:.1f}% "
              f"| batch={avg_batch_ms:.1f}ms train={epoch_train_s:.1f}s "
              f"train_eval={train_eval_s:.1f}s test_eval={test_eval_s:.1f}s "
              f"total={total_s:.1f}s")

    print("\nDone")


if __name__ == "__main__":
    main()
