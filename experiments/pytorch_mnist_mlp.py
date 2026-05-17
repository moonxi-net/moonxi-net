"""
PyTorch reference implementation matching the MoonBit MNIST MLP training.
Goal: verify that loss/accuracy converge to the same ballpark.

Architecture:  MLP 784→128→ReLU→64→ReLU→10
Optimizer:     SGD + Momentum(0.9) + WeightDecay(1e-4), grad_clip=35
LR Scheduler:  StepLR(initial=0.01, step_size=5, gamma=0.5)
Loss:          CrossEntropyLoss (integer class labels)
Init:          Xavier/Glorot uniform for weights, zeros for biases
Data:          MNIST, normalized to [0,1], batch_size=64

Usage:
  python pytorch_mnist_mlp.py -e 2 -b 64
  python pytorch_mnist_mlp.py -e 2 -b 64 --gpu
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import time


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-l", "--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=35.0)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and not torch.cuda.is_available():
        print("WARNING: --gpu requested but CUDA not available, falling back to CPU")

    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    backend = "CUDA GPU" if device.type == "cuda" else "CPU"
    print(f"PyTorch MNIST MLP Reference ({backend})")
    print(f"Device: {device}")
    print(f"Epochs={args.epochs}, Batch={args.batch_size}, "
          f"LR={args.lr}, Momentum={args.momentum}, WD={args.weight_decay}")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"Model: MLP (784->128->64->10)")
    print()

    t0_total = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0_epoch = time.perf_counter()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        t_epoch = time.perf_counter() - t0_epoch

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        acc = 100.0 * correct / total
        train_acc_correct = 0
        train_acc_total = 0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                train_acc_correct += (preds == batch_y).sum().item()
                train_acc_total += batch_y.size(0)
        train_acc = 100.0 * train_acc_correct / train_acc_total

        current_lr = scheduler.get_last_lr()[0]
        print(f"{epoch},{current_lr:.16f},{avg_loss},{train_acc},{acc},{t_epoch}")
        scheduler.step()

    t_total = time.perf_counter() - t0_total
    print(f"\nTotal time: {t_total:.3f}s ({backend})")


if __name__ == "__main__":
    main()
