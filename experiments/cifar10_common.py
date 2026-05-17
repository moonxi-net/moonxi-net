"""
Shared ResNet-18 model and CIFAR-10 data loading for PyTorch experiments.

Matches the MoonBit codebase architecture exactly:
  - stem: Conv7x7(3,64,stride=2,pad=3) → BN → ReLU → MaxPool(3,stride=2,pad=1)
  - 4 layers of 2 BasicBlocks
  - AdaptiveAvgPool → Linear
  - Kaiming He init for conv, Xavier for FC head
  - BN momentum=0.1, eps=1e-5
"""

import os
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model ──


class BasicBlock(nn.Module):
    """ResNet BasicBlock: conv3x3 → BN → ReLU → conv3x3 → BN + shortcut → ReLU"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)

        self.has_shortcut = (in_channels != out_channels) or (stride != 1)
        if self.has_shortcut:
            self.shortcut_conv = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, bias=True
            )
            self.shortcut_bn = nn.BatchNorm2d(
                out_channels, momentum=0.1, eps=1e-5
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.has_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        else:
            shortcut = x

        out = F.relu(out + shortcut)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.stem_bn = nn.BatchNorm2d(64, momentum=0.1, eps=1e-5)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = (2.0 / fan_in) ** 0.5
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_in = m.in_features
                fan_out = m.out_features
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_conv(x)
        x = F.relu(self.stem_bn(x))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# ── Data loading ──


def load_cifar10_bin(data_dir: str = None):
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data", "cifar-10-batches-bin")
    """Load CIFAR-10 from raw binary files matching MoonBit's format.

    Returns (train_images, train_labels, test_images, test_labels)
    as float32 CPU tensors.
    """
    train_images, train_labels = [], []
    for i in range(1, 6):
        path = os.path.join(data_dir, f"data_batch_{i}.bin")
        with open(path, "rb") as f:
            raw = struct.unpack(f">{10000 * 3073}B", f.read(10000 * 3073))
        data = torch.tensor(raw, dtype=torch.float32).reshape(10000, 3073)
        train_images.append(data[:, 1:].reshape(10000, 3, 32, 32) / 255.0)
        train_labels.append(data[:, 0].long())

    path = os.path.join(data_dir, "test_batch.bin")
    with open(path, "rb") as f:
        raw = struct.unpack(f">{10000 * 3073}B", f.read(10000 * 3073))
    data = torch.tensor(raw, dtype=torch.float32).reshape(10000, 3073)
    test_images = data[:, 1:].reshape(10000, 3, 32, 32) / 255.0
    test_labels = data[:, 0].long()

    return (
        torch.cat(train_images),
        torch.cat(train_labels),
        test_images,
        test_labels,
    )


def create_model_and_optimizer(device, lr=0.01, momentum=0.9, weight_decay=0.0001):
    """Create ResNet-18 + SGD optimizer on device."""
    model = ResNet18(num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    return model, optimizer


# ── Training loop ──


def train_step(model, images, labels, optimizer, clip_norm=35.0):
    """Single training step. Returns loss scalar (Python float)."""
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    optimizer.step()
    return loss


def evaluate(model, images, labels, batch_size=256):
    """Evaluate accuracy on given tensors (already on device)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            logits = model(images[i : i + batch_size])
            correct += (logits.argmax(1) == labels[i : i + batch_size]).sum().item()
            total += len(labels[i : i + batch_size])
    model.train()
    return correct / total
