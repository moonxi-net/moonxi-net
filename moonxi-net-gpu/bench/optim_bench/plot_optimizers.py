"""
Plot optimizer benchmark results from bench/results/all.csv.

Generates:
  bench/results/loss_curve.png     — training loss per epoch
  bench/results/accuracy_curve.png — test accuracy per epoch
  bench/results/epoch_time.png     — epoch wall-clock time

Usage: python3 bench/plot_optimizers.py
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "bench/results/all.csv"
OUT_DIR = "bench/results"


def load_csv(path):
    by_opt = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_opt[row["optimizer"]].append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]),
                    "train_acc": float(row["train_acc"]),
                    "test_acc": float(row["test_acc"]),
                    "epoch_s": float(row["epoch_s"]),
                }
            )
    for k in by_opt:
        by_opt[k].sort(key=lambda r: r["epoch"])
    return by_opt


def plot_and_save(by_opt, field, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for opt, rows in by_opt.items():
        epochs = [r["epoch"] for r in rows]
        vals = [r[field] for r in rows]
        ax.plot(epochs, vals, marker="o", markersize=3, label=opt)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{OUT_DIR}/{filename}"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    data = load_csv(CSV_PATH)
    if not data:
        print(f"No data in {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    plot_and_save(data, "loss", "Loss", "CIFAR-10 Training Loss", "loss_curve.png")
    plot_and_save(
        data, "test_acc", "Accuracy (%)", "CIFAR-10 Test Accuracy", "accuracy_curve.png"
    )
    plot_and_save(
        data, "epoch_s", "Seconds", "Epoch Wall-Clock Time", "epoch_time.png"
    )

    print("\nSummary (final epoch):")
    print(f"{'Optimizer':<25} {'Loss':>8} {'Test Acc':>10} {'Time/Epoch':>10}")
    print("-" * 55)
    for opt, rows in sorted(data.items()):
        last = rows[-1]
        print(
            f"{opt:<25} {last['loss']:>8.4f} {last['test_acc']:>9.2f}% {last['epoch_s']:>9.1f}s"
        )


if __name__ == "__main__":
    main()
