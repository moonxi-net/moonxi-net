"""
Plot optimizer benchmark results as SVG — zero external dependencies.

Generates:
  bench/results/loss_curve.svg     — training loss per epoch
  bench/results/accuracy_curve.svg — test accuracy per epoch
  bench/results/epoch_time.svg     — epoch wall-clock time

Usage: python3 bench/plot_optimizers_svg.py
"""

import csv
import sys
import html
from collections import OrderedDict

CSV_PATH = "moonxi-net-gpu/bench/optim_bench/results/all.csv"
OUT_DIR = "moonxi-net-gpu/bench/optim_bench/results"

COLORS = {
    "momentum_sgd": "#377eb8",
    "adam": "#984ea3",
    "rmsprop": "#ff7f00",
}

WIDTH = 900
HEIGHT = 500
MARGIN_L = 70
MARGIN_R = 30
MARGIN_T = 40
MARGIN_B = 50
PLOT_W = WIDTH - MARGIN_L - MARGIN_R
PLOT_H = HEIGHT - MARGIN_T - MARGIN_B


def load_csv(path):
    by_opt = OrderedDict()
    with open(path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (r["optimizer"], int(r["epoch"])))
    for row in rows:
        opt = row["optimizer"]
        if opt not in by_opt:
            by_opt[opt] = []
        by_opt[opt].append(
            {
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "train_acc": float(row["train_acc"]),
                "test_acc": float(row["test_acc"]),
                "epoch_s": float(row["epoch_s"]),
            }
        )
    return by_opt


def nice_ticks(lo, hi, n=6):
    import math

    span = hi - lo
    if span == 0:
        return [lo]
    step = span / n
    mag = 10 ** math.floor(math.log10(step))
    norm = step / mag
    if norm <= 1.5:
        nice = 1
    elif norm <= 3:
        nice = 2
    elif norm <= 7:
        nice = 5
    else:
        nice = 10
    step = nice * mag
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.01:
        ticks.append(round(v, 10))
        v += step
    return ticks


def fmt(v):
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def make_svg(by_opt, field, ylabel, title, filename):
    all_x = []
    all_y = []
    for rows in by_opt.values():
        for r in rows:
            all_x.append(r["epoch"])
            all_y.append(r[field])

    x_lo, x_hi = min(all_x), max(all_x)
    y_lo, y_hi = min(all_y), max(all_y)
    y_pad = (y_hi - y_lo) * 0.05 or 0.1
    y_lo -= y_pad
    y_hi += y_pad

    def sx(x):
        return MARGIN_L + (x - x_lo) / (x_hi - x_lo) * PLOT_W

    def sy(y):
        return MARGIN_T + PLOT_H - (y - y_lo) / (y_hi - y_lo) * PLOT_H

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
                 f'viewBox="0 0 {WIDTH} {HEIGHT}">')
    parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>')

    # Grid
    parts.append('<g stroke="#e0e0e0" stroke-width="0.5">')
    for t in nice_ticks(x_lo, x_hi):
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{MARGIN_T}" x2="{x:.1f}" y2="{MARGIN_T + PLOT_H}"/>')
    for t in nice_ticks(y_lo, y_hi):
        y = sy(t)
        parts.append(f'<line x1="{MARGIN_L}" y1="{y:.1f}" x2="{MARGIN_L + PLOT_W}" y2="{y:.1f}"/>')
    parts.append('</g>')

    # Axes
    parts.append(f'<g stroke="black" stroke-width="1">')
    parts.append(f'<line x1="{MARGIN_L}" y1="{MARGIN_T}" x2="{MARGIN_L}" y2="{MARGIN_T + PLOT_H}"/>')
    parts.append(f'<line x1="{MARGIN_L}" y1="{MARGIN_T + PLOT_H}" x2="{MARGIN_L + PLOT_W}" y2="{MARGIN_T + PLOT_H}"/>')
    parts.append('</g>')

    # X ticks
    parts.append('<g font-family="sans-serif" font-size="11" text-anchor="middle">')
    for t in nice_ticks(x_lo, x_hi):
        x = sx(t)
        parts.append(f'<text x="{x:.1f}" y="{MARGIN_T + PLOT_H + 18}">{int(t)}</text>')
    parts.append(f'<text x="{MARGIN_L + PLOT_W / 2}" y="{HEIGHT - 5}" '
                 f'font-size="13" font-weight="bold">Epoch</text>')
    parts.append('</g>')

    # Y ticks
    parts.append('<g font-family="sans-serif" font-size="11" text-anchor="end">')
    for t in nice_ticks(y_lo, y_hi):
        y = sy(t)
        parts.append(f'<text x="{MARGIN_L - 6}" y="{y + 4:.1f}">{fmt(t)}</text>')
    parts.append(f'<text x="14" y="{MARGIN_T + PLOT_H / 2}" '
                 f'font-size="13" font-weight="bold" text-anchor="middle" '
                 f'transform="rotate(-90,14,{MARGIN_T + PLOT_H / 2})">{html.escape(ylabel)}</text>')
    parts.append('</g>')

    # Title
    parts.append(f'<text x="{WIDTH / 2}" y="22" font-family="sans-serif" font-size="16" '
                 f'font-weight="bold" text-anchor="middle">{html.escape(title)}</text>')

    # Data lines
    for opt, rows in by_opt.items():
        color = COLORS.get(opt, "#333333")
        pts = " ".join(f"{sx(r['epoch']):.1f},{sy(r[field]):.1f}" for r in rows)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
        # Dots at each point
        for r in rows:
            cx, cy = sx(r["epoch"]), sy(r[field])
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}"/>')

    # Legend
    lx = MARGIN_L + PLOT_W - 180
    ly = MARGIN_T + 10
    parts.append(f'<rect x="{lx}" y="{ly}" width="170" height="{len(by_opt) * 22 + 8}" '
                 f'fill="white" stroke="#ccc" rx="4"/>')
    for i, opt in enumerate(by_opt):
        color = COLORS.get(opt, "#333333")
        yy = ly + 14 + i * 22
        parts.append(f'<line x1="{lx + 8}" y1="{yy}" x2="{lx + 28}" y2="{yy}" '
                     f'stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 34}" y="{yy + 4}" font-family="sans-serif" '
                     f'font-size="11">{html.escape(opt)}</text>')

    parts.append("</svg>")

    path = f"{OUT_DIR}/{filename}"
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"Saved {path}")


def main():
    data = load_csv(CSV_PATH)
    if not data:
        print(f"No data in {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    make_svg(data, "loss", "Loss", "CIFAR-10 Training Loss", "loss_curve.svg")
    make_svg(data, "test_acc", "Accuracy (%)", "CIFAR-10 Test Accuracy", "accuracy_curve.svg")
    make_svg(data, "epoch_s", "Seconds", "Epoch Wall-Clock Time", "epoch_time.svg")

    print("\nSummary (final epoch):")
    print(f"{'Optimizer':<25} {'Loss':>8} {'Test Acc':>10} {'Time/Epoch':>10}")
    print("-" * 55)
    for opt, rows in data.items():
        last = rows[-1]
        print(
            f"{opt:<25} {last['loss']:>8.4f} {last['test_acc']:>9.2f}% {last['epoch_s']:>9.1f}s"
        )


if __name__ == "__main__":
    main()
