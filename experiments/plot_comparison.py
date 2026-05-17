#!/usr/bin/env python3
"""Generate SVG comparison plots from arbitrary training CSV series."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Style per framework: detect from label prefix
# MoonBit → pink family, PyTorch → light blue family
# Within each family, differentiate by line style only (no markers)
MOONBIT_COLORS = ["#e91e63", "#f06292", "#f48fb1"]  # pink → light pink
PYTORCH_COLORS = ["#2196f3", "#64b5f6", "#90caf9"]  # blue → light blue
FAMILY_LINESTYLES = ["-", "--", "-."]


def _series_style(label: str, moonbit_idx: int, pytorch_idx: int):
    if label.lower().startswith("moonbit"):
        color = MOONBIT_COLORS[moonbit_idx % len(MOONBIT_COLORS)]
        ls = FAMILY_LINESTYLES[moonbit_idx % len(FAMILY_LINESTYLES)]
        return color, ls
    else:
        color = PYTORCH_COLORS[pytorch_idx % len(PYTORCH_COLORS)]
        ls = FAMILY_LINESTYLES[pytorch_idx % len(FAMILY_LINESTYLES)]
        return color, ls


def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  [skip] file not found: {path}")
        return None
    try:
        df = pd.read_csv(path, nrows=1)
        first_col = df.columns[0]
        try:
            float(first_col)
            df = pd.read_csv(
                path,
                header=None,
                names=["epoch", "lr", "loss", "train_acc", "test_acc", "time_s"],
            )
        except ValueError:
            df = pd.read_csv(path)
            if "epoch_s" in df.columns and "time_s" not in df.columns:
                df = df.rename(columns={"epoch_s": "time_s"})
            elif "total_s" in df.columns and "time_s" not in df.columns:
                df = df.rename(columns={"total_s": "time_s"})
        return df
    except Exception as e:
        print(f"  [error] reading {path}: {e}")
        return None


def resolve_time_col(df: pd.DataFrame) -> pd.Series | None:
    for col in ("time_s", "epoch_s", "total_s"):
        if col in df.columns:
            return df[col]
    return None


def plot_metric(
    series_data: list[tuple[str, pd.DataFrame]],
    col: str,
    ylabel: str,
    title: str,
    out_path: str,
    legend_loc: str,
):
    available = []
    for label, df in series_data:
        if col in df.columns:
            available.append((label, df))
        else:
            print(f"  [skip] '{label}' missing column '{col}'")

    if not available:
        print(f"  [skip] {os.path.basename(out_path)} — column '{col}' missing in all series")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fig.patch.set_facecolor("white")

    moonbit_idx = 0
    pytorch_idx = 0
    for label, df in available:
        color, ls = _series_style(label, moonbit_idx, pytorch_idx)
        if label.lower().startswith("moonbit"):
            moonbit_idx += 1
        else:
            pytorch_idx += 1
        ax.plot(
            df["epoch"],
            df[col],
            color=color,
            linestyle=ls,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  [wrote] {out_path}")


def plot_epoch_time(
    series_data: list[tuple[str, pd.DataFrame]],
    title: str,
    out_path: str,
):
    available = []
    for label, df in series_data:
        time = resolve_time_col(df)
        if time is not None:
            available.append((label, df, time))
        else:
            print(f"  [skip] '{label}' missing epoch-time column")

    if not available:
        print(f"  [skip] {os.path.basename(out_path)} — no epoch-time data in any series")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fig.patch.set_facecolor("white")

    moonbit_idx = 0
    pytorch_idx = 0
    for label, df, time in available:
        color, ls = _series_style(label, moonbit_idx, pytorch_idx)
        if label.lower().startswith("moonbit"):
            moonbit_idx += 1
        else:
            pytorch_idx += 1
        ax.plot(
            df["epoch"],
            time,
            color=color,
            linestyle=ls,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  [wrote] {out_path}")


def parse_series(series_args: list[str], reports_dir: str) -> list[tuple[str, pd.DataFrame]]:
    result = []
    for spec in series_args:
        if ":" not in spec:
            print(f"[error] invalid --series '{spec}': expected LABEL:FILENAME")
            sys.exit(1)
        label, filename = spec.split(":", 1)
        path = os.path.join(reports_dir, filename)
        print(f"Loading '{label}': {path}")
        df = load_csv(path)
        if df is not None:
            result.append((label, df))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate SVG comparison plots from training CSV series"
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL:CSV_FILENAME",
        help="A series to plot (repeatable). FILENAME is relative to --reports-dir.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Output file prefix (e.g. mnist_20260516). Empty = metric-only filenames.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory containing CSVs and where SVGs go (default: reports)",
    )
    args = parser.parse_args()

    series_data = parse_series(args.series, args.reports_dir)

    if not series_data:
        print("No data files loaded. Exiting.")
        sys.exit(1)

    prefix = args.prefix
    rdir = args.reports_dir
    generated = []

    sep = "_" if prefix else ""
    label_prefix = f"{prefix} " if prefix else ""

    print(f"\nGenerating plots for '{prefix or rdir}' ({len(series_data)} series):")

    plot_metric(
        series_data,
        "loss",
        "Loss",
        f"{label_prefix}Loss",
        os.path.join(rdir, f"{prefix}{sep}loss.svg"),
        "upper right",
    )
    generated.append(f"{prefix}{sep}loss.svg")

    plot_metric(
        series_data,
        "test_acc",
        "Test Accuracy (%)",
        f"{label_prefix}Test Accuracy",
        os.path.join(rdir, f"{prefix}{sep}accuracy.svg"),
        "lower right",
    )
    generated.append(f"{prefix}{sep}accuracy.svg")

    plot_metric(
        series_data,
        "train_acc",
        "Train Accuracy (%)",
        f"{label_prefix}Train Accuracy",
        os.path.join(rdir, f"{prefix}{sep}train_accuracy.svg"),
        "lower right",
    )
    generated.append(f"{prefix}{sep}train_accuracy.svg")

    plot_epoch_time(
        series_data,
        f"{label_prefix}Epoch Time",
        os.path.join(rdir, f"{prefix}{sep}epoch_time.svg"),
    )
    generated.append(f"{prefix}{sep}epoch_time.svg")

    print(f"\nDone. Generated: {', '.join(generated)}")


if __name__ == "__main__":
    main()
