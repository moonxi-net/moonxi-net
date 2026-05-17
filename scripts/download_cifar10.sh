#!/usr/bin/env bash
# Download and extract CIFAR-10 binary version.
# Data ends up at data/cifar-10-batches-bin/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT/data/cifar-10-batches-bin"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

if [[ -f "$DATA_DIR/data_batch_1.bin" ]]; then
  echo "CIFAR-10 data already exists at $DATA_DIR"
  exit 0
fi

echo "Downloading CIFAR-10 binary from $URL ..."
mkdir -p "$PROJECT/data"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

curl -L --progress-bar -o "$tmp" "$URL"

echo "Extracting ..."
tar xzf "$tmp" -C "$PROJECT/data"

if [[ -f "$DATA_DIR/data_batch_1.bin" ]]; then
  echo "Done. $(ls "$DATA_DIR"/*.bin | wc -l) files in $DATA_DIR"
else
  echo "Error: extraction failed, expected files in $DATA_DIR" >&2
  exit 1
fi
