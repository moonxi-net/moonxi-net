#!/bin/bash
# Download MNIST dataset from Yann LeCun's website
# Files are placed in data/ directory

set -e

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

BASE_URL="https://yann.lecun.com/exdb/mnist"
FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

echo "Downloading MNIST dataset to $DATA_DIR/..."

for file in "${FILES[@]}"; do
  if [ -f "$DATA_DIR/${file%.gz}" ]; then
    echo "  Already exists: ${file%.gz}"
  else
    echo "  Downloading $file..."
    curl -L -o "$DATA_DIR/$file" "$BASE_URL/$file"
    echo "  Extracting..."
    gunzip -f "$DATA_DIR/$file"
  fi
done

echo "Done! MNIST files in $DATA_DIR/:"
ls -lh "$DATA_DIR"/*-idx*-ubyte