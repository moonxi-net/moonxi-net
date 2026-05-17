#!/bin/bash
# Benchmark all optimizers on CIFAR-10 ResNet-18 for 20 epochs.
# Usage: bash moonxi-net-gpu/bench/optim_bench/bench_optimizers.sh [--epochs N]
#
# Outputs: moonxi-net-gpu/bench/optim_bench/results/<optimizer>.csv per optimizer
#          moonxi-net-gpu/bench/optim_bench/results/all.csv combined

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EPOCHS=20
if [[ "${1:-}" == "--epochs" ]]; then
  EPOCHS="${2:-20}"
fi

OPTIMIZERS=("momentum_sgd" "adam" "rmsprop")
OUTDIR="moonxi-net-gpu/bench/optim_bench/results"
mkdir -p "$OUTDIR"

echo "Building..."
moon build --target native moonxi-net-gpu/examples/cifar10

for OPT in "${OPTIMIZERS[@]}"; do
  CSV="$OUTDIR/${OPT}.csv"
  echo "=== Running optimizer=$OPT epochs=$EPOCHS ==="
  echo "optimizer,epoch,lr,loss,train_acc,test_acc,epoch_s" > "$CSV"
  moon run moonxi-net-gpu/examples/cifar10 --target native --release -- \
    -e "$EPOCHS" -o "$OPT" \
    | grep "^${OPT}," >> "$CSV" || true
  echo "  → $CSV ($(wc -l < "$CSV") data rows)"
done

echo "optimizer,epoch,lr,loss,train_acc,test_acc,epoch_s" > "$OUTDIR/all.csv"
for OPT in "${OPTIMIZERS[@]}"; do
  tail -n +2 "$OUTDIR/${OPT}.csv" >> "$OUTDIR/all.csv"
done
echo "=== Combined: $OUTDIR/all.csv ==="
