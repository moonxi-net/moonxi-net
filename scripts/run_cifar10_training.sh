#!/usr/bin/env bash
# run_cifar10_training.sh — CIFAR-10 ResNet-18 training comparison:
#   MoonBit GPU backend vs PyTorch backend, across 3 optimizers.
#
# Configurations (3 optimizers × 2 frameworks = 6 runs):
#   MoonBit: momentum_sgd, adam, rmsprop
#   PyTorch: sgd,          adam, rmsprop
#
# Usage:
#   ./scripts/run_cifar10_training.sh                # 20 epochs default
#   ./scripts/run_cifar10_training.sh --epochs 10    # custom epochs
#   ./scripts/run_cifar10_training.sh --no-build     # skip build
#   ./scripts/run_cifar10_training.sh --epochs 10 --no-build

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT/data/cifar-10-batches-bin"

# CUDA / cuDNN library path
export LD_LIBRARY_PATH="$PROJECT/cuda/lib/cudnn:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Optimizers
OPTIMIZERS=(sgd adam rmsprop)

# MoonBit optimizer flag names
MOONBIT_OPTIM_FLAGS=(momentum_sgd adam rmsprop)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

EPOCHS=20
NO_BUILD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --no-build)
      NO_BUILD=true
      shift
      ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

section() {
  echo ""
  echo -e "${YELLOW}━━━ $1 ━━━${NC}"
}

elapsed() {
  local start="$1" label="$2"
  local dur
  dur=$(( SECONDS - start ))
  local mins=$(( dur / 60 ))
  local secs=$(( dur % 60 ))
  echo -e "${CYAN}  [${label}] elapsed: ${mins}m ${secs}s${NC}"
}

# Run one MoonBit training and extract CSV.
# Usage: run_moonbit <optim_index> <phase_number> <phase_label>
# Globals used: RUN_ID, EPOCHS, REPORTS_DIR, PROJECT
run_moonbit() {
  local idx="$1"
  local phase="$2"
  local label="$3"
  local optim="${OPTIMIZERS[$idx]}"
  local flag="${MOONBIT_OPTIM_FLAGS[$idx]}"

  section "Phase ${phase}: MoonBit ${label}"

  local LOG="$REPORTS_DIR/moonbit_${optim}.log"
  local CSV="$REPORTS_DIR/moonbit_${optim}.csv"

  echo -e "  Log  : $LOG"
  echo -e "  CSV  : $CSV"
  echo ""

  local START=$SECONDS

  moon run moonxi-net-gpu/examples/cifar10 --target native --release -- -e "$EPOCHS" -o "$flag" \
    2>&1 | tee "$LOG"

  elapsed "$START" "MoonBit ${label}"

  # Extract CSV lines (lines starting with digit followed by comma)
  grep -E '^[0-9][0-9]*,' "$LOG" > "$CSV" || true

  local CSV_LINES
  CSV_LINES=$(wc -l < "$CSV")
  echo -e "  Extracted ${CSV_LINES} epoch rows to ${CSV}"

  if [[ "$CSV_LINES" -eq 0 ]]; then
    echo -e "${RED}Warning: No CSV rows found in MoonBit ${label} output${NC}"
  fi
}

# Run one PyTorch training and extract CSV.
# Usage: run_pytorch <optim_index> <phase_number> <phase_label>
# Globals used: RUN_ID, EPOCHS, REPORTS_DIR, PROJECT
run_pytorch() {
  local idx="$1"
  local phase="$2"
  local label="$3"
  local optim="${OPTIMIZERS[$idx]}"

  section "Phase ${phase}: PyTorch ${label}"

  local LOG="$REPORTS_DIR/pytorch_${optim}.log"
  local CSV="$REPORTS_DIR/pytorch_${optim}.csv"

  echo -e "  Log  : $LOG"
  echo -e "  CSV  : $CSV"
  echo ""

  local START=$SECONDS

  (
    cd "$PROJECT/experiments"
    uv run train_cifar10.py -e "$EPOCHS" -o "$optim" 2>&1
  ) | tee "$LOG"

  elapsed "$START" "PyTorch ${label}"

  # Parse PyTorch human-readable output into CSV
  # Format: "Epoch E/N loss=L train_acc=A% test_acc=B% ... total=T s"
  echo "epoch,loss,train_acc,test_acc,total_s" > "$CSV"

  grep -P 'Epoch \d+/\d+ loss=' "$LOG" | grep -oP 'Epoch \K[0-9]+(?=/)' > /tmp/_pytorch_epochs.txt || true
  grep -P 'Epoch \d+/\d+ loss=' "$LOG" | grep -oP 'loss=\K[0-9.]+' > /tmp/_pytorch_loss.txt || true
  grep -P 'Epoch \d+/\d+ loss=' "$LOG" | grep -oP 'train_acc=\K[0-9.]+' > /tmp/_pytorch_train_acc.txt || true
  grep -P 'Epoch \d+/\d+ loss=' "$LOG" | grep -oP 'test_acc=\K[0-9.]+' > /tmp/_pytorch_test_acc.txt || true
  grep -P 'Epoch \d+/\d+ loss=' "$LOG" | grep -oP 'total=\K[0-9.]+' > /tmp/_pytorch_total.txt || true

  paste -d',' \
    /tmp/_pytorch_epochs.txt \
    /tmp/_pytorch_loss.txt \
    /tmp/_pytorch_train_acc.txt \
    /tmp/_pytorch_test_acc.txt \
    /tmp/_pytorch_total.txt \
    >> "$CSV" 2>/dev/null || true

  rm -f /tmp/_pytorch_epochs.txt /tmp/_pytorch_loss.txt \
        /tmp/_pytorch_train_acc.txt /tmp/_pytorch_test_acc.txt \
        /tmp/_pytorch_total.txt

  local CSV_LINES
  CSV_LINES=$(($(wc -l < "$CSV") - 1))  # subtract header
  echo -e "  Extracted ${CSV_LINES} epoch rows to ${CSV}"

  if [[ "$CSV_LINES" -eq 0 ]]; then
    echo -e "${RED}Warning: No CSV rows parsed from PyTorch ${label} output${NC}"
  fi
}

# Get last CSV data line from a file. For PyTorch CSVs (with header), skip header.
# Prints: last_data_line
get_last_csv_line() {
  local csv="$1"
  local has_header="$2"  # "yes" or "no"
  if [[ ! -f "$csv" ]]; then
    echo ""
    return
  fi
  local total
  total=$(wc -l < "$csv")
  if [[ "$total" -eq 0 ]]; then
    echo ""
    return
  fi
  if [[ "$has_header" == "yes" ]]; then
    if [[ "$total" -le 1 ]]; then
      echo ""
      return
    fi
    tail -1 "$csv"
  else
    tail -1 "$csv"
  fi
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

RUN_ID="cifar10_$(date +%Y%m%d_%H%M%S)"
REPORTS_DIR="$PROJECT/reports/$RUN_ID"

mkdir -p "$REPORTS_DIR"

section "CIFAR-10 ResNet-18 Training Comparison (3 Optimizers × 2 Frameworks)"
echo "  Run ID   : $RUN_ID"
echo "  Epochs   : $EPOCHS"
echo "  Reports  : $REPORTS_DIR"
echo "  Project  : $PROJECT"
echo "  Optimizers: ${OPTIMIZERS[*]}"

# ---------------------------------------------------------------------------
# Phase 1: Build (unless --no-build)
# ---------------------------------------------------------------------------

if [[ "$NO_BUILD" == false ]]; then
  section "Phase 1: Build"
  BUILD_START=$SECONDS
  moon build --target native --release 2>&1 | tail -5
  elapsed "$BUILD_START" "Build"
fi

# ---------------------------------------------------------------------------
# Phase 2: Check CIFAR-10 data
# ---------------------------------------------------------------------------

section "Phase 2: Verify CIFAR-10 data"

if [[ ! -d "$DATA_DIR" ]]; then
  echo -e "${RED}Error: CIFAR-10 data directory not found at $DATA_DIR${NC}"
  echo "  Download with: bash scripts/download_cifar10.sh"
  exit 1
fi

TRAIN_FILE_COUNT=$(ls "$DATA_DIR"/data_batch_*.bin 2>/dev/null | wc -l)
TEST_FILE="$DATA_DIR/test_batch.bin"

echo "  Training batches : $TRAIN_FILE_COUNT / 5"
echo "  Test batch       : $(test -f "$TEST_FILE" && echo "OK" || echo "MISSING")"

if [[ "$TRAIN_FILE_COUNT" -lt 5 || ! -f "$TEST_FILE" ]]; then
  echo -e "${RED}Error: Incomplete CIFAR-10 dataset.${NC}"
  echo "  Expected 5 data_batch_*.bin + 1 test_batch.bin in $DATA_DIR"
  exit 1
fi

echo -e "${GREEN}  Data OK${NC}"

# ---------------------------------------------------------------------------
# Phases 3-5: Run MoonBit SGD / Adam / RMSprop
# ---------------------------------------------------------------------------

for i in 0 1 2; do
  run_moonbit "$i" "$((3 + i))" "${OPTIMIZERS[$i]^}"
done

# ---------------------------------------------------------------------------
# Phases 6-8: Run PyTorch SGD / Adam / RMSprop
# ---------------------------------------------------------------------------

for i in 0 1 2; do
  run_pytorch "$i" "$((6 + i))" "${OPTIMIZERS[$i]^}"
done

# ---------------------------------------------------------------------------
# Phase 9: Generate comparison plots
# ---------------------------------------------------------------------------

section "Phase 9: Generate comparison plots"

PLOT_SCRIPT="$PROJECT/experiments/plot_comparison.py"
if [[ -f "$PLOT_SCRIPT" ]]; then
  PLOT_START=$SECONDS
  (
    cd "$PROJECT/experiments"
    uv run plot_comparison.py \
      --reports-dir "$REPORTS_DIR" \
      --series "MoonBit SGD:moonbit_sgd.csv" \
      --series "MoonBit Adam:moonbit_adam.csv" \
      --series "MoonBit RMSprop:moonbit_rmsprop.csv" \
      --series "PyTorch SGD:pytorch_sgd.csv" \
      --series "PyTorch Adam:pytorch_adam.csv" \
      --series "PyTorch RMSprop:pytorch_rmsprop.csv"
  ) || {
    echo -e "${RED}Warning: Plot generation failed (continuing)${NC}"
  }
  elapsed "$PLOT_START" "Plotting"
else
  echo -e "${YELLOW}  Skipping: $PLOT_SCRIPT not found yet${NC}"
fi

# ---------------------------------------------------------------------------
# Phase 10: Generate summary markdown
# ---------------------------------------------------------------------------

section "Phase 10: Generate summary report"

SUMMARY="$REPORTS_DIR/summary.md"

# Detect GPU model
GPU_MODEL="N/A"
if command -v nvidia-smi &>/dev/null; then
  GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
fi

TODAY=$(date +%Y-%m-%d)

# Find plot SVGs if they exist
PLOT_FILES=$(ls "$REPORTS_DIR"/*.svg 2>/dev/null || true)

# Collect final-epoch results for each optimizer × framework
# Arrays indexed by optimizer (0=sgd, 1=adam, 2=rmsprop)
declare -a MOONBIT_FINAL_LOSS MOONBIT_FINAL_TRAIN_ACC MOONBIT_FINAL_TEST_ACC MOONBIT_FINAL_EPOCH_S
declare -a PYTORCH_FINAL_LOSS PYTORCH_FINAL_TRAIN_ACC PYTORCH_FINAL_TEST_ACC PYTORCH_FINAL_TOTAL_S
declare -a MOONBIT_FINAL_EPOCH PYTORCH_FINAL_EPOCH

for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"

  # MoonBit final epoch: epoch,lr,loss,train_acc,test_acc,epoch_s
  MOONBIT_LAST=$(get_last_csv_line "$REPORTS_DIR/moonbit_${local_optim}.csv" "no")
  if [[ -n "$MOONBIT_LAST" ]]; then
    IFS=',' read -r me mlr mloss mtrain_acc mtest_acc mepoch_s <<< "$MOONBIT_LAST"
    MOONBIT_FINAL_EPOCH[$i]="$me"
    MOONBIT_FINAL_LOSS[$i]="$mloss"
    MOONBIT_FINAL_TRAIN_ACC[$i]="$mtrain_acc"
    MOONBIT_FINAL_TEST_ACC[$i]="$mtest_acc"
    MOONBIT_FINAL_EPOCH_S[$i]="$mepoch_s"
  else
    MOONBIT_FINAL_EPOCH[$i]="N/A"
    MOONBIT_FINAL_LOSS[$i]="N/A"
    MOONBIT_FINAL_TRAIN_ACC[$i]="N/A"
    MOONBIT_FINAL_TEST_ACC[$i]="N/A"
    MOONBIT_FINAL_EPOCH_S[$i]="N/A"
  fi

  # PyTorch final epoch: epoch,loss,train_acc,test_acc,total_s
  PYTORCH_LAST=$(get_last_csv_line "$REPORTS_DIR/pytorch_${local_optim}.csv" "yes")
  if [[ -n "$PYTORCH_LAST" ]]; then
    IFS=',' read -r pepoch ploss ptrain_acc ptest_acc ptotal_s <<< "$PYTORCH_LAST"
    PYTORCH_FINAL_EPOCH[$i]="$pepoch"
    PYTORCH_FINAL_LOSS[$i]="$ploss"
    PYTORCH_FINAL_TRAIN_ACC[$i]="$ptrain_acc"
    PYTORCH_FINAL_TEST_ACC[$i]="$ptest_acc"
    PYTORCH_FINAL_TOTAL_S[$i]="$ptotal_s"
  else
    PYTORCH_FINAL_EPOCH[$i]="N/A"
    PYTORCH_FINAL_LOSS[$i]="N/A"
    PYTORCH_FINAL_TRAIN_ACC[$i]="N/A"
    PYTORCH_FINAL_TEST_ACC[$i]="N/A"
    PYTORCH_FINAL_TOTAL_S[$i]="N/A"
  fi
done

# Build the summary markdown
cat > "$SUMMARY" << SUMMARY_EOF
# CIFAR-10 ResNet-18 Training Comparison — ${TODAY}

**Run ID:** \`${RUN_ID}\`

## Setup

| Parameter | Value |
|-----------|-------|
| Model | ResNet-18 |
| Epochs | ${EPOCHS} |
| Batch size | 128 |
| Optimizers | SGD, Adam, RMSprop |
| GPU | ${GPU_MODEL} |

## Final Epoch Results (All 6 Configurations)

| # | Framework | Optimizer | Loss | Train Acc | Test Acc | Epoch Time |
|--:|-----------|-----------|-----:|----------:|---------:|-----------:|
SUMMARY_EOF

for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"
  local_label="${local_optim^^}"
  echo "| $((i * 2 + 1)) | MoonBit | ${local_label} | ${MOONBIT_FINAL_LOSS[$i]:-N/A} | ${MOONBIT_FINAL_TRAIN_ACC[$i]:-N/A}% | ${MOONBIT_FINAL_TEST_ACC[$i]:-N/A}% | ${MOONBIT_FINAL_EPOCH_S[$i]:-N/A}s |" >> "$SUMMARY"
  echo "| $((i * 2 + 2)) | PyTorch | ${local_label} | ${PYTORCH_FINAL_LOSS[$i]:-N/A} | ${PYTORCH_FINAL_TRAIN_ACC[$i]:-N/A}% | ${PYTORCH_FINAL_TEST_ACC[$i]:-N/A}% | ${PYTORCH_FINAL_TOTAL_S[$i]:-N/A}s |" >> "$SUMMARY"
done

# Per-optimizer comparison sections
for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"
  local_label="${local_optim^^}"

  cat >> "$SUMMARY" << OPTIM_EOF

## ${local_label}: MoonBit vs PyTorch

| Metric | MoonBit | PyTorch |
|--------|---------|---------|
| Final Epoch | ${MOONBIT_FINAL_EPOCH[$i]:-N/A} | ${PYTORCH_FINAL_EPOCH[$i]:-N/A} |
| Loss | ${MOONBIT_FINAL_LOSS[$i]:-N/A} | ${PYTORCH_FINAL_LOSS[$i]:-N/A} |
| Train Accuracy | ${MOONBIT_FINAL_TRAIN_ACC[$i]:-N/A}% | ${PYTORCH_FINAL_TRAIN_ACC[$i]:-N/A}% |
| Test Accuracy | ${MOONBIT_FINAL_TEST_ACC[$i]:-N/A}% | ${PYTORCH_FINAL_TEST_ACC[$i]:-N/A}% |
| Epoch Time | ${MOONBIT_FINAL_EPOCH_S[$i]:-N/A}s | ${PYTORCH_FINAL_TOTAL_S[$i]:-N/A}s |
OPTIM_EOF
done

# Files section
cat >> "$SUMMARY" << FILES_EOF

## Files

### Logs
FILES_EOF

for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"
  echo "- MoonBit ${local_optim^^}: [\`moonbit_${local_optim}.log\`](./moonbit_${local_optim}.log)" >> "$SUMMARY"
done
for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"
  echo "- PyTorch ${local_optim^^}: [\`pytorch_${local_optim}.log\`](./pytorch_${local_optim}.log)" >> "$SUMMARY"
done

echo "" >> "$SUMMARY"
echo "### CSV" >> "$SUMMARY"

for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"
  echo "- MoonBit ${local_optim^^}: [\`moonbit_${local_optim}.csv\`](./moonbit_${local_optim}.csv)" >> "$SUMMARY"
done
for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"
  echo "- PyTorch ${local_optim^^}: [\`pytorch_${local_optim}.csv\`](./pytorch_${local_optim}.csv)" >> "$SUMMARY"
done

echo "" >> "$SUMMARY"
echo "### Plots" >> "$SUMMARY"

if [[ -n "$PLOT_FILES" ]]; then
  for svg in $PLOT_FILES; do
    echo "- [$(basename "$svg")]($(basename "$svg"))" >> "$SUMMARY"
  done
else
  echo "- _No plots generated_" >> "$SUMMARY"
fi

echo ""
echo -e "${GREEN}  Summary: $SUMMARY${NC}"

# ---------------------------------------------------------------------------
# Convergence Summary
# ---------------------------------------------------------------------------

section "Convergence Summary"

echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│         CIFAR-10 ResNet-18 — Best Test Accuracy per Optimizer   │"
echo "├────────────┬───────────────────┬────────────────────────────────┤"
echo "│ Optimizer  │ MoonBit GPU       │ PyTorch                        │"
echo "├────────────┼───────────────────┼────────────────────────────────┤"

for i in 0 1 2; do
  local_optim="${OPTIMIZERS[$i]}"
  local_label="${local_optim^^}"
  printf "│ %-10s │ %14s%%  │ %14s%%             │\n" \
    "$local_label" \
    "${MOONBIT_FINAL_TEST_ACC[$i]:-—}" \
    "${PYTORCH_FINAL_TEST_ACC[$i]:-—}"
done

echo "└────────────┴───────────────────┴────────────────────────────────┘"
echo ""
echo -e "${GREEN}  Run ID: $RUN_ID${NC}"
echo -e "${GREEN}  Reports: $REPORTS_DIR/${NC}"
echo ""
