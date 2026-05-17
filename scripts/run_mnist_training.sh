#!/usr/bin/env bash
# run_mnist_training.sh — Run MNIST MLP training across 4 backend configurations,
# capture CSV logs, generate comparison plots & summary.
#
# Configurations:
#   1. MoonBit GPU
#   2. MoonBit CPU
#   3. PyTorch CPU
#   4. PyTorch GPU
#
# Usage:
#   ./run_mnist_training.sh                      # defaults: 20 epochs, batch 64
#   ./run_mnist_training.sh --epochs 20          # custom epochs
#   ./run_mnist_training.sh --batch-size 128     # custom batch size
#   ./run_mnist_training.sh --no-build           # skip moon build
#   ./run_mnist_training.sh -e 5 -b 32 --no-build

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Defaults ─────────────────────────────────────────────────────────────────
EPOCHS=20
BATCH_SIZE=64
NO_BUILD=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            echo "Usage: $0 [-e|--epochs E] [-b|--batch-size B] [--no-build]"
            exit 1
            ;;
    esac
done

# ── Environment ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$PROJECT/cuda/lib/cudnn:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── Run ID ───────────────────────────────────────────────────────────────────
RUN_ID="mnist_$(date +%Y%m%d_%H%M%S)"

# ── Reports directory (per-run subdirectory) ─────────────────────────────────
REPORTS_DIR="$PROJECT/reports/$RUN_ID"
mkdir -p "$REPORTS_DIR"

# ── Helper: elapsed string ───────────────────────────────────────────────────
elapsed_str() {
    local secs="$1"
    local m=$((secs / 60))
    local s=$((secs % 60))
    printf "%dm%02ds" "$m" "$s"
}

# ── Helper: detect GPU model ─────────────────────────────────────────────────
detect_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown"
    else
        echo "n/a"
    fi
}

# ── Helper: run one configuration ────────────────────────────────────────────
# Usage: run_config <label> <log_suffix> <command...>
# Sets LOG_FILE and CSV_FILE globals for the caller.
run_config() {
    local label="$1"
    local suffix="$2"
    shift 2

    local log="$REPORTS_DIR/${suffix}.log"
    local csv="$REPORTS_DIR/${suffix}.csv"

    local start end
    start=$(date +%s)

    echo -e "${CYAN}  → ${label}${NC}"
    "$@" 2>&1 | tee "$log" || true

    end=$(date +%s)

    # Extract CSV lines (lines starting with digit)
    grep -E '^[0-9][0-9]*,' "$log" > "$csv" || true

    local nrows=0
    if [[ -f "$csv" ]]; then nrows=$(wc -l < "$csv"); fi

    echo -e "  ${GREEN}✓ ${label} done in $(elapsed_str $((end - start)))${NC}  (${nrows} epoch rows)"
    echo "    Log: ${log}"
    echo "    CSV: ${csv}"
    echo ""

    # Return metrics via globals
    eval "LOG_${suffix}=\"${log}\""
    eval "CSV_${suffix}=\"${csv}\""
    eval "DURATION_${suffix}=$((end - start))"
}

# ── Helper: read last line of CSV ────────────────────────────────────────────
read_csv_last() {
    local csv="$1"
    if [[ -f "$csv" && -s "$csv" ]]; then
        tail -1 "$csv"
    else
        echo "n/a,n/a,n/a,n/a,n/a,n/a"
    fi
}

# ── Header ───────────────────────────────────────────────────────────────────
echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║       MNIST MLP Training — 4-Backend Comparison              ║${NC}"
echo -e "${YELLOW}║       MoonBit GPU · MoonBit CPU · PyTorch CPU · PyTorch GPU   ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Run ID     : $RUN_ID"
echo "  Epochs     : $EPOCHS"
echo "  Batch size : $BATCH_SIZE"
echo "  GPU        : $(detect_gpu)"
echo "  Reports    : $REPORTS_DIR/"
echo ""

TOTAL_START=$(date +%s)

# ── Phase 1: Build ───────────────────────────────────────────────────────────
if [[ "$NO_BUILD" == false ]]; then
    echo -e "${YELLOW}[1/7] Building MoonBit project...${NC}"
    BUILD_START=$(date +%s)
    moon build --target native --release 2>&1 | tail -5
    BUILD_END=$(date +%s)
    echo -e "  ${GREEN}Build done in $(elapsed_str $((BUILD_END - BUILD_START)))${NC}"
    echo ""
else
    echo -e "${YELLOW}[1/7] Build skipped (--no-build)${NC}"
    echo ""
fi

# ── Phase 2: MoonBit GPU ─────────────────────────────────────────────────────
echo -e "${YELLOW}[2/7] Running MoonBit GPU MNIST (${EPOCHS} epochs, batch ${BATCH_SIZE})...${NC}"
run_config "MoonBit GPU" "moonbit_gpu" \
    moon run moonxi-net-gpu/examples/mnist --target native --release -- \
    -e "$EPOCHS" -b "$BATCH_SIZE" --gpu

# ── Phase 3: MoonBit CPU ─────────────────────────────────────────────────────
echo -e "${YELLOW}[3/7] Running MoonBit CPU MNIST (${EPOCHS} epochs, batch ${BATCH_SIZE})...${NC}"
run_config "MoonBit CPU" "moonbit_cpu" \
    moon run moonxi-net-gpu/examples/mnist --target native --release -- \
    -e "$EPOCHS" -b "$BATCH_SIZE"

# ── Phase 4: PyTorch CPU ─────────────────────────────────────────────────────
echo -e "${YELLOW}[4/7] Running PyTorch CPU MNIST (${EPOCHS} epochs, batch ${BATCH_SIZE})...${NC}"
run_config "PyTorch CPU" "pytorch_cpu" \
    bash -c "cd '$PROJECT/experiments' && uv run pytorch_mnist_mlp.py -e '$EPOCHS' -b '$BATCH_SIZE'"

# ── Phase 5: PyTorch GPU ─────────────────────────────────────────────────────
echo -e "${YELLOW}[5/7] Running PyTorch GPU MNIST (${EPOCHS} epochs, batch ${BATCH_SIZE})...${NC}"
run_config "PyTorch GPU" "pytorch_gpu" \
    bash -c "cd '$PROJECT/experiments' && uv run pytorch_mnist_mlp.py -e '$EPOCHS' -b '$BATCH_SIZE' --gpu"

# ── Phase 6: Plotting ────────────────────────────────────────────────────────
echo -e "${YELLOW}[6/7] Generating comparison plots...${NC}"
PLOT_START=$(date +%s)
(
  cd "$PROJECT/experiments"
  uv run plot_comparison.py \
    --prefix "$RUN_ID" \
    --reports-dir "$REPORTS_DIR" \
    --series "MoonBit GPU:moonbit_gpu.csv" \
    --series "MoonBit CPU:moonbit_cpu.csv" \
    --series "PyTorch CPU:pytorch_cpu.csv" \
    --series "PyTorch GPU:pytorch_gpu.csv"
) || {
    echo -e "  ${RED}Plotting failed${NC}"
}
PLOT_END=$(date +%s)
echo -e "  ${GREEN}Plots done in $(elapsed_str $((PLOT_END - PLOT_START)))${NC}"
echo ""

# ── Phase 7: Generate summary markdown ──────────────────────────────────────
echo -e "${YELLOW}[7/7] Generating summary...${NC}"

SUMMARY="$REPORTS_DIR/summary.md"
DATE_STR="$(date +%Y-%m-%d\ %H:%M:%S)"
GPU_MODEL="$(detect_gpu)"

# CSV file paths (populated by run_config)
CSV_MG="$REPORTS_DIR/moonbit_gpu.csv"
CSV_MC="$REPORTS_DIR/moonbit_cpu.csv"
CSV_PC="$REPORTS_DIR/pytorch_cpu.csv"
CSV_PG="$REPORTS_DIR/pytorch_gpu.csv"

# Extract final-epoch metrics from each CSV
# CSV format: epoch,lr,loss,train_acc,test_acc,epoch_s
MG_LAST="$(read_csv_last "$CSV_MG")"
MC_LAST="$(read_csv_last "$CSV_MC")"
PC_LAST="$(read_csv_last "$CSV_PC")"
PG_LAST="$(read_csv_last "$CSV_PG")"

# Parse fields: epoch,lr,loss,train_acc,test_acc,epoch_s
IFS=',' read -r _ _ mg_loss mg_train_acc mg_test_acc mg_epoch_s <<< "$MG_LAST"
IFS=',' read -r _ _ mc_loss mc_train_acc mc_test_acc mc_epoch_s <<< "$MC_LAST"
IFS=',' read -r _ _ pc_loss pc_train_acc pc_test_acc pc_epoch_s <<< "$PC_LAST"
IFS=',' read -r _ _ pg_loss pg_train_acc pg_test_acc pg_epoch_s <<< "$PG_LAST"

# ── Per-epoch comparison table (4 configs side by side) ──────────────────────
per_epoch_table() {
    local csv_mg="$1" csv_mc="$2" csv_pc="$3" csv_pg="$4"

    # Read each CSV into arrays
    local -a arr_mg=() arr_mc=() arr_pc=() arr_pg=()
    local nmg=0 nmc=0 npc=0 npg=0 nmax=0

    if [[ -f "$csv_mg" && -s "$csv_mg" ]]; then
        mapfile -t arr_mg < "$csv_mg"; nmg=${#arr_mg[@]}
    fi
    if [[ -f "$csv_mc" && -s "$csv_mc" ]]; then
        mapfile -t arr_mc < "$csv_mc"; nmc=${#arr_mc[@]}
    fi
    if [[ -f "$csv_pc" && -s "$csv_pc" ]]; then
        mapfile -t arr_pc < "$csv_pc"; npc=${#arr_pc[@]}
    fi
    if [[ -f "$csv_pg" && -s "$csv_pg" ]]; then
        mapfile -t arr_pg < "$csv_pg"; npg=${#arr_pg[@]}
    fi

    nmax=$nmg
    (( nmc > nmax )) && nmax=$nmc
    (( npc > nmax )) && nmax=$npc
    (( npg > nmax )) && nmax=$npg

    echo "| Epoch | MB-GPU Loss | MB-CPU Loss | PT-CPU Loss | PT-GPU Loss | MB-GPU Test% | MB-CPU Test% | PT-CPU Test% | PT-GPU Test% | MB-GPU Time | MB-CPU Time | PT-CPU Time | PT-GPU Time |"
    echo "|------:|------------:|------------:|------------:|------------:|-------------:|-------------:|-------------:|-------------:|------------:|------------:|------------:|------------:|"

    for i in $(seq 0 $((nmax - 1))); do
        local e="—" mgl="—" mcl="—" pcl="—" pgl="—"
        local mgt="—" mct="—" pct="—" pgt="—"
        local mgs="—" mcs="—" pcs="—" pgs="—"

        if [[ $i -lt $nmg ]]; then
            IFS=',' read -r e mgl _ mct2 mgt mgs <<< "${arr_mg[$i]}"
        fi
        if [[ $i -lt $nmc ]]; then
            local emc
            IFS=',' read -r emc mcl _ mct pgt2 mcs <<< "${arr_mc[$i]}"
            (( i >= nmg )) && e="$emc"
        fi
        if [[ $i -lt $npc ]]; then
            local epc
            IFS=',' read -r epc pcl _ pct2 pct pcs <<< "${arr_pc[$i]}"
            (( i >= nmg && i >= nmc )) && e="$epc"
        fi
        if [[ $i -lt $npg ]]; then
            local epg
            IFS=',' read -r epg pgl _ pgt2 pgt pgs <<< "${arr_pg[$i]}"
            (( i >= nmg && i >= nmc && i >= npc )) && e="$epg"
        fi

        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
            "$e" "$mgl" "$mcl" "$pcl" "$pgl" "$mgt" "$mct" "$pct" "$pgt" "$mgs" "$mcs" "$pcs" "$pgs"
    done
}

# Detect SVG files
SVG_FILES="$(find "$REPORTS_DIR" -name "*.svg" 2>/dev/null | sort || true)"

cat > "$SUMMARY" << SUMMARY_EOF
# MNIST Training Comparison — ${DATE_STR}

## Setup

| Parameter   | Value         |
|-------------|---------------|
| Epochs      | ${EPOCHS}     |
| Batch size  | ${BATCH_SIZE} |
| GPU         | ${GPU_MODEL}  |
| Run ID      | ${RUN_ID}     |

## Final Epoch Results

| Backend      | Loss          | Train Acc     | Test Acc      | Epoch Time (s) |
|------------- |-------------:|-------------:|-------------:|---------------:|
| MoonBit GPU  | ${mg_loss}    | ${mg_train_acc}% | ${mg_test_acc}%  | ${mg_epoch_s}    |
| MoonBit CPU  | ${mc_loss}    | ${mc_train_acc}% | ${mc_test_acc}%  | ${mc_epoch_s}    |
| PyTorch CPU  | ${pc_loss}    | ${pc_train_acc}% | ${pc_test_acc}%  | ${pc_epoch_s}    |
| PyTorch GPU  | ${pg_loss}    | ${pg_train_acc}% | ${pg_test_acc}%  | ${pg_epoch_s}    |

## Per-Epoch Comparison

$(per_epoch_table "$CSV_MG" "$CSV_MC" "$CSV_PC" "$CSV_PG")

## Artifacts

### Logs
- MoonBit GPU log: [\`moonbit_gpu.log\`](moonbit_gpu.log)
- MoonBit CPU log: [\`moonbit_cpu.log\`](moonbit_cpu.log)
- PyTorch CPU log: [\`pytorch_cpu.log\`](pytorch_cpu.log)
- PyTorch GPU log: [\`pytorch_gpu.log\`](pytorch_gpu.log)

### CSV Data
- MoonBit GPU CSV: [\`moonbit_gpu.csv\`](moonbit_gpu.csv)
- MoonBit CPU CSV: [\`moonbit_cpu.csv\`](moonbit_cpu.csv)
- PyTorch CPU CSV: [\`pytorch_cpu.csv\`](pytorch_cpu.csv)
- PyTorch GPU CSV: [\`pytorch_gpu.csv\`](pytorch_gpu.csv)

### Plots
SUMMARY_EOF

if [[ -n "$SVG_FILES" ]]; then
    while IFS= read -r svg; do
        local_name="$(basename "$svg")"
        echo "- ![${local_name}](${local_name})" >> "$SUMMARY"
    done <<< "$SVG_FILES"
else
    echo "- _(no SVG plots generated)_" >> "$SUMMARY"
fi

echo ""
echo -e "${GREEN}Summary written to $SUMMARY${NC}"

# ── Convergence summary ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Convergence Summary (4 Backends)${NC}"
echo -e "${YELLOW}══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  MoonBit GPU : loss=${mg_loss}  test_acc=${mg_test_acc}%  (${mg_epoch_s}s/epoch)"
echo "  MoonBit CPU : loss=${mc_loss}  test_acc=${mc_test_acc}%  (${mc_epoch_s}s/epoch)"
echo "  PyTorch CPU : loss=${pc_loss}  test_acc=${pc_test_acc}%  (${pc_epoch_s}s/epoch)"
echo "  PyTorch GPU : loss=${pg_loss}  test_acc=${pg_test_acc}%  (${pg_epoch_s}s/epoch)"
echo ""

TOTAL_END=$(date +%s)
echo -e "${GREEN}All done. Total wall time: $(elapsed_str $((TOTAL_END - TOTAL_START)))${NC}"
echo -e "  Reports in: $REPORTS_DIR/"
