#!/bin/bash
# =============================================================================
# Run nsys profiling on profile_process_image.py (direct process_image call).
#
# Usage:
#   run_benchmark_profiling.sh <image_file> <instrument_name> [repetitions]
#
# This is like run_profiling.sh but calls profile_process_image.py instead
# of profile_image_processing.py — no Celery, no database, just the core
# GPU photometry pipeline.
# =============================================================================

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <image_file> <instrument_name> [repetitions]"
    exit 1
fi

IMAGE_FILE=$1
INSTRUMENT_NAME=$2
COUNT=${3:-1}

# Validate count
if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
    echo "Error: repetitions must be a positive integer."
    exit 1
fi

# GPU
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# nsys detection
if [ -d "/opt/nsight-systems-host" ]; then
    NSYS_PATH=$(readlink -f "$(which nsys)")
    if [[ "$NSYS_PATH" == /opt/nsight-systems-host/* ]]; then
        NSYS_CMD="$NSYS_PATH"
    else
        NSYS_CMD="nsys"
    fi
else
    NSYS_CMD="nsys"
fi

OUTPUT_DIR="/app/profiling_results"
mkdir -p "$OUTPUT_DIR"

# GPU metrics (skip on Jetson)
EXTRA_PARAMS="--gpu-metrics-devices=$GPU_ID"
if [ -f /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' </proc/device-tree/model)
    if echo "$MODEL" | grep -qi "jetson"; then
        EXTRA_PARAMS=""
    fi
fi
[ -n "${JETSON_TYPE:-}" ] && EXTRA_PARAMS=""
uname -a | grep -qi "tegra" && EXTRA_PARAMS=""

echo "=============================================="
echo " GPUPhot Benchmark Profiling"
echo " Image:       $IMAGE_FILE"
echo " Instrument:  $INSTRUMENT_NAME"
echo " Repetitions: $COUNT"
echo " GPU:         $GPU_ID"
echo " Output:      $OUTPUT_DIR"
echo "=============================================="

for ((i=1; i<=COUNT; i++)); do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${HOSTNAME}_${INSTRUMENT_NAME}_${TIMESTAMP}_run${i}.nsys-rep"

    echo ""
    echo "[Run $i/$COUNT] Starting nsys profile..."
    $NSYS_CMD profile \
        --trace=cuda,nvtx,osrt \
        --sample=process-tree \
        --stats=true \
        --cuda-memory-usage=true \
        $EXTRA_PARAMS \
        -o "$OUTPUT_FILE" \
        python3 "$SCRIPT_DIR/profile_process_image.py" "$IMAGE_FILE" "$INSTRUMENT_NAME"

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Run $i/$COUNT] OK -> $OUTPUT_FILE"
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "[Run $i/$COUNT] process_image FAILED (nsys data still captured) -> $OUTPUT_FILE"
    else
        echo "[Run $i/$COUNT] ERROR (exit code $EXIT_CODE)"
    fi
done

echo ""
echo "=============================================="
echo " All $COUNT profiling runs completed."
echo "=============================================="
