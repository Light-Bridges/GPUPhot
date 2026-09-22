#!/bin/bash
# run_localcat_all_images.sh
#
# Runs profile_localcat.py (local PostgreSQL catalog) sequentially on all
# available benchmark images for both py312 and py38 containers.
#
# Usage:
#   bash run_localcat_all_images.sh [GPU_ID]
#
# GPU_ID defaults to 0. On hp3 verify with:
#   nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
# and pick a GPU with <5 GB used and 0% utilization.
#
# Rules:
#   - py312 runs BEFORE py38 (sequential, same GPU)
#   - Containers must already be Up (docker ps | grep profiler)
#   - Images are on VAST (/data/images inside container)
#   - Results go to /tmp/localcat_<py_ver>_<image_label>.log on the host

set -euo pipefail

GPU_ID="${1:-0}"
LOGDIR="/tmp/localcat_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"

echo "=== LOCAL CATALOG BENCHMARK ===" | tee "$SUMMARY"
echo "GPU_ID: $GPU_ID" | tee -a "$SUMMARY"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# Image -> instrument mapping
declare -A INSTRUMENT=(
  ["TST_QHY411-3"]="QHY411-3"
  ["TTT1_QHY411-1"]="QHY411-1"
  ["TTT2_QHY600-4"]="QHY600-4"
  ["TTT3_QHY600-3"]="QHY600-3"
  ["TTT3_iKon936-1"]="iKon936-1"
)

# Build ordered image list from container (skip gpuphot_processed dir)
IMAGES=$(docker exec gpuphot-profiler-1 ls /data/images/*.fits 2>/dev/null \
         | sed 's|/data/images/||' | sort)

if [ -z "$IMAGES" ]; then
  echo "ERROR: no .fits files in /data/images inside gpuphot-profiler-1" >&2
  exit 1
fi

run_image() {
  local CONTAINER="$1"
  local PY_VER="$2"
  local IMAGE="$3"
  local INSTR="$4"

  local LABEL
  LABEL=$(echo "$IMAGE" | sed 's/\.fits$//')
  local LOGFILE="$LOGDIR/${PY_VER}_${LABEL}.log"

  echo -n "  [$PY_VER] $IMAGE ... " | tee -a "$SUMMARY"
  docker exec -e CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$CONTAINER" \
    python /app/profiling_scripts/profile_localcat.py "$IMAGE" "$INSTR" \
    > "$LOGFILE" 2>&1

  # Extract result
  ELAPSED=$(grep "Time elapsed:" "$LOGFILE" | tail -1 | awk '{print $3}')
  OBJETS=$(grep "objets" "$LOGFILE" | tail -1 | grep -oP "'objets': \K[0-9]+")
  ERROR=$(grep -c "Error durante\|MemoryError\|OutOfMemory\|Traceback" "$LOGFILE" || true)

  if [ "$ERROR" -gt 0 ]; then
    echo "FAILED ($ELAPSED)" | tee -a "$SUMMARY"
  else
    echo "OK — ${ELAPSED} — ${OBJETS} obj" | tee -a "$SUMMARY"
  fi
}

# --- py312 pass ---
echo "=== py312 (gpuphot-profiler-1) ===" | tee -a "$SUMMARY"
for IMAGE in $IMAGES; do
  # Determine instrument from filename prefix
  INSTR=""
  for PREFIX in "${!INSTRUMENT[@]}"; do
    if [[ "$IMAGE" == ${PREFIX}_* ]]; then
      INSTR="${INSTRUMENT[$PREFIX]}"
      break
    fi
  done
  if [ -z "$INSTR" ]; then
    echo "  [py312] $IMAGE ... SKIP (unknown instrument)" | tee -a "$SUMMARY"
    continue
  fi
  run_image "gpuphot-profiler-1" "py312" "$IMAGE" "$INSTR"
done

echo "" | tee -a "$SUMMARY"

# --- py38 pass ---
echo "=== py38 (gpuphot-profiler_38-1) ===" | tee -a "$SUMMARY"
for IMAGE in $IMAGES; do
  INSTR=""
  for PREFIX in "${!INSTRUMENT[@]}"; do
    if [[ "$IMAGE" == ${PREFIX}_* ]]; then
      INSTR="${INSTRUMENT[$PREFIX]}"
      break
    fi
  done
  if [ -z "$INSTR" ]; then
    echo "  [py38] $IMAGE ... SKIP (unknown instrument)" | tee -a "$SUMMARY"
    continue
  fi
  run_image "gpuphot-profiler_38-1" "py38" "$IMAGE" "$INSTR"
done

echo "" | tee -a "$SUMMARY"
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
echo "Logs: $LOGDIR" | tee -a "$SUMMARY"
