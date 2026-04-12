#!/bin/bash
# =============================================================================
# Run cuML crossover benchmark on all x86 GPUs and merge results
#
# This script runs benchmark_cuml_crossover.py inside the py312 profiler
# container on each machine (Jetson excluded — no cuML on ARM), collects
# per-machine CSVs, and merges them into a single combined CSV compatible
# with generate_manuscript_figures.py (fig 5).
#
# Usage:
#   ./dev/run_cuml_crossover_all.sh [OPTIONS]
#
# Options:
#   --machines LIST    Comma-separated: lenovo,hp3,azken,ttt_server,ttt1,local
#                      (default: all x86 machines)
#   --logspace N MIN MAX  Log-spaced sizes (default: 25 100 200000)
#   --repeats N        Measured repetitions (default: 10)
#   --no-auto-refine   Disable auto-refine (not recommended)
#   --sequential       Run machines sequentially instead of in parallel
#   --merge-only       Skip benchmark runs, only merge existing per-machine CSVs
#
# Output:
#   benchmarks/results_collected/cuml_crossover_<label>_YYYYMMDD.csv  (per machine)
#   benchmarks/results_collected/cuml_crossover_synthetic_all_gpus_YYYYMMDD.csv (combined)
#
# After running, update CUML_CSV in benchmarks/generate_manuscript_figures.py:
#   CUML_CSV = os.path.join(DATA_DIR, 'cuml_crossover_synthetic_all_gpus_YYYYMMDD.csv')
#
# Examples:
#   # Full run on all machines (parallel)
#   ./dev/run_cuml_crossover_all.sh
#
#   # Only local GPU (RTX 3050 Ti)
#   ./dev/run_cuml_crossover_all.sh --machines local
#
#   # Coarser sweep, faster (for testing)
#   ./dev/run_cuml_crossover_all.sh --logspace 15 100 200000 --repeats 5
#
#   # Merge existing per-machine CSVs without re-running benchmarks
#   ./dev/run_cuml_crossover_all.sh --merge-only
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATE=$(date +%Y%m%d)
MACHINE_FILTER=""
LOGSPACE_N=25
LOGSPACE_MIN=100
LOGSPACE_MAX=200000
REPEATS=10
AUTO_REFINE=true
SEQUENTIAL=false
MERGE_ONLY=false
DATA_DIR="benchmarks/results_collected"
COMBINED_CSV="${DATA_DIR}/cuml_crossover_synthetic_all_gpus_${DATE}.csv"

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --machines)      MACHINE_FILTER="$2"; shift 2 ;;
        --logspace)      LOGSPACE_N="$2"; LOGSPACE_MIN="$3"; LOGSPACE_MAX="$4"; shift 4 ;;
        --repeats)       REPEATS="$2"; shift 2 ;;
        --no-auto-refine) AUTO_REFINE=false; shift ;;
        --sequential)    SEQUENTIAL=true; shift ;;
        --merge-only)    MERGE_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Machine definitions (x86 only — jetson excluded: no cuML on ARM) ──────────
# Format: "host|container|label"
MACHINES=(
    "lenovo_tttserver|gpuphot-profiler-1|lenovo_A100"
    "hp3|gpuphot-profiler-1|hp3_L40S"
    "azken|gpuphot-profiler-1|azken_H100"
    "ttt_server|gpuphot-profiler-1|ttt_server_RTX3090"
    "ttt1|gpuphot-profiler-1|ttt1_RTX3060"
    "local|gpuphotfinal-profiler-1|local_RTX3050Ti"
)

# ── Build benchmark command args ───────────────────────────────────────────────
BENCH_ARGS="--logspace ${LOGSPACE_N} ${LOGSPACE_MIN} ${LOGSPACE_MAX} --repeats ${REPEATS}"
$AUTO_REFINE && BENCH_ARGS="${BENCH_ARGS} --auto-refine"

echo "======================================================================"
echo " GPUPhot cuML Crossover Benchmark — All GPUs"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Args: ${BENCH_ARGS}"
echo " Combined output: ${COMBINED_CSV}"
echo "======================================================================"

# ── Function: run one machine ──────────────────────────────────────────────────
run_machine() {
    local host="$1" container="$2" label="$3"
    local per_machine_csv="${DATA_DIR}/cuml_crossover_${label}_${DATE}.csv"

    echo ""
    echo ">>> $label ($host)"

    # Check connectivity
    if [ "$host" != "local" ]; then
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" &>/dev/null; then
            echo "  SKIP: $host is offline or unreachable"
            return 1
        fi
    fi

    # Check container is running
    if [ "$host" = "local" ]; then
        if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            echo "  SKIP: container $container not running on local"
            return 1
        fi
    else
        if ! ssh "$host" "docker ps --format '{{.Names}}' | grep -q '^${container}$'" 2>/dev/null; then
            echo "  SKIP: container $container not running on $host"
            return 1
        fi
    fi

    # Verify cuML is available in container
    if [ "$host" = "local" ]; then
        if ! docker exec "$container" python3 -c "from cuml.neighbors import NearestNeighbors" 2>/dev/null; then
            echo "  SKIP: cuML not available in $container"
            return 1
        fi
    else
        if ! ssh "$host" "docker exec $container python3 -c 'from cuml.neighbors import NearestNeighbors'" 2>/dev/null; then
            echo "  SKIP: cuML not available in $container on $host"
            return 1
        fi
    fi

    echo "  Running benchmark (args: ${BENCH_ARGS}) ..."
    echo "  Progress → /tmp/cuml_crossover_${label}.log"
    # Truncate log file before writing (avoid binary contamination from previous runs)
    > "/tmp/cuml_crossover_${label}.log"

    # The benchmarks/ dir is NOT mounted inside profiler containers.
    # Copy the script into /tmp of the container and run from there.
    # Avoids docker exec -i stdin issues when running in background subshells.
    local SCRIPT_PATH
    SCRIPT_PATH="$(cd "$(dirname "$0")/.." && pwd)/benchmarks/benchmark_cuml_crossover.py"

    if [ "$host" = "local" ]; then
        docker cp "${SCRIPT_PATH}" "${container}:/tmp/benchmark_cuml_crossover.py" 2>/dev/null
        docker exec "$container" \
            python3 /tmp/benchmark_cuml_crossover.py ${BENCH_ARGS} \
            > "${per_machine_csv}" \
            2>/tmp/cuml_crossover_${label}.log
    else
        # Copy script to remote host /tmp, then into container, then run
        scp -q "${SCRIPT_PATH}" "${host}:/tmp/benchmark_cuml_crossover.py"
        ssh -o ServerAliveInterval=60 "$host" \
            "docker cp /tmp/benchmark_cuml_crossover.py ${container}:/tmp/benchmark_cuml_crossover.py && \
             docker exec ${container} python3 /tmp/benchmark_cuml_crossover.py ${BENCH_ARGS}" \
            > "${per_machine_csv}" \
            2>/tmp/cuml_crossover_${label}.log
    fi

    local nrows
    nrows=$(tail -n +2 "${per_machine_csv}" | wc -l)
    echo "  Saved: ${per_machine_csv} (${nrows} data rows)"
    echo "  Recommendation:"
    grep "GPUPHOT_CUML" /tmp/cuml_crossover_${label}.log | sed 's/^/#     /'
}

# ── Run benchmarks (skip if --merge-only) ─────────────────────────────────────
if ! $MERGE_ONLY; then
    PIDS=()
    for machine_def in "${MACHINES[@]}"; do
        IFS='|' read -r host container label <<< "$machine_def"

        # Apply --machines filter
        if [ -n "$MACHINE_FILTER" ]; then
            echo ",$MACHINE_FILTER," | grep -q ",$host," || continue
        fi

        if $SEQUENTIAL; then
            run_machine "$host" "$container" "$label"
        else
            run_machine "$host" "$container" "$label" &
            PIDS+=($!)
        fi
    done

    # Wait for parallel jobs
    if ! $SEQUENTIAL; then
        for pid in "${PIDS[@]}"; do
            wait "$pid" || true
        done
    fi
fi

# ── Merge per-machine CSVs into combined file ──────────────────────────────────
echo ""
echo "======================================================================"
echo " Merging per-machine CSVs → ${COMBINED_CSV}"
echo "======================================================================"

# Write combined CSV header (compatible with generate_manuscript_figures.py)
echo "gpu_name,machine,N,cpu_ms,gpu_e2e_ms,speedup,winner,cpu_q25,cpu_q75,gpu_q25,gpu_q75" > "${COMBINED_CSV}"

TOTAL_ROWS=0
for machine_def in "${MACHINES[@]}"; do
    IFS='|' read -r host container label <<< "$machine_def"

    # Find the per-machine CSV (today's or most recent if merge-only)
    per_machine_csv="${DATA_DIR}/cuml_crossover_${label}_${DATE}.csv"
    if [ ! -f "${per_machine_csv}" ] && $MERGE_ONLY; then
        # Fall back to most recent CSV for this label
        per_machine_csv=$(ls -t "${DATA_DIR}/cuml_crossover_${label}_"*.csv 2>/dev/null | head -1)
    fi

    if [ ! -f "${per_machine_csv}" ]; then
        echo "  MISSING: ${per_machine_csv} — skipped"
        continue
    fi

    # Get GPU name from the first data row's stderr log, or from nvidia-smi
    # Strip "NVIDIA " prefix for consistency with 2026-03-28 CSV format used by figure5()
    GPU_NAME=""
    if [ -f "/tmp/cuml_crossover_${label}.log" ]; then
        GPU_NAME=$(grep "^# GPU:" /tmp/cuml_crossover_${label}.log | sed 's/# GPU: //' | sed 's/^NVIDIA //' | tr -d '\r')
    fi
    if [ -z "$GPU_NAME" ]; then
        # Try to get from container (best-effort)
        if [ "$host" = "local" ]; then
            GPU_NAME=$(docker exec "${container}" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^NVIDIA //' | tr -d '\r' || echo "unknown")
        else
            GPU_NAME=$(ssh -o ConnectTimeout=5 "$host" "docker exec ${container} nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null | sed 's/^NVIDIA //' | tr -d '\r' || echo "unknown")
        fi
    fi

    # Append data rows with gpu_name and machine columns
    # Input:  N,cpu_ms,gpu_ms,speedup_e2e,winner
    # Output: gpu_name,machine,N,cpu_ms,gpu_e2e_ms,speedup,winner
    nrows=$(tail -n +2 "${per_machine_csv}" | wc -l)
    tail -n +2 "${per_machine_csv}" | \
        awk -v gpu="${GPU_NAME}" -v machine="${host}" -F',' \
        'BEGIN{OFS=","} NF>=5 {print gpu, machine, $1, $2, $3, $4, $5, ($6!="" ? $6 : ""), ($7!="" ? $7 : ""), ($8!="" ? $8 : ""), ($9!="" ? $9 : "")}' \
        >> "${COMBINED_CSV}"

    TOTAL_ROWS=$((TOTAL_ROWS + nrows))
    echo "  ${label}: ${nrows} rows  (GPU: ${GPU_NAME})"
done

echo ""
echo "Combined CSV written: ${COMBINED_CSV}"
echo "Total rows: ${TOTAL_ROWS}"
echo ""
echo "======================================================================"
echo " Next step: update CUML_CSV constant in generate_manuscript_figures.py"
echo "======================================================================"
echo ""
echo "  Line ~68:"
echo "  CUML_CSV = os.path.join(DATA_DIR, 'cuml_crossover_synthetic_all_gpus_${DATE}.csv')"
echo ""
echo " Then regenerate figure 5:"
echo "  python3 benchmarks/generate_manuscript_figures.py"
echo "  cd GPUPHOT_manuscript && pdflatex main.tex && pdflatex main.tex"
echo "======================================================================"
