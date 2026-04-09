#!/bin/bash
# =============================================================================
# Master Benchmark Orchestrator
#
# Runs the full benchmark suite on ALL configured machines, collects results,
# and merges them into a single CSV for analysis.
#
# Usage:
#   ./profiling_scripts/run_all_benchmarks.sh [--warmup N] [--reps N] [--test]
#
# Options:
#   --warmup N    Number of warmup iterations (default: 2)
#   --reps N      Number of measured iterations (default: 10)
#   --test        Quick test mode: warmup=1, reps=1, 1 small image only
#   --max-mp N    Max megapixels to process (0=all, default: 0)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE="${PROJECT_DIR}/benchmarks/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_BASE}/run_${TIMESTAMP}"

# Defaults
WARMUP=2
REPS=10
MAX_MP=0
TEST_MODE=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --warmup) WARMUP="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --test) TEST_MODE=true; WARMUP=1; REPS=1; MAX_MP=5; shift ;;
        --max-mp) MAX_MP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "======================================================================"
echo " GPUPhot Master Benchmark Orchestrator"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Warmup: $WARMUP, Repetitions: $REPS, Max MP: $MAX_MP"
echo " Results: $RESULTS_DIR"
echo "======================================================================"

# ---- Machine definitions ----
# Format: name|ssh_host|profilers|images_dir|configs_dir|astro_cache|docker_extra_args
# profilers: comma-separated list (312,38)
# docker_extra_args: extra docker run flags (e.g. for Jetson)

MACHINES=(
    "lenovo|lenovo_tttserver|312,38|/mnt/vast/samueltest/gpuphot/profiling_test_images|/mnt/vast/samueltest/gpuphot/cameras_config|/mnt/vast/cache/astrometry|"
    "hp3|hp3|312,38|/mnt/vast/samueltest/gpuphot/profiling_test_images|/mnt/vast/samueltest/gpuphot/cameras_config|/mnt/vast/cache/astrometry|"
    "azken|azken|312,38|/mnt/vast/samueltest/gpuphot/profiling_test_images|/mnt/vast/samueltest/gpuphot/cameras_config|/mnt/vast/cache/astrometry|"
    "ttt_server|ttt_server|312,38|/mnt/vast/samueltest/gpuphot/profiling_test_images|/mnt/vast/samueltest/gpuphot/cameras_config|/mnt/vast/cache/astrometry|"
    "ttt1|ttt1|312,38|~/GPUPhot/benchmarks/benchmark_images|~/DTO/ttt/tasks/cameras_config|~/GPUPhot/tests/astronomy_cache|"
    "local|local|312,38|${PROJECT_DIR}/benchmarks/benchmark_images|/mnt/vast/samueltest/gpuphot/cameras_config|${PROJECT_DIR}/tests/astronomy_cache|"
    "jetson_local|jetson_local|312,38|~/GPUPhot/benchmarks/benchmark_images|~/GPUPhot/gpuphot/instrument_configs|~/GPUPhot/tests/astronomy_cache|--runtime nvidia --privileged --network host -v /usr/local/cuda:/usr/local/cuda:ro -v /usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/nvidia:ro -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia"
    "jetson_orin|jetson_orin|38|~/GPUPhot/benchmarks/benchmark_images|~/GPUPhot/gpuphot/instrument_configs|~/GPUPhot/tests/astronomy_cache|--runtime nvidia --privileged --network host -v /usr/local/cuda:/usr/local/cuda:ro -v /usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/nvidia:ro -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia"
)

# ---- Helper: run benchmark on one machine, one profiler ----
run_machine_profiler() {
    local name=$1 ssh_host=$2 profiler=$3 images_dir=$4 configs_dir=$5 astro_cache=$6 docker_extra=$7
    local out_dir="${RESULTS_DIR}/${name}/py${profiler}"
    local log_file="${RESULTS_DIR}/${name}_py${profiler}.log"
    local profiler_image="gpuphotfinal-profiler"
    [ "$profiler" = "38" ] && profiler_image="gpuphotfinal-profiler_38"

    mkdir -p "$out_dir"

    echo ""
    echo "  [$name] Profiler py${profiler} (${profiler_image})"

    local mp_arg=""
    [ "$MAX_MP" != "0" ] && mp_arg="--max-megapixels $MAX_MP"

    if [ "$ssh_host" = "local" ]; then
        python3 "${SCRIPT_DIR}/run_benchmark_suite.py" \
            --local --skip-gpu-check \
            --profilers "$profiler" \
            --warmup "$WARMUP" --repetitions "$REPS" \
            --images-dir "$images_dir" \
            --configs-dir "$configs_dir" \
            --astro-cache "$astro_cache" \
            --results-dir "$out_dir" \
            $mp_arg \
            > "$log_file" 2>&1
    else
        ssh -o ServerAliveInterval=30 -o ConnectTimeout=15 "$ssh_host" \
            "cd ~/GPUPhot && python3 profiling_scripts/run_benchmark_suite.py \
                --local --skip-gpu-check \
                --profilers $profiler \
                --warmup $WARMUP --repetitions $REPS \
                --images-dir $images_dir \
                --configs-dir $configs_dir \
                --astro-cache $astro_cache \
                --results-dir ~/GPUPhot/benchmarks/benchmark_results/run_${TIMESTAMP}/py${profiler} \
                $mp_arg" \
            > "$log_file" 2>&1
    fi

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "  [$name] py${profiler}: OK (log: ${log_file})"

        # Retrieve sqlite files from remote
        if [ "$ssh_host" != "local" ]; then
            echo "  [$name] Fetching results..."
            scp -q "${ssh_host}:~/GPUPhot/benchmarks/benchmark_results/run_${TIMESTAMP}/py${profiler}/py${profiler}/*.sqlite" "$out_dir/" 2>/dev/null || true
            scp -q "${ssh_host}:~/GPUPhot/benchmarks/benchmark_results/run_${TIMESTAMP}/py${profiler}/py${profiler}/*.nsys-rep" "$out_dir/" 2>/dev/null || true
            scp -q "${ssh_host}:~/GPUPhot/benchmarks/benchmark_results/run_${TIMESTAMP}/py${profiler}/benchmark_run_log.json" "$out_dir/" 2>/dev/null || true
        fi

        # Extract NVTX CSV
        echo "  [$name] Extracting NVTX data..."
        python3 "${SCRIPT_DIR}/extract_benchmark_csv.py" \
            "$out_dir" \
            --machine "$name" \
            --profiler "py${profiler}" \
            --output "${out_dir}/benchmark_nvtx.csv" 2>/dev/null || true
    else
        echo "  [$name] py${profiler}: FAILED (exit=$exit_code, log: ${log_file})"
    fi
}


# ---- Main execution ----
echo ""
echo "----------------------------------------------------------------------"
echo " Phase 1: Running benchmarks on all machines"
echo "----------------------------------------------------------------------"

for machine_def in "${MACHINES[@]}"; do
    IFS='|' read -r name ssh_host profilers images_dir configs_dir astro_cache docker_extra <<< "$machine_def"

    echo ""
    echo "============ $name ============"

    # Check connectivity
    if [ "$ssh_host" != "local" ]; then
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$ssh_host" "echo ok" &>/dev/null; then
            echo "  [$name] SKIP: cannot connect via SSH"
            continue
        fi
    fi

    # Run each profiler sequentially (one at a time to avoid GPU contention)
    IFS=',' read -ra PROF_LIST <<< "$profilers"
    for prof in "${PROF_LIST[@]}"; do
        run_machine_profiler "$name" "$ssh_host" "$prof" "$images_dir" "$configs_dir" "$astro_cache" "$docker_extra"
    done
done

echo ""
echo "----------------------------------------------------------------------"
echo " Phase 2: Merging all results"
echo "----------------------------------------------------------------------"

# Merge all per-machine CSVs into one
MERGED_CSV="${RESULTS_DIR}/benchmark_all_machines.csv"
FIRST=true
for csv_file in $(find "$RESULTS_DIR" -name 'benchmark_nvtx.csv' | sort); do
    if $FIRST; then
        cat "$csv_file" > "$MERGED_CSV"
        FIRST=false
    else
        tail -n +2 "$csv_file" >> "$MERGED_CSV"
    fi
done

if [ -f "$MERGED_CSV" ]; then
    TOTAL_ROWS=$(wc -l < "$MERGED_CSV")
    echo "  Merged CSV: $MERGED_CSV ($((TOTAL_ROWS - 1)) data rows)"
else
    echo "  WARNING: No CSVs to merge"
fi

echo ""
echo "======================================================================"
echo " Benchmark complete."
echo " Results: $RESULTS_DIR"
echo " Merged:  $MERGED_CSV"
echo "======================================================================"
