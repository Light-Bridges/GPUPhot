#!/bin/bash
# =============================================================================
# Run benchmark on ALL machines via docker exec
#
# Usage:
#   ./dev/run_benchmark_all_machines.sh [OPTIONS]
#
# Options:
#   --warmup N      Warmup iterations (default: 2)
#   --reps N        Measured repetitions (default: 10)
#   --nsys N        Nsys profiling reps after clean reps (default: 0)
#   --profilers X   Comma-separated: 312,38 (default: 312,38)
#   --stop-workers  Stop GPU0 workers on production machines before benchmark
#   --images LIST   Comma-separated image indices 1-10 (default: all)
#   --max-mp N      Max megapixels (0=all, default: 0)
#
# Examples:
#   # Full benchmark: 2 warmup + 10 reps + 2 nsys, both profilers
#   ./dev/run_benchmark_all_machines.sh --stop-workers
#
#   # Quick test: 1 warmup + 1 rep, only small images
#   ./dev/run_benchmark_all_machines.sh --warmup 1 --reps 1 --max-mp 20
#
#   # Only py312 with nsys profiling
#   ./dev/run_benchmark_all_machines.sh --profilers 312 --nsys 2
#
#   # Specific images only (1=iKon SDSSg, 2=iKon Lum, 3=QHY600-3, etc.)
#   ./dev/run_benchmark_all_machines.sh --images 1,2,3 --reps 5
# =============================================================================

set -uo pipefail

# Defaults
WARMUP=2
REPS=10
NSYS_REPS=0
PROFILERS="312,38"
STOP_WORKERS=false
MAX_MP=0
IMAGE_FILTER=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BENCHMARK_LOG_DIR:-/tmp/benchmark_${TIMESTAMP}}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --warmup) WARMUP="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --nsys) NSYS_REPS="$2"; shift 2 ;;
        --profilers) PROFILERS="$2"; shift 2 ;;
        --stop-workers) STOP_WORKERS=true; shift ;;
        --max-mp) MAX_MP="$2"; shift 2 ;;
        --images) IMAGE_FILTER="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# All images ordered by size
ALL_IMAGES=(
    "1|4.2|TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits|iKon936-1|iKon936_SDSSg"
    "2|4.2|TTT3_iKon936-1_2025-09-15-05-31-52-256207_C2025A6_Lum.fits|iKon936-1|iKon936_Lum"
    "3|6.8|TTT3_QHY600-3_2025-12-01-23-02-47-487643_C2025R2_Lum.fits|QHY600-3|QHY600-3_Lum"
    "4|15.3|TTT2_QHY600-4_2026-02-14-23-46-19-497908_WASP-43-b_SDSSg.fits|QHY600-4|QHY600-4_SDSSg"
    "5|15.3|TTT2_QHY600-4_2026-02-14-00-13-51-310869_NGC2903_Ha.fits|QHY600-4|QHY600-4_Ha"
    "6|37.8|TTT1_QHY411-1_2026-03-09-21-22-48-661122_2012QD8_Lum.fits|QHY411-1|QHY411-1_Lum_bin2"
    "7|37.8|TTT1_QHY411-1_2026-02-14-03-52-12-471080_GaiaDR33534005919872722560_SDSSi.fits|QHY411-1|QHY411-1_SDSSi_bin2"
    "8|151.2|TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits|QHY411-1|QHY411-1_Lum_full"
    "9|151.2|TST_QHY411-3_2026-02-14-06-35-10-547282_24P_Lum.fits|QHY411-3|QHY411-3_Lum_full"
    "10|151.2|TST_QHY411-3_2026-02-14-23-09-41-463160_M81_SDSSr.fits|QHY411-3|QHY411-3_SDSSr_full"
)

# Machine definitions: host|profiler_container_312|profiler_container_38|label|flags
# flags: production = has workers to stop; workers field = comma-separated container names
MACHINES=(
    "lenovo_tttserver|gpuphot-profiler-1|gpuphot-profiler_38-1|lenovo_A100|production"
    "hp3|gpuphot-profiler-1|gpuphot-profiler_38-1|hp3_L40S|production"
    "azken|gpuphot-profiler-1|gpuphot-profiler_38-1|azken_H100|production"
    "ttt_server|gpuphot-profiler-1|gpuphot-profiler_38-1|ttt_server_RTX3090|"
    "ttt1|gpuphot-profiler-1|gpuphot-profiler_38-1|ttt1_RTX3060|"
    "local|gpuphotfinal-profiler-1|gpuphotfinal-profiler_38-1|local_RTX3050Ti|"
    "jetson_orin|gpuphot-profiler_jetson_orin-1||jetson_orin_Orin8GB|"
    "jetson_local|gpuphot-profiler_jetson_orin_super-1|gpuphot-profiler_jetson_orin_super_38-1|jetson_local_OrinSuper|"
)

# Workers to stop/start on production machines (per host)
declare -A WORKERS
WORKERS[lenovo_tttserver]="dto-worker0-1 dto-worker-ast-1"
WORKERS[hp3]="dto-worker0-1"
WORKERS[azken]="dto-worker0-1"

# Filter images
IMAGES=()
for img in "${ALL_IMAGES[@]}"; do
    IFS='|' read -r idx mp rest <<< "$img"
    if [ "$MAX_MP" != "0" ]; then
        if (( $(echo "$mp > $MAX_MP" | bc -l) )); then
            continue
        fi
    fi
    if [ -n "$IMAGE_FILTER" ]; then
        echo ",$IMAGE_FILTER," | grep -q ",$idx," || continue
    fi
    IMAGES+=("$img")
done

echo "======================================================================"
echo " GPUPhot Benchmark — All Machines"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Warmup: $WARMUP, Clean reps: $REPS, Nsys reps: $NSYS_REPS"
echo " Profilers: $PROFILERS"
echo " Images: ${#IMAGES[@]} (max ${MAX_MP}MP)"
echo " Logs: $LOG_DIR"
echo "======================================================================"

# Stop workers if requested
if $STOP_WORKERS; then
    echo ""
    echo "Stopping GPU0 workers on production machines..."
    for machine_def in "${MACHINES[@]}"; do
        IFS='|' read -r host c312 c38 label flags <<< "$machine_def"
        if echo "$flags" | grep -q "production"; then
            worker_list="${WORKERS[$host]:-}"
            if [ -n "$worker_list" ]; then
                for w in $worker_list; do
                    ssh -o ConnectTimeout=10 "$host" "docker stop $w 2>/dev/null" &
                done
                echo "  $host: stopping $worker_list"
            fi
        fi
    done
    wait
    echo "  Workers stopped."
fi

# Run benchmark function
run_benchmark() {
    local host=$1 container=$2 label=$3

    for img in "${IMAGES[@]}"; do
        IFS='|' read -r idx mp file inst img_label <<< "$img"
        echo "--- $img_label (${mp}MP) ---"

        # Warmup
        FAILED=0
        for w in $(seq 1 $WARMUP); do
            echo -n "  warmup $w: "
            if [ "$host" = "local" ]; then
                OUT=$(docker exec "$container" python3 /app/profiling_scripts/profile_process_image.py "$file" "$inst" 2>&1)
            else
                OUT=$(ssh -o ServerAliveInterval=30 "$host" "docker exec $container python3 /app/profiling_scripts/profile_process_image.py '$file' '$inst'" 2>&1)
            fi
            STATUS=$(echo "$OUT" | grep "^Status:" | awk '{print $2}')
            TIME=$(echo "$OUT" | grep "^Time:" | sed 's/.*(\([0-9.]*\)s)/\1/')
            [ "$STATUS" = "OK" ] && echo "OK ${TIME}s" || { echo "FAILED"; FAILED=$((FAILED+1)); [ $FAILED -ge 2 ] && echo "  Skip" && continue 2; }
        done

        # Clean reps
        for rep in $(seq 1 $REPS); do
            echo -n "  rep $rep/$REPS: "
            if [ "$host" = "local" ]; then
                OUT=$(docker exec "$container" python3 /app/profiling_scripts/profile_process_image.py "$file" "$inst" 2>&1)
            else
                OUT=$(ssh -o ServerAliveInterval=30 "$host" "docker exec $container python3 /app/profiling_scripts/profile_process_image.py '$file' '$inst'" 2>&1)
            fi
            STATUS=$(echo "$OUT" | grep "^Status:" | awk '{print $2}')
            TIME=$(echo "$OUT" | grep "^Time:" | sed 's/.*(\([0-9.]*\)s)/\1/')
            echo "$STATUS ${TIME}s"
        done

        # Nsys reps (if requested)
        if [ "$NSYS_REPS" -gt 0 ]; then
            for rep in $(seq 1 $NSYS_REPS); do
                echo -n "  nsys $rep/$NSYS_REPS: "
                if [ "$host" = "local" ]; then
                    OUT=$(docker exec "$container" bash /app/profiling_scripts/run_benchmark_profiling.sh "$file" "$inst" 1 2>&1)
                else
                    OUT=$(ssh -o ServerAliveInterval=30 "$host" "docker exec $container bash /app/profiling_scripts/run_benchmark_profiling.sh '$file' '$inst' 1" 2>&1)
                fi
                STATUS=$(echo "$OUT" | grep "^Status:" | awk '{print $2}')
                TIME=$(echo "$OUT" | grep "^Time:" | sed 's/.*(\([0-9.]*\)s)/\1/')
                echo "$STATUS ${TIME}s"
            done
        fi
    done
}

# Launch all machines (parallel between machines, sequential between profilers per machine)
IFS=',' read -ra PROF_LIST <<< "$PROFILERS"

for machine_def in "${MACHINES[@]}"; do
    IFS='|' read -r host c312 c38 label flags <<< "$machine_def"

    # Check connectivity
    if [ "$host" != "local" ]; then
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" &>/dev/null || { echo "SKIP $host (offline)"; continue; }
    fi

    # Each machine runs in background, but profilers run SEQUENTIALLY within
    # each machine to avoid GPU contention (both profilers share GPU 0)
    echo "Launching: $label (profilers: ${PROFILERS})"
    (
        for prof in "${PROF_LIST[@]}"; do
            container="$c312"
            [ "$prof" = "38" ] && container="$c38"
            [ -z "$container" ] && continue

            log_file="${LOG_DIR}/${label}_py${prof}.log"
            echo "=== $label py${prof} ($(date)) ===" > "$log_file"
            run_benchmark "$host" "$container" "${label}_py${prof}" >> "$log_file" 2>&1
            echo "=== DONE $(date) ===" >> "$log_file"
        done
    ) &
done

echo ""
echo "All launched in background."
echo "Monitor:  tail -1 ${LOG_DIR}/*.log"
echo "Status:   grep -l 'DONE' ${LOG_DIR}/*.log | wc -l"
echo ""

# Wait for all background jobs
wait

echo "======================================================================"
echo " ALL BENCHMARKS COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# Restore workers
if $STOP_WORKERS; then
    echo "Restoring workers..."
    for machine_def in "${MACHINES[@]}"; do
        IFS='|' read -r host c312 c38 label flags <<< "$machine_def"
        if echo "$flags" | grep -q "production"; then
            worker_list="${WORKERS[$host]:-}"
            if [ -n "$worker_list" ]; then
                for w in $worker_list; do
                    ssh -o ConnectTimeout=10 "$host" "docker start $w 2>/dev/null" &
                done
                echo "  $host: restored $worker_list"
            fi
        fi
    done
    wait
fi

echo "Logs in: $LOG_DIR"
echo "Collect Elastic data:"
echo "  python benchmarks/collect_times_from_elastic.py --time-from '$(date -u +%Y-%m-%dT%H:%M:%S)' --out benchmarks/es_times_profiler_latest.csv --es-query-file <(echo '{\"size\":10000,\"query\":{\"bool\":{\"filter\":[{\"match_phrase\":{\"extra.environment\":\"profiler\"}},{\"match_phrase\":{\"extra.function_name\":\"process_image\"}},{\"range\":{\"@timestamp\":{\"gte\":\"$(date -u +%Y-%m-%dT%H:%M:%S)\"}}}]}}}')"
