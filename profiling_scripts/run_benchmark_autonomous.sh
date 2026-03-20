#!/bin/bash
# =============================================================================
# Autonomous Benchmark Runner v2
#
# Self-contained script that runs on each machine independently.
# Survives SSH disconnection (use with nohup).
#
# Strategy:
#   - WARMUP warmup runs (discarded)
#   - REPS_CLEAN measured runs WITHOUT nsys (real performance times)
#   - REPS_NSYS measured runs WITH nsys (for NVTX phase breakdown only)
#   - After each nsys run: extract NVTX to CSV, delete .sqlite
#
# Logging to Elastic:
#   The profiler container sends logs to Logstash (10.0.210.30:5000)
#   with GPUPHOT_ENVIRONMENT=profiler for easy filtering.
#
# Usage (on the remote machine):
#   nohup bash profiling_scripts/run_benchmark_autonomous.sh > benchmark.log 2>&1 &
# =============================================================================

set -uo pipefail

# ---- Configuration (override via environment) ----
WARMUP=${WARMUP:-2}
REPS_CLEAN=${REPS_CLEAN:-10}    # measured runs WITHOUT nsys (real times)
REPS_NSYS=${REPS_NSYS:-2}       # measured runs WITH nsys (phase breakdown)
IMAGES_DIR="${IMAGES_DIR:-/mnt/vast/samueltest/gpuphot/profiling_test_images}"
CONFIGS_DIR="${CONFIGS_DIR:-/mnt/vast/samueltest/gpuphot/cameras_config}"
ASTRO_CACHE="${ASTRO_CACHE:-/mnt/vast/cache/astrometry}"
PROFILERS="${PROFILERS:-312,38}"
DOCKER_EXTRA="${DOCKER_EXTRA:-}"
MACHINE_NAME="${MACHINE_NAME:-$(hostname)}"
GPU_DEVICE="${GPU_DEVICE:-0}"

# Logstash / Elastic config
LOGSTASH_HOST="${LOGSTASH_HOST:-10.0.210.30}"
LOGSTASH_PORT="${LOGSTASH_PORT:-5000}"
GPUPHOT_ENVIRONMENT="${GPUPHOT_ENVIRONMENT:-profiler}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_DIR}/benchmarks/benchmark_results/run_${TIMESTAMP}"
CSV_FILE="${RESULTS_DIR}/benchmark_${MACHINE_NAME}.csv"

mkdir -p "$RESULTS_DIR"

# ---- Images ordered by size (smallest first) ----
IMAGES=(
    "TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits|iKon936|4.2MP|iKon936_SDSSg"
    "TTT3_iKon936-1_2025-09-15-05-31-52-256207_C2025A6_Lum.fits|iKon936|4.2MP|iKon936_Lum"
    "TTT3_QHY600-3_2025-12-01-23-02-47-487643_C2025R2_Lum.fits|QHY600M|6.8MP|QHY600-3_Lum"
    "TTT2_QHY600-4_2026-02-14-23-46-19-497908_WASP-43-b_SDSSg.fits|QHY600M|15.3MP|QHY600-4_SDSSg"
    "TTT2_QHY600-4_2026-02-14-00-13-51-310869_NGC2903_Ha.fits|QHY600M|15.3MP|QHY600-4_Ha"
    "TTT1_QHY411-1_2026-03-09-21-22-48-661122_2012QD8_Lum.fits|QHY411MERIS|37.8MP|QHY411-1_Lum_bin2"
    "TTT1_QHY411-1_2026-02-14-03-52-12-471080_GaiaDR33534005919872722560_SDSSi.fits|QHY411MERIS|37.8MP|QHY411-1_SDSSi_bin2"
    "TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits|QHY411MERIS|151.2MP|QHY411-1_Lum_full"
    "TST_QHY411-3_2026-02-14-06-35-10-547282_24P_Lum.fits|QHY411MERIS|151.2MP|QHY411-3_Lum_full"
    "TST_QHY411-3_2026-02-14-23-09-41-463160_M81_SDSSr.fits|QHY411MERIS|151.2MP|QHY411-3_SDSSr_full"
)

# ---- Profiler image map ----
declare -A PROFILER_IMAGES
PROFILER_IMAGES[312]="gpuphotfinal-profiler"
PROFILER_IMAGES[38]="gpuphotfinal-profiler_38"

declare -A PROFILER_LABELS
PROFILER_LABELS[312]="py312"
PROFILER_LABELS[38]="py38"

# ---- CSV header ----
write_csv_header() {
    if [ ! -f "$CSV_FILE" ]; then
        echo "machine,profiler,image_label,image_file,instrument,megapixels,rep,mode,status,time_s,objects,transients,gpu_name,gpu_uuid,gpu_driver,gpu_power_max_w,gpu_vram_before_mb,gpu_vram_after_mb,gpu_vram_total_mb,gpu_temp_before_c,gpu_temp_after_c,cupy_version,python_version,nvtx_csv" >> "$CSV_FILE"
    fi
}

# ---- Parse output into CSV row ----
parse_output_to_csv() {
    local profiler_label=$1 image_label=$2 image_file=$3 instrument=$4 mp=$5 rep=$6 mode=$7 output=$8 nvtx_csv=$9

    local status=$(echo "$output" | grep "^Status:" | awk '{print $2}')
    local time_s=$(echo "$output" | grep "^Time:" | sed 's/.*(\([0-9.]*\)s)/\1/')
    local objects=$(echo "$output" | grep "^Objects:" | awk '{print $2}')
    local transients=$(echo "$output" | grep "^Transients:" | awk '{print $2}')
    local gpu_name=$(echo "$output" | grep "^GPU:" | head -1 | sed 's/GPU: *//')
    local gpu_uuid=$(echo "$output" | grep "^GPU UUID:" | sed 's/GPU UUID: *//')
    local gpu_driver=$(echo "$output" | grep "^GPU driver:" | awk '{print $3}')
    local gpu_power=$(echo "$output" | grep "^GPU power:" | sed 's/.*max=\([0-9.]*\).*/\1/')
    local gpu_vram_before=$(echo "$output" | grep "^GPU VRAM:" | head -1 | awk '{print $2}')
    local gpu_vram_after=$(echo "$output" | grep "^GPU VRAM after:" | awk '{print $4}')
    local gpu_vram_total=$(echo "$output" | grep "^GPU VRAM:" | head -1 | awk '{print $4}')
    local gpu_temp_before=$(echo "$output" | grep "^GPU temp:" | head -1 | awk '{print $3}')
    local gpu_temp_after=$(echo "$output" | grep "^GPU temp after:" | awk '{print $4}')
    local cupy_ver=$(echo "$output" | grep "^CuPy:" | awk '{print $2}')
    local python_ver=$(echo "$output" | grep "^Python:" | awk '{print $2}')

    echo "${MACHINE_NAME},${profiler_label},${image_label},${image_file},${instrument},${mp},${rep},${mode},${status:-UNKNOWN},${time_s:-0},${objects:-0},${transients:-0},${gpu_name},${gpu_uuid},${gpu_driver},${gpu_power},${gpu_vram_before},${gpu_vram_after},${gpu_vram_total},${gpu_temp_before},${gpu_temp_after},${cupy_ver},${python_ver},${nvtx_csv}" >> "$CSV_FILE"
}

# ---- Extract NVTX from sqlite and delete it ----
extract_and_cleanup() {
    local sqlite_file=$1 nvtx_csv=$2

    if [ -f "$sqlite_file" ]; then
        python3 "$SCRIPT_DIR/extract_benchmark_csv.py" \
            "$(dirname "$sqlite_file")" \
            --machine "$MACHINE_NAME" \
            --profiler "$current_profiler_label" \
            --output "$nvtx_csv" 2>/dev/null

        rm -f "$sqlite_file"
        local nsys_rep="${sqlite_file%.sqlite}.nsys-rep"
        if [ -f "$nsys_rep" ]; then
            local size_mb=$(du -m "$nsys_rep" | cut -f1)
            if [ "$size_mb" -gt 100 ]; then
                rm -f "$nsys_rep"
                echo "    Deleted ${nsys_rep} (${size_mb}MB)"
            fi
        fi
    fi
}

# ---- Main ----
echo "======================================================================"
echo " Autonomous Benchmark v2 — ${MACHINE_NAME}"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Warmup: $WARMUP, Clean reps: $REPS_CLEAN, Nsys reps: $REPS_NSYS"
echo " Logstash: ${LOGSTASH_HOST}:${LOGSTASH_PORT} (env=${GPUPHOT_ENVIRONMENT})"
echo " Images dir: $IMAGES_DIR"
echo " Results: $RESULTS_DIR"
echo " CSV: $CSV_FILE"
echo "======================================================================"

write_csv_header

IFS=',' read -ra PROF_LIST <<< "$PROFILERS"

for prof_key in "${PROF_LIST[@]}"; do
    profiler_image="${PROFILER_IMAGES[$prof_key]}"
    current_profiler_label="${PROFILER_LABELS[$prof_key]}"

    echo ""
    echo "============================================================"
    echo " Profiler: ${current_profiler_label} (${profiler_image})"
    echo "============================================================"

    if ! docker image inspect "$profiler_image" >/dev/null 2>&1; then
        echo "  SKIP: Docker image ${profiler_image} not found"
        continue
    fi

    NSYS_DIR="${RESULTS_DIR}/nsys_${current_profiler_label}"
    mkdir -p "$NSYS_DIR"

    # Docker command with logging env vars
    DOCKER_CMD="docker run --rm --gpus \"device=${GPU_DEVICE}\" \
        --privileged --cap-add SYS_ADMIN --cap-add SYS_PTRACE \
        -v ${IMAGES_DIR}:/data/images:z \
        -v ${CONFIGS_DIR}:/data/instrument_configs:z \
        -v ${ASTRO_CACHE}:/data/astrometry_cache:z \
        -v ${NSYS_DIR}:/app/profiling_results:z \
        -v ${SCRIPT_DIR}:/app/profiling_scripts:z \
        -e INSTRUMENT_CONFIG_BASE_PATH=/data/instrument_configs \
        -e IMAGE_BASE_PATH=/data/images \
        -e GPU_ID=0 \
        -e LOGSTASH_LOGGING=True \
        -e LOGSTASH_HOST=${LOGSTASH_HOST} \
        -e LOGSTASH_PORT=${LOGSTASH_PORT} \
        -e GPUPHOT_ENVIRONMENT=${GPUPHOT_ENVIRONMENT} \
        -e GPUPHOT_DEBUG=True \
        -e GPUPHOT_LOG_LEVEL=DEBUG \
        -e POSTGRES_HOST=nostore \
        ${DOCKER_EXTRA} \
        ${profiler_image}"

    for img_entry in "${IMAGES[@]}"; do
        IFS='|' read -r img_file instrument mp img_label <<< "$img_entry"

        if [ ! -f "${IMAGES_DIR}/${img_file}" ]; then
            echo "  SKIP: ${img_file} not found"
            continue
        fi

        echo ""
        echo "  --- ${img_label} (${mp}) ---"

        # ==== WARMUP ====
        for w in $(seq 1 $WARMUP); do
            echo -n "    warmup ${w}: "
            output=$(eval $DOCKER_CMD python3 /app/profiling_scripts/profile_process_image.py \
                "$img_file" "$instrument" 2>&1)
            status=$(echo "$output" | grep "^Status:" | awk '{print $2}')
            time_str=$(echo "$output" | grep "^Time:" | head -1)
            if [ "$status" = "OK" ]; then
                echo "$time_str"
            else
                error=$(echo "$output" | grep "^Error:" | head -1)
                echo "FAILED — $error"
                if [ "$w" -eq 1 ]; then
                    echo "    Skipping image (warmup failed)"
                    continue 2
                fi
            fi
        done

        # ==== CLEAN RUNS (no nsys — real performance) ====
        echo "    --- Clean runs (${REPS_CLEAN} reps, no nsys) ---"
        for rep in $(seq 1 $REPS_CLEAN); do
            echo -n "    clean ${rep}/${REPS_CLEAN}: "
            output=$(eval $DOCKER_CMD python3 /app/profiling_scripts/profile_process_image.py \
                "$img_file" "$instrument" 2>&1)
            status=$(echo "$output" | grep "^Status:" | awk '{print $2}')
            time_s=$(echo "$output" | grep "^Time:" | sed 's/.*(\([0-9.]*\)s)/\1/')
            echo "${status} ${time_s}s"
            parse_output_to_csv "$current_profiler_label" "$img_label" "$img_file" "$instrument" "$mp" "$rep" "clean" "$output" ""
        done

        # ==== NSYS RUNS (with nsys — for NVTX phase breakdown) ====
        echo "    --- Nsys runs (${REPS_NSYS} reps, with profiling) ---"
        for rep in $(seq 1 $REPS_NSYS); do
            echo -n "    nsys ${rep}/${REPS_NSYS}: "
            output=$(eval $DOCKER_CMD bash /app/profiling_scripts/run_benchmark_profiling.sh \
                "$img_file" "$instrument" 1 2>&1)
            status=$(echo "$output" | grep "^Status:" | awk '{print $2}')
            time_s=$(echo "$output" | grep "^Time:" | sed 's/.*(\([0-9.]*\)s)/\1/')
            echo "${status} ${time_s}s"

            # Extract NVTX and cleanup
            latest_sqlite=$(ls -t ${NSYS_DIR}/*.sqlite 2>/dev/null | head -1)
            nvtx_csv="${NSYS_DIR}/nvtx_${current_profiler_label}_${img_label}_nsys${rep}.csv"
            if [ -n "$latest_sqlite" ]; then
                extract_and_cleanup "$latest_sqlite" "$nvtx_csv"
            fi

            parse_output_to_csv "$current_profiler_label" "$img_label" "$img_file" "$instrument" "$mp" "$rep" "nsys" "$output" "$nvtx_csv"
        done

        echo "    Done: ${img_label} ($(wc -l < "$CSV_FILE") csv rows)"
        echo "    Disk: $(df -h / | tail -1 | awk '{print $4}') free"
    done
done

echo ""
echo "======================================================================"
echo " Benchmark complete — ${MACHINE_NAME}"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSV: $CSV_FILE ($(wc -l < "$CSV_FILE") rows)"
echo " Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "======================================================================"
