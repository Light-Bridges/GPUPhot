#!/usr/bin/env bash
# run_cuml_ablation_a100.sh — cuML ablation benchmark for 5 dense-field images on A100 GPU 0
#
# Runs 1 warmup + 3 reps with cuml=no + 3 reps with cuml=yes for each image.
# Captures timing from profile_process_image.py stdout.
# Output: /tmp/cuml_ablation_new_a100_YYYYMMDD_HHMMSS.csv
#
# Usage (on lenovo_tttserver host, NOT inside container):
#   bash /tmp/run_cuml_ablation_a100.sh 2>&1 | tee /tmp/cuml_ablation_a100.log

set -euo pipefail

CONTAINER="gpuphot-profiler-1"
WARMUP=1
REPS=3
# Timeout per rep in seconds (20 min = 1200s; C2025N1+cuml=yes could be very slow)
REP_TIMEOUT=1200

# Output CSV
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT="/tmp/cuml_ablation_new_a100_${TIMESTAMP}.csv"

# 5 target images: (filename, camera, mp, label, n_sources_expected)
declare -a IMAGES=(
    "TTT1_QHY411-1_2026-04-07-21-24-09-765452_Eugenia_SDSSg.fits|QHY411-1|37.8|QHY411-1_SDSSg_2k|1998"
    "TTT1_QHY411-1_2026-03-16-20-27-44-616412_V445Pup-griz_SDSSr.fits|QHY411-1|37.8|QHY411-1_SDSSr_7k|6932"
    "TST_QHY411-3_2026-03-17-04-00-21-799080_M106_SDSSg.fits|QHY411-3|151.2|QHY411-3_SDSSg_10k|10005"
    "TST_QHY411-3_2026-03-16-22-54-47-549608_NGC2683_SDSSr.fits|QHY411-3|151.2|QHY411-3_SDSSr_19k|19565"
    "TST_QHY411-3_2026-03-16-20-28-14-304289_C2025N1_Lum.fits|QHY411-3|151.2|QHY411-3_Lum_131k|131397"
)

# Write CSV header
echo "image,mp,cuml,rep,time_s,sources" > "$OUTPUT"
echo "Output: $OUTPUT"
echo "Container: $CONTAINER"
echo "Warmup: $WARMUP, Reps: $REPS"
echo "Images: ${#IMAGES[@]}"
echo ""

# Helper: run one image rep, return time_s and n_sources
run_one() {
    local file="$1" inst="$2" docker_env="${3:-}"
    local out
    out=$(timeout "$REP_TIMEOUT" docker exec $docker_env "$CONTAINER" \
        python3 /app/profiling_scripts/profile_process_image.py "$file" "$inst" 2>&1)
    local status time_s sources
    # profile_process_image.py output format:
    #   Status:     OK
    #   Objects:    <n>
    #   Time:       <readable_time> (<elapsed_s>s)
    status=$(echo "$out" | grep "^Status:" | awk '{print $NF}')
    time_s=$(echo "$out" | grep "^Time:" | grep -o '([0-9]*\.[0-9]*s)' | tr -d '()s')
    sources=$(echo "$out" | grep "^Objects:" | awk '{print $NF}')
    if [ "$status" != "OK" ]; then
        echo "FAILED"
        echo "  Output:" >&2
        echo "$out" | head -20 >&2
        return 1
    fi
    echo "${time_s}|${sources}"
}

# Main loop
for img_def in "${IMAGES[@]}"; do
    IFS='|' read -r file inst mp label expected_src <<< "$img_def"

    echo "========================================================================"
    echo "Image: $label  (${mp}MP, ~${expected_src} src)"
    echo "File:  $file"
    echo "========================================================================"

    # Warmup (cuml=no, no timing captured)
    echo -n "  warmup 1 (cuml=no): "
    for w in $(seq 1 $WARMUP); do
        result=$(run_one "$file" "$inst" "" 2>&1) || { echo "FAILED (skipping image)"; continue 2; }
        IFS='|' read -r t_s n_src <<< "$result"
        echo "OK ${t_s}s (${n_src} src)"
    done

    # cuml=no reps
    echo "  --- cuml=no (cKDTree forced) ---"
    for rep in $(seq 1 $REPS); do
        echo -n "    rep $rep/$REPS: "
        result=$(run_one "$file" "$inst" "" 2>&1) || { echo "FAILED"; continue; }
        IFS='|' read -r t_s n_src <<< "$result"
        echo "OK ${t_s}s (${n_src} src)"
        echo "$file,$mp,no,$rep,$t_s,$n_src" >> "$OUTPUT"
    done

    # Warmup for cuml=yes (separate warmup to ensure cuML is loaded)
    echo -n "  warmup 1 (cuml=yes): "
    for w in $(seq 1 $WARMUP); do
        result=$(run_one "$file" "$inst" "-e GPUPHOT_USE_CUML_CROSSMATCH=1" 2>&1) || { echo "FAILED"; break; }
        IFS='|' read -r t_s n_src <<< "$result"
        echo "OK ${t_s}s (${n_src} src)"
    done

    # cuml=yes reps
    echo "  --- cuml=yes (cuML forced) ---"
    for rep in $(seq 1 $REPS); do
        echo -n "    rep $rep/$REPS: "
        result=$(run_one "$file" "$inst" "-e GPUPHOT_USE_CUML_CROSSMATCH=1" 2>&1) || { echo "FAILED"; continue; }
        IFS='|' read -r t_s n_src <<< "$result"
        echo "OK ${t_s}s (${n_src} src)"
        echo "$file,$mp,yes,$rep,$t_s,$n_src" >> "$OUTPUT"
    done

    echo ""
done

echo "========================================================================"
echo "Done. Results saved to: $OUTPUT"
echo "Lines: $(wc -l < "$OUTPUT")"
cat "$OUTPUT"
