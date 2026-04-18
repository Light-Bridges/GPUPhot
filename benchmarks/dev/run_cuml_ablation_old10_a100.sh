#!/usr/bin/env bash
# run_cuml_ablation_old10_a100.sh — re-run the 10 original ablation images
# with the current cuML mechanism (GPUPHOT_USE_CUML_CROSSMATCH=1/0)
# to produce a homogeneous dataset comparable to the 5 new images.
#
# Usage (on lenovo_tttserver host):
#   bash /tmp/run_cuml_ablation_old10_a100.sh 2>&1 | tee /tmp/cuml_ablation_old10_a100.log

set -euo pipefail

CONTAINER="gpuphot-profiler-1"
WARMUP=1
REPS=3
REP_TIMEOUT=1200  # 20 min max per rep

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT="/tmp/cuml_ablation_old10_a100_${TIMESTAMP}.csv"

# 10 original images: (filename, camera, mp)
declare -a IMAGES=(
    "TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits|iKon936-1|4.2"
    "TTT3_iKon936-1_2025-09-15-05-31-52-256207_C2025A6_Lum.fits|iKon936-1|4.2"
    "TTT3_QHY600-3_2025-12-01-23-02-47-487643_C2025R2_Lum.fits|QHY600-3|6.8"
    "TTT2_QHY600-4_2026-02-14-23-46-19-497908_WASP-43-b_SDSSg.fits|QHY600-4|15.3"
    "TTT2_QHY600-4_2026-02-14-00-13-51-310869_NGC2903_Ha.fits|QHY600-4|15.3"
    "TTT1_QHY411-1_2026-03-09-21-22-48-661122_2012QD8_Lum.fits|QHY411-1|37.8"
    "TTT1_QHY411-1_2026-02-14-03-52-12-471080_GaiaDR33534005919872722560_SDSSi.fits|QHY411-1|37.8"
    "TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits|QHY411-1|151.2"
    "TST_QHY411-3_2026-02-14-23-09-41-463160_M81_SDSSr.fits|QHY411-3|151.2"
    "TST_QHY411-3_2026-02-14-06-35-10-547282_24P_Lum.fits|QHY411-3|151.2"
)

echo "image,mp,cuml,rep,time_s,sources" > "$OUTPUT"
echo "Output: $OUTPUT"
echo "Container: $CONTAINER  Warmup: $WARMUP  Reps: $REPS"
echo "Images: ${#IMAGES[@]}"
echo ""

run_one() {
    local file="$1" inst="$2" docker_env="${3:-}"
    local out
    out=$(timeout "$REP_TIMEOUT" docker exec $docker_env "$CONTAINER" \
        python3 /app/profiling_scripts/profile_process_image.py "$file" "$inst" 2>&1)
    local status time_s sources
    status=$(echo "$out" | grep "^Status:" | awk '{print $NF}')
    time_s=$(echo "$out" | grep "^Time:" | grep -o '([0-9]*\.[0-9]*s)' | tr -d '()s')
    sources=$(echo "$out" | grep "^Objects:" | awk '{print $NF}')
    if [ "$status" != "OK" ]; then
        echo "FAILED"
        echo "$out" | head -10 >&2
        return 1
    fi
    echo "${time_s}|${sources}"
}

for img_def in "${IMAGES[@]}"; do
    IFS='|' read -r file inst mp <<< "$img_def"
    label=$(echo "$file" | grep -oE '[A-Za-z0-9_+]+\.(fits)' | sed 's/\.fits//' | cut -c1-30)

    echo "========================================================================"
    echo "Image: $label  (${mp}MP)"
    echo "========================================================================"

    # Warmup cuml=no
    echo -n "  warmup 1 (cuml=no): "
    result=$(run_one "$file" "$inst" "" 2>&1) || { echo "FAILED — skipping"; continue; }
    IFS='|' read -r t_s n_src <<< "$result"
    echo "OK ${t_s}s (${n_src} src)"

    # cuml=no reps
    echo "  --- cuml=no ---"
    for rep in $(seq 1 $REPS); do
        echo -n "    rep $rep/$REPS: "
        result=$(run_one "$file" "$inst" "" 2>&1) || { echo "FAILED"; continue; }
        IFS='|' read -r t_s n_src <<< "$result"
        echo "OK ${t_s}s (${n_src} src)"
        echo "$file,$mp,no,$rep,$t_s,$n_src" >> "$OUTPUT"
    done

    # Warmup cuml=yes
    echo -n "  warmup 1 (cuml=yes): "
    result=$(run_one "$file" "$inst" "-e GPUPHOT_USE_CUML_CROSSMATCH=1" 2>&1) || { echo "FAILED"; }
    IFS='|' read -r t_s n_src <<< "$result"
    echo "OK ${t_s}s (${n_src} src)"

    # cuml=yes reps
    echo "  --- cuml=yes ---"
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
echo "Done. Results: $OUTPUT  ($(wc -l < "$OUTPUT") lines)"
cat "$OUTPUT"
