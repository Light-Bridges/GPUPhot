#!/bin/bash
# =============================================================================
# Fetch benchmark images from remote server
#
# Usage:
#   ./benchmarks/fetch_benchmark_images.sh [remote_host]
#
# The script will try these hosts in order:
#   1. Argument passed on command line
#   2. lenovo_tttserver
#   3. hp3
#   4. azken
#
# Images are downloaded to benchmarks/benchmark_images/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/benchmark_images"

mkdir -p "$DEST_DIR"

# Remote source paths (from /mnt/vast/red/)
declare -A IMAGES=(
    # OBLINEID  =>  remote_path
    ["5108788"]="/mnt/vast/red/2026-01-14/TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits"
    ["4124611"]="/mnt/vast/red/2025-09-14/TTT3_iKon936-1_2025-09-15-05-31-52-256207_C2025A6_Lum.fits"
    ["4778615"]="/mnt/vast/red/2025-12-01/TTT3_QHY600-3_2025-12-01-23-02-47-487643_C2025R2_Lum.fits"
    ["5411873"]="/mnt/vast/red/2026-02-14/TTT2_QHY600-4_2026-02-14-23-46-19-497908_WASP-43-b_SDSSg.fits"
    ["5405929"]="/mnt/vast/red/2026-02-13/TTT2_QHY600-4_2026-02-14-00-13-51-310869_NGC2903_Ha.fits"
    ["5597395"]="/mnt/vast/red/2026-03-09/TTT1_QHY411-1_2026-03-09-21-22-48-661122_2012QD8_Lum.fits"
    ["5406447"]="/mnt/vast/red/2026-02-13/TTT1_QHY411-1_2026-02-14-03-52-12-471080_GaiaDR33534005919872722560_SDSSi.fits"
    ["3903693"]="/mnt/vast/red/2025-08-14/TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits"
    ["5404937"]="/mnt/vast/red/2026-02-13/TST_QHY411-3_2026-02-14-06-35-10-547282_24P_Lum.fits"
    ["5412552"]="/mnt/vast/red/2026-02-14/TST_QHY411-3_2026-02-14-23-09-41-463160_M81_SDSSr.fits"
)

# Descriptions for progress output
declare -A LABELS=(
    ["5108788"]="01. iKon936 2048x2048 SDSSg   (QSO0957+561)"
    ["4124611"]="02. iKon936 2048x2048 Lum     (C2025A6)"
    ["4778615"]="03. QHY600-3 3191x2129 Lum    (C2025R2)"
    ["5411873"]="04. QHY600-4 4787x3193 SDSSg  (WASP-43-b)"
    ["5405929"]="05. QHY600-4 4787x3193 Ha     (NGC2903)"
    ["5597395"]="06. QHY411-1 7100x5325 Lum    (2012QD8)"
    ["5406447"]="07. QHY411-1 7100x5325 SDSSi  (GaiaDR3...)"
    ["3903693"]="08. QHY411-1 14200x10650 Lum  (2025PR1)"
    ["5404937"]="09. QHY411-3 14200x10650 Lum  (24P)"
    ["5412552"]="10. QHY411-3 14200x10650 SDSSr (M81)"
)

# Determine remote host
HOSTS=("${1:-}" "lenovo_tttserver" "hp3" "azken")
REMOTE_HOST=""

for host in "${HOSTS[@]}"; do
    [ -z "$host" ] && continue
    echo "Trying $host..."
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" &>/dev/null; then
        REMOTE_HOST="$host"
        echo "Connected to $REMOTE_HOST"
        break
    fi
done

if [ -z "$REMOTE_HOST" ]; then
    echo "ERROR: Could not connect to any remote host."
    echo "Usage: $0 [remote_host]"
    exit 1
fi

echo ""
echo "=============================================="
echo " Fetching benchmark images from $REMOTE_HOST"
echo " Destination: $DEST_DIR"
echo "=============================================="

# Ordered list of OBLINEIDs
ORDERED_IDS=(5108788 4124611 4778615 5411873 5405929 5597395 5406447 3903693 5404937 5412552)

TOTAL=${#ORDERED_IDS[@]}
OK=0
FAIL=0
SKIP=0

for obid in "${ORDERED_IDS[@]}"; do
    remote_path="${IMAGES[$obid]}"
    filename="$(basename "$remote_path")"
    local_path="${DEST_DIR}/${filename}"
    label="${LABELS[$obid]}"

    if [ -f "$local_path" ]; then
        echo "[SKIP] $label — already exists"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "[FETCH] $label"
    echo "        $remote_path"
    if scp -q "$REMOTE_HOST:$remote_path" "$local_path" 2>/dev/null; then
        size=$(du -h "$local_path" | cut -f1)
        echo "        -> $filename ($size)"
        OK=$((OK + 1))
    else
        echo "        -> FAILED"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=============================================="
echo " Done: $OK fetched, $SKIP skipped, $FAIL failed (of $TOTAL)"
echo "=============================================="
ls -lhS "$DEST_DIR"/*.fits 2>/dev/null || echo "(no files)"
