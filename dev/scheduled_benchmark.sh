#!/bin/bash
# =============================================================================
# Scheduled Benchmark — Full run on all 8 machines
#
# This script is designed to be run via cron or at.
# It logs everything to /tmp/benchmark_scheduled_<timestamp>/
#
# What it does:
#   1. Ensures SSH tunnel for jetson_local logstash
#   2. Stops GPU0 workers on production machines
#   3. Runs 2 warmup + 10 clean reps on all 8 machines (all profilers)
#   4. Restores workers when ALL machines finish
#   5. Collects timing data from Elasticsearch
# =============================================================================

set -uo pipefail

# --- Cron environment setup ---
# Cron runs with minimal env; we need SSH agent and PATH
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export SSH_AUTH_SOCK="/run/user/$(id -u)/keyring/ssh"
export HOME="${HOME:-/home/slemes}"

# Verify SSH works before proceeding
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes lenovo_tttserver "echo ok" &>/dev/null; then
    echo "FATAL: SSH not working (agent not available?). Aborting." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/benchmark_scheduled_${TIMESTAMP}"
MASTER_LOG="${LOG_DIR}/master.log"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "=========================================================="
log " GPUPhot Scheduled Benchmark"
log " Started: $(date '+%Y-%m-%d %H:%M:%S')"
log " Logs: $LOG_DIR"
log "=========================================================="

# Record the start time for Elastic query later (UTC)
BENCHMARK_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%S)
log "Elastic query start time (UTC): $BENCHMARK_START_UTC"

# --- Step 1: Ensure SSH tunnel for jetson_local ---
log ""
log "--- Step 1: SSH tunnel for jetson_local ---"
if ssh -o ConnectTimeout=5 -o BatchMode=yes jetson_local "echo ok" &>/dev/null; then
    # Check if tunnel is already open
    if ! ssh -o ConnectTimeout=5 jetson_local "ss -tln | grep -q ':5000'" 2>/dev/null; then
        log "Opening SSH tunnel for jetson_local logstash..."
        ssh -o ConnectTimeout=5 -fN -R 5000:10.0.210.30:5000 jetson_local 2>/dev/null
        sleep 2
        log "Tunnel opened."
    else
        log "Tunnel already active."
    fi
else
    log "WARNING: jetson_local offline, skipping tunnel."
fi

# --- Step 2: Run benchmark with --stop-workers ---
log ""
log "--- Step 2: Launching benchmark ---"
log "Command: $SCRIPT_DIR/run_benchmark_all_machines.sh --stop-workers --warmup 2 --reps 10 --nsys 2"

# Override LOG_DIR in the child script via environment
export BENCHMARK_LOG_DIR="$LOG_DIR"

bash "$SCRIPT_DIR/run_benchmark_all_machines.sh" \
    --stop-workers \
    --warmup 2 \
    --reps 10 \
    --nsys 2 \
    >> "$MASTER_LOG" 2>&1

log ""
log "=========================================================="
log " Benchmark finished: $(date '+%Y-%m-%d %H:%M:%S')"
log "=========================================================="

# --- Step 3: Collect Elastic data ---
log ""
log "--- Step 3: Collecting data from Elasticsearch ---"
ELASTIC_CSV="${PROJECT_DIR}/benchmarks/es_times_benchmark_${TIMESTAMP}.csv"

cd "$PROJECT_DIR"
python3 benchmarks/collect_times_from_elastic.py \
    --time-from "$BENCHMARK_START_UTC" \
    --out "$ELASTIC_CSV" \
    >> "$MASTER_LOG" 2>&1

if [ -f "$ELASTIC_CSV" ]; then
    ROWS=$(wc -l < "$ELASTIC_CSV")
    log "Elastic data collected: $ELASTIC_CSV ($ROWS rows)"
else
    log "WARNING: Failed to collect Elastic data"
fi

# --- Summary ---
log ""
log "=========================================================="
log " SUMMARY"
log "=========================================================="
log "Log directory: $LOG_DIR"
log "Elastic CSV:   $ELASTIC_CSV"
log ""
log "Per-machine results:"
for f in "$LOG_DIR"/*.log; do
    [ "$f" = "$MASTER_LOG" ] && continue
    name=$(basename "$f" .log)
    if grep -q "^=== DONE" "$f" 2>/dev/null; then
        ok=$(grep -c "rep .*/.*: OK" "$f" 2>/dev/null || echo 0)
        skip=$(grep -c "Skip\|SKIP" "$f" 2>/dev/null || echo 0)
        log "  $name: DONE ($ok reps OK, $skip skips)"
    else
        log "  $name: INCOMPLETE"
    fi
done

log ""
log "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
