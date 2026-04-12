#!/bin/bash
# =============================================================================
# Full benchmark run — scheduled for 07:00
#
# Runs SEQUENTIALLY on each machine to avoid GPU contention in production.
# Workers are removed with `docker rm -f` (not stopped) to prevent the
# production cron from restarting them mid-benchmark. The cron recreates them
# automatically after each machine finishes.
#
# What this runs:
#   • All 22 images (1-10 originals + 11-22 high-source-count)
#   • 3 profiler modes: py312_cuml_adaptive, py312_cuml_always, py38_baseline
#   • 2 warmup + 15 measured reps per image
#   • nsys profiling after benchmarks complete (warmup 1, reps 0, nsys 2)
#   • ES data collection at the end
#
# Schedule: run from the repo root
#   at 07:00 <<< "cd /home/slemes/PycharmProjects/GPUPhotFinal && ./dev/run_full_benchmark_7am.sh"
# Or:
#   echo "cd /home/slemes/PycharmProjects/GPUPhotFinal && ./dev/run_full_benchmark_7am.sh" | at 07:00
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="/tmp/full_benchmark_$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOG_DIR/master.log"
mkdir -p "$LOG_DIR"

DATE_TAG=$(date +%Y%m%d)
TIME_FROM=$(date +%Y-%m-%dT%H:%M:%S)

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOGFILE"; }

log "======================================================================"
log " GPUPhot Full Benchmark — $(date '+%Y-%m-%d %H:%M:%S')"
log " LOG_DIR: $LOG_DIR"
log "======================================================================"

# ── Máquinas vast (secuencial para no competir en GPU entre ellas) ─────────
VAST_MACHINES="azken hp3 lenovo_tttserver ttt_server ttt1"

for machine in $VAST_MACHINES; do
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "BENCHMARK → $machine (todas las imágenes 1-22, 3 profilers, 15 reps)"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    BENCHMARK_LOG_DIR="$LOG_DIR" \
    ./dev/run_benchmark_all_machines.sh \
        --machines "$machine" \
        --profilers 312,312_always,38 \
        --warmup 2 --reps 15 \
        --stop-workers \
        2>&1 | tee -a "$LOGFILE"

    log "$machine benchmark done."
done

log ""
log "======================================================================"
log " TODOS LOS BENCHMARKS COMPLETADOS — $(date '+%Y-%m-%d %H:%M:%S')"
log "======================================================================"

# ── nsys: una máquina cada vez ─────────────────────────────────────────────
log ""
log "Iniciando nsys profiling en todas las máquinas..."

for machine in $VAST_MACHINES; do
    log ""
    log "━━━ nsys → $machine ━━━"
    BENCHMARK_LOG_DIR="$LOG_DIR" \
    ./dev/run_benchmark_all_machines.sh \
        --machines "$machine" \
        --profilers 312,312_always,38 \
        --warmup 1 --reps 0 --nsys 2 \
        --stop-workers \
        2>&1 | tee -a "$LOGFILE"
    log "$machine nsys done."
done

# ── Colección de datos ES ───────────────────────────────────────────────────
log ""
log "======================================================================"
log " Recolectando datos de Elasticsearch..."
log "======================================================================"

TIME_TO=$(date +%Y-%m-%dT%H:%M:%S)
ES_OUT="benchmarks/results_collected/benchmark_all_cuml_v2.csv"

/home/slemes/PycharmProjects/venv/TTT/bin/python3 benchmarks/collect_and_merge_all.py \
    --time-from "$TIME_FROM" \
    --time-to   "$TIME_TO" \
    --base-csv  "$ES_OUT" \
    --out       "$ES_OUT" \
    2>&1 | tee -a "$LOGFILE"

log ""
log "ES data collection done. CSV: $ES_OUT"
log ""
log "======================================================================"
log " Próximos pasos manuales:"
log "   1. Extraer VRAM de nsys sqlite en cada máquina con /tmp/extract_nsys_memory.py"
log "   2. python3 benchmarks/generate_manuscript_tables.py"
log "   3. python3 benchmarks/generate_manuscript_figures.py"
log "   4. cd GPUPHOT_manuscript && pdflatex main.tex && pdflatex main.tex"
log "======================================================================"
log " DONE — $(date '+%Y-%m-%d %H:%M:%S')"
