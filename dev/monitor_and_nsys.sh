#!/bin/bash
# =============================================================================
# Benchmark monitor + nsys auto-launcher
#
# Monitorea benchmarks en curso y, cuando cada máquina termina:
#   1. (ttt1) espera a que la copia de imágenes termine y lanza el benchmark
#   2. Lanza nsys en esa máquina automáticamente (312, 312_always, 38)
#
# Usage:
#   ./dev/monitor_and_nsys.sh [OPTIONS]
#
# Options:
#   --vast-log-dir DIR        Log dir para lenovo/hp3/azken/ttt_server (si mismo dir)
#   --machine HOST:LABEL:DIR[:MAXMP]
#                             Registra una máquina con su propio log dir.
#                             Puede usarse múltiples veces. MAXMP opcional (default 0=all).
#                             Ejemplo: --machine ttt_server:ttt_server_RTX3090:/tmp/bench_old
#                                      --machine lenovo_tttserver:lenovo_A100:/tmp/bench_new
#   --local-log-dir DIR       Log dir del benchmark local (RTX 3050 Ti)
#   --ttt1-copy-log FILE      Log de la copia de imágenes a ttt1
#   --ttt1-copy-total N       Nº de ficheros a copiar en ttt1 (default: 9)
#   --poll N                  Intervalo de polling en segundos (default: 30)
#   --dry-run                 Muestra comandos sin ejecutarlos
#
# Ejemplos:
#   # Caso simple — todos en el mismo log dir:
#   ./dev/monitor_and_nsys.sh --vast-log-dir /tmp/bench_20260411_013219 ...
#
#   # Caso avanzado — máquinas con distintos log dirs:
#   ./dev/monitor_and_nsys.sh \
#     --machine ttt_server:ttt_server_RTX3090:/tmp/benchmark_old \
#     --machine lenovo_tttserver:lenovo_A100:/tmp/benchmark_new \
#     --machine hp3:hp3_L40S:/tmp/benchmark_new \
#     --machine azken:azken_H100:/tmp/benchmark_new \
#     --local-log-dir /tmp/benchmark_local \
#     --ttt1-copy-log /tmp/copy_ttt1_relay.log
# =============================================================================

set -uo pipefail

VAST_LOG_DIR=""
LOCAL_LOG_DIR=""
TTT1_COPY_LOG=""
TTT1_COPY_TOTAL=9
POLL=30
DRY_RUN=false
EXTRA_MACHINES=()  # HOST:LABEL:DIR[:MAXMP] entries from --machine

while [[ $# -gt 0 ]]; do
    case $1 in
        --vast-log-dir)      VAST_LOG_DIR="$2";              shift 2 ;;
        --machine)           EXTRA_MACHINES+=("$2");         shift 2 ;;
        --local-log-dir)     LOCAL_LOG_DIR="$2";             shift 2 ;;
        --ttt1-copy-log)     TTT1_COPY_LOG="$2";             shift 2 ;;
        --ttt1-copy-total)   TTT1_COPY_TOTAL="$2";           shift 2 ;;
        --poll)              POLL="$2";                       shift 2 ;;
        --dry-run)           DRY_RUN=true;                   shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Profiler modes ────────────────────────────────────────────────────────────
BENCH_PROFILERS=(312 312_always 38)
NSYS_PROFILERS="312,312_always,38"

# ── Machine registry ──────────────────────────────────────────────────────────
declare -A MACHINE_LOG_DIR
declare -A MACHINE_LABEL
declare -A MACHINE_MAX_MP

register_machine() {
    local host="$1" label="$2" log_dir="$3" max_mp="${4:-0}"
    MACHINE_LOG_DIR[$host]="$log_dir"
    MACHINE_LABEL[$host]="$label"
    MACHINE_MAX_MP[$host]="$max_mp"
}

# --vast-log-dir registra las 4 máquinas vast con el mismo dir
[ -n "$VAST_LOG_DIR" ] && {
    register_machine lenovo_tttserver lenovo_A100        "$VAST_LOG_DIR"
    register_machine hp3              hp3_L40S           "$VAST_LOG_DIR"
    register_machine azken            azken_H100         "$VAST_LOG_DIR"
    register_machine ttt_server       ttt_server_RTX3090 "$VAST_LOG_DIR"
}

# --machine HOST:LABEL:DIR[:MAXMP] — sobreescribe o añade máquinas individualmente
for entry in "${EXTRA_MACHINES[@]:-}"; do
    IFS=':' read -r host label log_dir max_mp <<< "${entry}:0"
    register_machine "$host" "$label" "$log_dir" "$max_mp"
done

[ -n "$LOCAL_LOG_DIR" ] && register_machine local local_RTX3050Ti "$LOCAL_LOG_DIR"

TTT1_REGISTERED=false
TTT1_BENCH_LAUNCHED=false
TTT1_LOG_DIR=""

NSYS_LAUNCHED=()   # hosts where nsys has been launched

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "$(date '+%H:%M:%S')  $*"; }

machine_bench_done() {
    local host="$1"
    local label="${MACHINE_LABEL[$host]}"
    local log_dir="${MACHINE_LOG_DIR[$host]}"
    for prof in "${BENCH_PROFILERS[@]}"; do
        local f="${log_dir}/${label}_py${prof}.log"
        [ -f "$f" ] && grep -q "=== DONE" "$f" || return 1
    done
    return 0
}

machine_bench_progress() {
    local host="$1"
    local label="${MACHINE_LABEL[$host]}"
    local log_dir="${MACHINE_LOG_DIR[$host]}"
    for prof in "${BENCH_PROFILERS[@]}"; do
        local f="${log_dir}/${label}_py${prof}.log"
        if [ -f "$f" ] && ! grep -q "=== DONE" "$f"; then
            tail -1 "$f" 2>/dev/null | tr -d '\n'
            return
        fi
    done
}

ttt1_copy_done() {
    [ -z "$TTT1_COPY_LOG" ] && return 1
    [ -f "$TTT1_COPY_LOG" ] || return 1
    local ok_count fail_count total
    ok_count=$(grep " OK$"   "$TTT1_COPY_LOG" 2>/dev/null | wc -l)
    fail_count=$(grep " FAIL$" "$TTT1_COPY_LOG" 2>/dev/null | wc -l)
    total=$(( ok_count + fail_count ))
    [ "$total" -ge "$TTT1_COPY_TOTAL" ]
}

launch_ttt1_bench() {
    TTT1_LOG_DIR="/tmp/benchmark_ttt1_$(date +%Y%m%d_%H%M%S)"
    local nsys_log="${TTT1_LOG_DIR}_bench.log"
    local ok_count
    ok_count=$(grep " OK$" "$TTT1_COPY_LOG" 2>/dev/null | wc -l)
    log "ttt1 copy done (${ok_count}/${TTT1_COPY_TOTAL} files OK) → launching benchmark"
    log "  ttt1 log dir: $TTT1_LOG_DIR"

    local cmd="./dev/run_benchmark_all_machines.sh \
        --machines ttt1 \
        --profilers 312,312_always,38 \
        --max-mp 40 \
        --warmup 2 --reps 10"

    if $DRY_RUN; then
        log "  [DRY-RUN] $cmd"
        mkdir -p "$TTT1_LOG_DIR"  # create empty dir so registration works
    else
        eval "$cmd" > "$nsys_log" 2>&1 &
        log "  ttt1 benchmark started (PID $!)"
    fi

    register_machine ttt1 ttt1_RTX3060 "$TTT1_LOG_DIR" 40
    TTT1_REGISTERED=true
    TTT1_BENCH_LAUNCHED=true
}

launch_nsys() {
    local host="$1"
    local label="${MACHINE_LABEL[$host]}"
    local max_mp="${MACHINE_MAX_MP[$host]}"
    local mp_arg=""
    [ "$max_mp" != "0" ] && mp_arg="--max-mp $max_mp"

    local nsys_log="/tmp/nsys_${label}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "NSYS → $label  (log: $nsys_log)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local cmd="./dev/run_benchmark_all_machines.sh \
        --machines $host \
        --profilers $NSYS_PROFILERS \
        --warmup 1 --reps 0 --nsys 2 \
        --stop-workers $mp_arg"

    if $DRY_RUN; then
        log "  [DRY-RUN] $cmd"
    else
        eval "$cmd" > "$nsys_log" 2>&1 &
        log "  nsys started (PID $!)"
    fi

    NSYS_LAUNCHED+=("$host")
}

# ── Header ────────────────────────────────────────────────────────────────────
echo "======================================================================"
echo " GPUPhot Benchmark Monitor + nsys auto-launcher"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Poll: ${POLL}s | Profilers (bench): ${BENCH_PROFILERS[*]}"
echo " nsys profilers: $NSYS_PROFILERS"
$DRY_RUN && echo " *** DRY RUN ***"
echo " Vast log:   ${VAST_LOG_DIR:-none}"
echo " Local log:  ${LOCAL_LOG_DIR:-none}"
echo " ttt1 copy:  ${TTT1_COPY_LOG:-none} (${TTT1_COPY_TOTAL} files)"
echo "======================================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
while true; do

    # ── Step 1: check ttt1 copy → launch benchmark if ready ──────────────────
    if [ -n "$TTT1_COPY_LOG" ] && ! $TTT1_BENCH_LAUNCHED; then
        if ttt1_copy_done; then
            launch_ttt1_bench
        else
            ok_n=$(grep " OK$" "$TTT1_COPY_LOG" 2>/dev/null | wc -l)
            log "ttt1 copy: ${ok_n}/${TTT1_COPY_TOTAL} files done"
        fi
    fi

    # ── Step 2: check each registered machine ────────────────────────────────
    total_machines=${#MACHINE_LOG_DIR[@]}
    total_nsys=${#NSYS_LAUNCHED[@]}

    for host in "${!MACHINE_LOG_DIR[@]}"; do
        [[ " ${NSYS_LAUNCHED[*]:-} " =~ " ${host} " ]] && continue

        if machine_bench_done "$host"; then
            launch_nsys "$host"
        else
            progress=$(machine_bench_progress "$host")
            log "  ${MACHINE_LABEL[$host]}: ${progress:0:90}"
        fi
    done

    echo ""

    # ── Step 3: exit when all registered machines have nsys launched ─────────
    total_machines=${#MACHINE_LOG_DIR[@]}
    total_nsys=${#NSYS_LAUNCHED[@]}

    # Wait for ttt1 to be registered before declaring "all done"
    ttt1_pending=false
    [ -n "$TTT1_COPY_LOG" ] && ! $TTT1_REGISTERED && ttt1_pending=true

    if ! $ttt1_pending && [ "$total_nsys" -eq "$total_machines" ] && [ "$total_machines" -gt 0 ]; then
        log "All machines nsys-launched. Waiting for background jobs..."
        wait
        echo ""
        echo "======================================================================"
        echo " ALL DONE — $(date '+%Y-%m-%d %H:%M:%S')"
        echo "======================================================================"
        break
    fi

    log "Progress: ${total_nsys}/${total_machines} nsys launched$(${ttt1_pending} && echo ' (+ttt1 copy pending)' || echo '')"
    sleep "$POLL"
done
