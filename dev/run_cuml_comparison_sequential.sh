#!/bin/bash
# =============================================================================
# Benchmark cuML-always vs cuML-adaptive — comparativa completa
#
# Modelo de ejecucion:
#   - Entre maquinas: PARALELO (cada maquina corre su secuencia en background)
#   - Dentro de cada maquina: SECUENCIAL:
#       1. py3.12  cuML always    (GPUPHOT_ENVIRONMENT=profiler_cuml_always_v2)
#       2. py3.12  cuML adaptive  (GPUPHOT_ENVIRONMENT=profiler_cuml_adaptive_v2)
#       3. py3.8   baseline       (GPUPHOT_ENVIRONMENT=profiler_py38_v2)
#       4. Restaurar workers y .env
#
# Los workers (dto-worker0-1) se paran ANTES del paso 1 y se restauran
# despues del paso 3. Un solo stop/start por maquina, no uno por fase.
#
# Uso:
#   bash dev/run_cuml_comparison_sequential.sh
#   bash dev/run_cuml_comparison_sequential.sh --machines local,ttt1
#
# Tiempo estimado por maquina (2 warmup + 10 reps, 3 runs de profiler):
#   local      RTX 3050 Ti   ~2h     (solo 4.2MP, 2 imagenes)
#   ttt1       RTX 3060      ~4h     (hasta 37.8MP, 7 imagenes)
#   ttt_server RTX 3090      ~4h     (hasta 37.8MP, 7 imagenes)
#   azken      H100 PCIe     ~7h     (10 imagenes, 151.2MP)
#   lenovo     A100-SXM4     ~10h    (10 imagenes, 151.2MP)
#   hp3        L40S          ~13h    (10 imagenes, 151.2MP)
#
# Nota: jetson_orin y jetson_local no tienen cuML (ARM), se omiten aqui.
# =============================================================================
set -uo pipefail

MACHINE_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --machines) MACHINE_FILTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="/tmp/cuml_comparison_v2_${TIMESTAMP}.log"
LOG_BASE="/tmp/cuml_v2_${TIMESTAMP}"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
logm() { echo "[$(date '+%H:%M:%S')] [$1] $2" | tee -a "$MASTER_LOG"; }

# ---------------------------------------------------------------------------
# Umbrales adaptativos por GPU (medidos 2026-03-28 y 2026-04-02)
# ---------------------------------------------------------------------------
declare -A CUML_MIN=( [local]=2000 [ttt1]=2000 [ttt_server]=2000 [lenovo_tttserver]=2000 [hp3]=2000 [azken]=5000 )
declare -A CUML_MAX=( [local]=50000 [ttt1]=20000 [ttt_server]=100000 [lenovo_tttserver]=100000 [hp3]=200000 [azken]=500000 )

# Workers de produccion a parar/restaurar
# Workers a parar antes del benchmark y restaurar al terminar.
# Solo parar los que usan GPU0 (misma GPU que el profiler).
# dto-worker-ast-1: NO parar — esencial para produccion, no interfiere
# dto-worker1-1:    NO parar — usa GPU1, no interfiere con profiler en GPU0
declare -A PROD_WORKERS=(
    [azken]="dto-worker0-1"
    [hp3]="dto-worker0-1"
    [lenovo_tttserver]="dto-worker0-1"
)

# Lista de maquinas (orden de lanzamiento — la duracion ya no importa porque van en paralelo)
ALL_MACHINES=(local ttt1 ttt_server azken lenovo_tttserver hp3)

# Aplicar filtro --machines
MACHINES=()
for m in "${ALL_MACHINES[@]}"; do
    if [ -n "$MACHINE_FILTER" ]; then
        echo ",$MACHINE_FILTER," | grep -q ",$m," || continue
    fi
    MACHINES+=("$m")
done

log "======================================================================"
log " cuML comparison v2"
log " Maquinas (paralelo entre ellas): ${MACHINES[*]}"
log " Orden por maquina: py312_always -> py312_adaptive -> py38 -> workers up"
log " Log maestro: $MASTER_LOG"
log " Logs por maquina: ${LOG_BASE}_<host>/"
log "======================================================================"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_remote() {
    local host=$1; shift
    if [ "$host" = "local" ]; then
        eval "$@"
    else
        ssh -o ServerAliveInterval=30 "$host" "$@"
    fi
}

stop_workers() {
    local host=$1
    local workers="${PROD_WORKERS[$host]:-}"
    [ -z "$workers" ] && return 0
    logm "$host" "Parando workers: $workers"
    for w in $workers; do
        run_remote "$host" "docker stop $w 2>/dev/null || true"
    done
}

restore_workers() {
    local host=$1
    local workers="${PROD_WORKERS[$host]:-}"
    [ -z "$workers" ] && return 0
    logm "$host" "Restaurando workers: $workers"
    for w in $workers; do
        run_remote "$host" "docker start $w 2>/dev/null || true"
    done
}

set_env_always() {
    local host=$1
    local env_val="profiler_cuml_always_v2"
    logm "$host" "Config -> cuML always (ENV=${env_val})"
    run_remote "$host" "cd ~/GPUPhot 2>/dev/null || true; \
        sed -i '/^GPUPHOT_ENVIRONMENT=/d; /^GPUPHOT_USE_CUML_CROSSMATCH=/d' .env; \
        echo 'GPUPHOT_ENVIRONMENT=${env_val}' >> .env; \
        echo 'GPUPHOT_USE_CUML_CROSSMATCH=1' >> .env; \
        docker compose up -d --force-recreate profiler 2>&1 | grep -E 'Started|Error|Recreat'"
}

set_env_adaptive() {
    local host=$1
    local min="${CUML_MIN[$host]}"
    local max="${CUML_MAX[$host]}"
    local env_val="profiler_cuml_adaptive_v2"
    logm "$host" "Config -> cuML adaptive (ENV=${env_val} MIN=${min} MAX=${max})"
    run_remote "$host" "cd ~/GPUPhot 2>/dev/null || true; \
        sed -i '/^GPUPHOT_ENVIRONMENT=/d; /^GPUPHOT_USE_CUML_CROSSMATCH=/d' .env; \
        sed -i '/^GPUPHOT_CUML_MIN_SOURCES=/d; /^GPUPHOT_CUML_MAX_SOURCES=/d' .env; \
        echo 'GPUPHOT_ENVIRONMENT=${env_val}' >> .env; \
        echo 'GPUPHOT_USE_CUML_CROSSMATCH=0' >> .env; \
        echo 'GPUPHOT_CUML_MIN_SOURCES=${min}' >> .env; \
        echo 'GPUPHOT_CUML_MAX_SOURCES=${max}' >> .env; \
        docker compose up -d --force-recreate profiler 2>&1 | grep -E 'Started|Error|Recreat'"
}

set_env_py38() {
    local host=$1
    local env_val="profiler_py38_v2"
    logm "$host" "Config -> py38 baseline (ENV=${env_val})"
    run_remote "$host" "cd ~/GPUPhot 2>/dev/null || true; \
        sed -i '/^GPUPHOT_ENVIRONMENT=/d; /^GPUPHOT_USE_CUML_CROSSMATCH=/d' .env; \
        echo 'GPUPHOT_ENVIRONMENT=${env_val}' >> .env; \
        echo 'GPUPHOT_USE_CUML_CROSSMATCH=0' >> .env; \
        docker compose up -d --force-recreate profiler_38 2>&1 | grep -E 'Started|Error|Recreat'"
}

restore_env() {
    local host=$1
    run_remote "$host" "cd ~/GPUPhot 2>/dev/null || true; \
        sed -i '/^GPUPHOT_ENVIRONMENT=/d; /^GPUPHOT_USE_CUML_CROSSMATCH=/d' .env; \
        sed -i 's/^GPU_ID=.*/GPU_ID=0/' .env 2>/dev/null || true"
    logm "$host" ".env restaurado (GPU_ID=0)"
}

# ---------------------------------------------------------------------------
# Secuencia completa por maquina (se lanza en background para cada maquina)
# ---------------------------------------------------------------------------
run_machine() {
    local host=$1
    local logdir="${LOG_BASE}_${host}"
    mkdir -p "$logdir"

    logm "$host" "====== INICIO ======"

    # Verificar conectividad
    if [ "$host" != "local" ]; then
        ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "echo ok" &>/dev/null \
            || { logm "$host" "ERROR: sin conectividad SSH, saltando"; return 1; }
    fi

    # Parar workers UNA vez al inicio
    stop_workers "$host"
    sleep 3

    # ------------------------------------------------------------------
    # PASO 1: py3.12 — cuML always
    # ------------------------------------------------------------------
    logm "$host" "--- PASO 1/3: py3.12 cuML always ---"
    set_env_always "$host"
    sleep 5  # tiempo para que el container recargue las vars

    BENCHMARK_LOG_DIR="${logdir}/py312_always" \
    bash dev/run_benchmark_all_machines.sh \
        --machines "$host" \
        --profilers 312 \
        --warmup 2 \
        --reps 10 \
        --nsys 0 \
        2>&1 | tee -a "$MASTER_LOG"
    logm "$host" "PASO 1/3 DONE"

    # ------------------------------------------------------------------
    # PASO 2: py3.12 — cuML adaptive
    # ------------------------------------------------------------------
    logm "$host" "--- PASO 2/3: py3.12 cuML adaptive ---"
    set_env_adaptive "$host"
    sleep 5

    BENCHMARK_LOG_DIR="${logdir}/py312_adaptive" \
    bash dev/run_benchmark_all_machines.sh \
        --machines "$host" \
        --profilers 312 \
        --warmup 2 \
        --reps 10 \
        --nsys 0 \
        2>&1 | tee -a "$MASTER_LOG"
    logm "$host" "PASO 2/3 DONE"

    # ------------------------------------------------------------------
    # PASO 3: py3.8 — baseline (sin cuML)
    # ------------------------------------------------------------------
    logm "$host" "--- PASO 3/3: py3.8 baseline ---"
    set_env_py38 "$host"
    sleep 5

    BENCHMARK_LOG_DIR="${logdir}/py38_baseline" \
    bash dev/run_benchmark_all_machines.sh \
        --machines "$host" \
        --profilers 38 \
        --warmup 2 \
        --reps 10 \
        --nsys 0 \
        2>&1 | tee -a "$MASTER_LOG"
    logm "$host" "PASO 3/3 DONE"

    # Restaurar todo
    restore_env "$host"
    restore_workers "$host"

    logm "$host" "====== COMPLETADO ======"
}

# ---------------------------------------------------------------------------
# Lanzar todas las maquinas en paralelo
# ---------------------------------------------------------------------------
for host in "${MACHINES[@]}"; do
    run_machine "$host" &
done

echo ""
echo "Todas las maquinas lanzadas en paralelo."
echo "Cada una ejecuta: py312_always -> py312_adaptive -> py38 -> workers up"
echo ""
echo "Monitorizar:"
echo "  tail -f $MASTER_LOG"
echo "  ls ${LOG_BASE}_*/"
echo ""

wait

log ""
log "======================================================================"
log " TODAS LAS MAQUINAS COMPLETADAS — $(date '+%Y-%m-%d %H:%M:%S')"
log " Recoger datos de ES con:"
log "   environment=profiler_cuml_always_v2    (py312 cuML forzado)"
log "   environment=profiler_cuml_adaptive_v2  (py312 cuML dinamico)"
log "   environment=profiler_py38_v2           (py38 baseline)"
log "======================================================================"
