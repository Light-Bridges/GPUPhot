#!/bin/bash

# Verificar argumentos mínimos
if [ "$#" -lt 2 ]; then
    echo "Uso: $0 <ruta_imagen> <instrument_name> [repeticiones]"
    exit 1
fi

IMAGE_PATH=$1
INSTRUMENT_NAME=$2

# Número de repeticiones (por defecto 1)
if [ -z "$3" ]; then
    COUNT=1
else
    if ! [[ "$3" =~ ^[0-9]+$ ]]; then
        echo "El contador debe ser un número entero."
        exit 1
    fi
    COUNT=$3
fi

# Establecer GPU_ID si no está definida y exportar CUDA_VISIBLE_DEVICES
if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Determinar la ruta absoluta del directorio donde se encuentra este script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Detección de la ruta de nsys
if [ -d "/opt/nsight-systems-host" ]; then
    NSYS_PATH=$(readlink -f "$(which nsys)")
    if [[ "$NSYS_PATH" == /opt/nsight-systems-host/* ]]; then
        NSYS_CMD="$NSYS_PATH"
    else
        NSYS_CMD="nsys"
    fi
else
    NSYS_CMD="nsys"
fi

OUTPUT_DIR="/app/profiling_results"
mkdir -p "$OUTPUT_DIR"

# Configurar el parámetro para GPU metrics por defecto
EXTRA_PARAMS="--gpu-metrics-devices=$GPU_ID"

# Detectar si es un dispositivo Jetson mediante /proc/device-tree/model
if [ -f /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' </proc/device-tree/model)
    if echo "$MODEL" | grep -qi "jetson"; then
        echo "Dispositivo Jetson detectado ($MODEL). Se omitirá el parámetro --gpu-metrics-devices."
        EXTRA_PARAMS=""
    fi
fi

# Alternativamente, si se define la variable de entorno JETSON_TYPE, se asume que es Jetson
if [ -n "$JETSON_TYPE" ]; then
    echo "Variable de entorno JETSON_TYPE detectada ($JETSON_TYPE). Se omitirá el parámetro --gpu-metrics-devices."
    EXTRA_PARAMS=""
fi

# Si aun no se ha omitido, se chequea mediante uname -a buscando 'tegra'
if uname -a | grep -qi "tegra"; then
    echo "Dispositivo Jetson detectado (según uname -a). Se omitirá el parámetro --gpu-metrics-devices."
    EXTRA_PARAMS=""
fi

# Bucle para ejecutar el perfilado COUNT veces
for ((i=1; i<=COUNT; i++)); do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="${OUTPUT_DIR}/profile_${HOSTNAME}_${INSTRUMENT_NAME}_${TIMESTAMP}_run${i}.nsys-rep"

    echo "Iniciando perfilado $i de $COUNT con Nsight Systems..."
    $NSYS_CMD profile --trace=cuda,nvtx,osrt --sample=process-tree --stats=true --cuda-memory-usage=true $EXTRA_PARAMS -o "$OUTPUT_FILE" python3 "$SCRIPT_DIR/profile_image_processing.py" "$IMAGE_PATH" "$INSTRUMENT_NAME"
    echo "Perfilado $i completado. Resultado guardado en: $OUTPUT_FILE"
done

echo "Todos los perfiles han sido generados."
