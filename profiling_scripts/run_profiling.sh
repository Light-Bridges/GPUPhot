#!/bin/bash

# Verificar argumentos
if [ "$#" -lt 1 ]; then
    echo "Uso: $0 <ruta_imagen> [instrument_name]"
    exit 1
fi

# Detección de ruta nsys
if [ -d "/opt/nsight-systems-host" ]; then
    NSYS_PATH=$(readlink -f $(which nsys))
    if [[ "$NSYS_PATH" == /opt/nsight-systems-host/* ]]; then
        NSYS_CMD="$NSYS_PATH"
    else
        NSYS_CMD="nsys"
    fi
else
    NSYS_CMD="nsys"
fi

IMAGE_PATH=$1
INSTRUMENT_NAME=${2:-$INSTRUMENT_NAME}
OUTPUT_DIR="/app/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/profile_${TIMESTAMP}.nsys-rep"

mkdir -p $OUTPUT_DIR

echo "Iniciando perfilado con Nsight Systems..."
$NSYS_CMD profile profile --trace=cuda,nvtx,osrt --sample=process-tree --stats=true --cuda-memory-usage=true --gpu-metrics-devices=all -o "$OUTPUT_FILE" --stats=true python3 /app/profiling_scripts/profile_image_processing.py "$IMAGE_PATH" "$INSTRUMENT_NAME"

echo "Perfilado completado. Resultados guardados en: $OUTPUT_FILE"
echo "Para analizar los resultados, puede usar Nsight Systems UI en su máquina local con el archivo generado"
