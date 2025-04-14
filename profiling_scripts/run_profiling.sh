#!/bin/bash

# Verificar argumentos mínimos
if [ "$#" -lt 2 ]; then
    echo "Uso: $0 <ruta_imagen> <instrument_name> [repeticiones]"
    exit 1
fi

IMAGE_PATH=$1
INSTRUMENT_NAME=$2

# Asignar el número de repeticiones (por defecto 1)
if [ -z "$3" ]; then
    COUNT=1
else
    # Validar que COUNT sea un número entero
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

# Detección de la ruta de nsys
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

OUTPUT_DIR="/app/profiling_results"
mkdir -p $OUTPUT_DIR

# Bucle para ejecutar el perfilado la cantidad de veces especificada
for ((i=1; i<=COUNT; i++)); do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="${OUTPUT_DIR}/profile_${HOSTNAME}_${INSTRUMENT_NAME}_${TIMESTAMP}_run${i}.nsys-rep"

    echo "Iniciando perfilado $i de $COUNT con Nsight Systems..."
    $NSYS_CMD profile --trace=cuda,nvtx,osrt --sample=process-tree --stats=true --cuda-memory-usage=true --gpu-metrics-devices=$GPU_ID -o "$OUTPUT_FILE" python3 profile_image_processing.py "$IMAGE_PATH" "$INSTRUMENT_NAME"
    echo "Perfilado $i completado. Resultado guardado en: $OUTPUT_FILE"
done

echo "Todos los perfiles han sido generados."
