#!/bin/bash

# Verificar argumentos
if [ "$#" -lt 1 ]; then
    echo "Uso: $0 <ruta_imagen> [instrument_name]"
    exit 1
fi

IMAGE_PATH=$1
INSTRUMENT_NAME=${2:-$INSTRUMENT_NAME}
OUTPUT_DIR="/app/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/profile_${TIMESTAMP}.nsys-rep"

# Crear directorio de salida si no existe
mkdir -p $OUTPUT_DIR

# Ejecutar perfil con Nsight Systems
echo "Iniciando perfilado con Nsight Systems..."
nsys profile -t cuda,nvtx,osrt -o "$OUTPUT_FILE" --stats=true python /app/profiling_scripts/profile_image_processing.py "$IMAGE_PATH" "$INSTRUMENT_NAME"

echo "Perfilado completado. Resultados guardados en: $OUTPUT_FILE"
echo "Para analizar los resultados, puede usar Nsight Systems UI en su máquina local con el archivo generado"
