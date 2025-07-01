#!/bin/bash
set -e # Salir inmediatamente si un comando falla

# Directorio donde se guardarán los notebooks generados
NOTEBOOK_DIR="/home/jovyan/work"
# Directorio donde se encuentran los scripts generadores
GENERATOR_DIR="/app/gpuphot_worker/generators"
# Marcador para saber si ya hemos generado los notebooks
INIT_MARKER_FILE="$NOTEBOOK_DIR/.notebooks_generated"

echo "Notebook Initializer: Checking for notebooks in $NOTEBOOK_DIR..."

if [ ! -f "$INIT_MARKER_FILE" ]; then
    echo "First time running or notebooks not found. Generating default notebooks..."

    mkdir -p "$NOTEBOOK_DIR"

    for generator_script in "$GENERATOR_DIR"/*.py; do
        if [ -f "$generator_script" ]; then
            echo "Running generator: $generator_script"
            # Llamamos a los scripts de Python con la ruta de salida
            /app/venv/bin/python "$generator_script" --output-dir "$NOTEBOOK_DIR"
        fi
    done

    echo "Default notebooks generated successfully."

    # Crear el fichero marcador
    touch "$INIT_MARKER_FILE"
    echo "Init marker created at $INIT_MARKER_FILE"
else
    echo "Notebooks already generated. Skipping generation."
fi

echo "Initialization script finished."