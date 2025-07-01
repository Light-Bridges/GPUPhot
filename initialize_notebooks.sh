#!/bin/bash
set -e # Salir inmediatamente si un comando falla

# --- DEFINICIÓN DE RUTAS ---
# Directorio de trabajo de Jupyter donde todo será visible
WORK_DIR="/home/jovyan/work"
# Directorio donde se encuentran los scripts generadores de notebooks
GENERATOR_DIR="/app/gpuphot_worker/generators"
# Rutas de datos que queremos enlazar
INSTRUMENT_CONFIGS_PATH="/data/instrument_configs"
IMAGES_PATH="/data/images"
# Marcador para saber si ya hemos generado los notebooks
INIT_MARKER_FILE="$WORK_DIR/.notebooks_generated"

# --- CREACIÓN DE ENLACES SIMBÓLICOS ---
# Esto se ejecuta cada vez que el contenedor arranca para asegurar que los enlaces existan.
# Es una operación muy rápida, no afecta al rendimiento.
echo "Creating/updating symbolic links for data directories..."
# Primero nos aseguramos de que el directorio de trabajo exista
mkdir -p "$WORK_DIR"
# Creamos los enlaces. -n previene errores si el enlace ya existe y apunta a un directorio.
ln -sfn "${INSTRUMENT_CONFIGS_PATH}" "${WORK_DIR}"
ln -sfn "${IMAGES_PATH}" "${WORK_DIR}"
echo "Symbolic links are set up."

# --- GENERACIÓN DE NOTEBOOKS (SOLO LA PRIMERA VEZ) ---
echo "Notebook Initializer: Checking for initial notebooks in $WORK_DIR..."
if [ ! -f "$INIT_MARKER_FILE" ]; then
    echo "First time running or marker file not found. Generating default notebooks..."

    for generator_script in "$GENERATOR_DIR"/*.py; do
        if [ -f "$generator_script" ]; then
            echo "Running generator: $generator_script"
            # Llamamos a los scripts de Python con la ruta de salida
            /app/venv/bin/python "$generator_script" --output-dir "$WORK_DIR"
        fi
    done

    echo "Default notebooks generated successfully."

    # Crear el fichero marcador para no volver a ejecutar esto
    touch "$INIT_MARKER_FILE"
    echo "Init marker created at $INIT_MARKER_FILE"
else
    echo "Notebooks already generated. Skipping generation."
fi

# --- EJECUTAR EL COMANDO PRINCIPAL (JUPYTER LAB) ---
echo "Initialization script finished. Starting Jupyter Lab..."
exec "$@"