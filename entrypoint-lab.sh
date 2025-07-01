#!/bin/bash
set -e # Salir inmediatamente si un comando falla

# Directorio donde se guardarán los notebooks generados (el volumen del usuario)
NOTEBOOK_DIR="/home/jovyan/work"
# Directorio donde se encuentran los scripts generadores dentro del contenedor
GENERATOR_DIR="/app/gpuphot_worker/generators"
# Marcador para saber si ya hemos generado los notebooks
INIT_MARKER_FILE="$NOTEBOOK_DIR/.notebooks_generated"

echo "JupyterLab Entrypoint: Checking for notebooks in $NOTEBOOK_DIR..."

# Comprobamos si el marcador existe. Si no existe, generamos los notebooks.
# Esto previene que se sobreescriban los notebooks si el usuario ya los ha modificado.
if [ ! -f "$INIT_MARKER_FILE" ]; then
    echo "First time running or notebooks not found. Generating default notebooks..."

    # Asegurarse de que el directorio de destino existe
    mkdir -p "$NOTEBOOK_DIR"

    # Ejecutar cada script generador de Python
    # El bucle buscará cualquier fichero .py en el directorio de generadores.
    for generator_script in "$GENERATOR_DIR"/*.py; do
        if [ -f "$generator_script" ]; then
            echo "Running generator: $generator_script"
            # Usamos el python del entorno virtual para ejecutar el script
            /app/venv/bin/python "$generator_script" --output-dir "$NOTEBOOK_DIR"
        fi
    done

    echo "Default notebooks generated successfully."

    # Crear el fichero marcador para no volver a ejecutar esto
    touch "$INIT_MARKER_FILE"
    echo "Init marker created at $INIT_MARKER_FILE"

else
    echo "Notebooks already generated. Skipping generation."
fi

## Finalmente, ejecutar el comando original de JupyterLab
## Usamos "exec" para que JupyterLab se convierta en el proceso principal (PID 1) del contenedor,
## lo que es importante para que las señales de Docker (como 'docker stop') se manejen correctamente.
#echo "Starting JupyterLab..."
#exec jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --notebook-dir="$NOTEBOOK_DIR"