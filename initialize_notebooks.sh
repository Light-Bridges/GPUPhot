#!/bin/bash
# ==============================================================================
# GPUPhot Notebook Initializer
#
# Author: Light-Bridges
# Date: 2024-08-02
#
# Description:
# This script is designed to be the entry point for the JupyterLab service
# container. Its primary responsibilities are:
#   1. Set up the JupyterLab working directory by creating symbolic links to
#      essential data and configuration directories.
#   2. Programmatically generate a set of tutorial Jupyter notebooks on the
#      first run. This ensures that users always start with a fresh,
#      up-to-date set of examples.
#   3. Execute the main container command (e.g., start JupyterLab).
#
# This script is idempotent; it can be run multiple times without causing
# errors, and the notebook generation will only occur once.
# ==============================================================================

# --- CONFIGURATION AND SETUP ---

# Exit immediately if a command exits with a non-zero status.
# This is a best practice for robust shell scripting.
set -e

# --- PATH DEFINITIONS ---
# Define key paths as variables for clarity and ease of maintenance.

# The primary working directory visible inside the JupyterLab interface.
# This corresponds to the persistent volume for user work.
WORK_DIR="/home/jovyan/work"

# The location of the Python scripts that generate the .ipynb notebook files.
GENERATOR_DIR="/app/gpuphot_worker/generators"

# The source paths for data and configuration that will be linked into the work directory.
# These paths are typically mounted from the host system into the container.
INSTRUMENT_CONFIGS_PATH="/data/instrument_configs"
IMAGES_PATH="/data/images"

# A marker file to track whether the initial notebook generation has been completed.
# This prevents the notebooks from being regenerated on every container restart.
INIT_MARKER_FILE="$WORK_DIR/.notebooks_generated"


# --- SYMBOLIC LINK CREATION ---
# This block runs every time the container starts to ensure that the necessary
# data directories are accessible from the JupyterLab environment. This is a
# fast and safe operation.

echo "Creating/updating symbolic links for data directories..."

# Ensure the target working directory exists. The -p flag prevents errors if it already exists.
mkdir -p "$WORK_DIR"

# Create symbolic links.
# -s: create a symbolic link.
# -f: force (remove any existing file/link at the destination).
# -n: no-dereference (treat the destination as a normal file if it's a symlink to a directory).
# This combination is robust for ensuring the link points to the correct source.
ln -sfn "${INSTRUMENT_CONFIGS_PATH}" "${WORK_DIR}"
ln -sfn "${IMAGES_PATH}" "${WORK_DIR}"

echo "Symbolic links are set up."


# --- NOTEBOOK GENERATION (FIRST RUN ONLY) ---
# This block checks for the existence of the marker file. If the file is not
# found, it iterates through the generator scripts and executes them to create
# the tutorial notebooks.

echo "Notebook Initializer: Checking for initial notebooks in $WORK_DIR..."
if [ ! -f "$INIT_MARKER_FILE" ]; then
    echo "First time running or marker file not found. Generating default notebooks..."

    # Loop through all Python scripts in the generator directory.
    for generator_script in "$GENERATOR_DIR"/*.py; do
        if [ -f "$generator_script" ]; then
            echo "Running generator: $generator_script"
            # Execute the Python script, passing the WORK_DIR as the output directory.
            # It's important to use the Python executable from the virtual environment
            # to ensure all dependencies are available.
            /app/venv/bin/python "$generator_script" --output-dir "$WORK_DIR"
        fi
    done

    echo "Default notebooks generated successfully."

    # Create the marker file to prevent this block from running again.
    touch "$INIT_MARKER_FILE"
    echo "Init marker created at $INIT_MARKER_FILE"
else
    echo "Notebooks already generated. Skipping generation."
fi


# --- EXECUTE MAIN COMMAND ---
# This is the final and most critical step. The `exec "$@"` command replaces
# the current shell process with the command passed as arguments to this script.
# In the context of Docker, this is typically the `CMD` or `command` specified
# in the docker-compose.yml file (e.g., 'jupyter-lab --ip=0.0.0.0 ...').
# This ensures that JupyterLab becomes the main process (PID 1) of the container,
# which is crucial for proper signal handling and container lifecycle management.

echo "Initialization script finished. Starting Jupyter Lab..."
exec "$@"
