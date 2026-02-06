#!/bin/bash
# ==============================================================================
# GPUPhot Main Launcher
#
# Description:
# This script launches the main GPUPhot application stack on a single machine
# using Docker Compose. It detects the number of available GPUs and scales the
# `gpuphot_worker` service accordingly. It also assigns a specific GPU_ID to
# each worker container.
#
# Usage:
#   ./launch_gpuphot.sh [MAX_GPUS]
#
# Arguments:
#   [MAX_GPUS]: (Optional) The maximum number of workers to launch. Defaults
#               to the total number of GPUs detected.
# ==============================================================================

# --- 1. GPU Detection and Worker Count Calculation ---
# This section determines the number of worker containers to launch based on
# the available GPUs on the host system.

# Count the number of NVIDIA GPUs available.
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Exit with an error if no GPUs are found.
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "No GPUs detected. Please check your NVIDIA drivers and installation."
  exit 1
fi

# Set the maximum number of GPUs to use from the first script argument,
# or default to the total number of available GPUs.
MAX_GPUS=${1:-$NUM_GPUS}

# Calculate the final number of workers, ensuring it does not exceed the
# available GPUs or the user-specified maximum.
if [ "$NUM_GPUS" -lt "$MAX_GPUS" ]; then
    NUM_WORKERS=$NUM_GPUS
else
    NUM_WORKERS=$MAX_GPUS
fi


# --- 2. Launch Docker Compose Services ---
# This section starts all services defined in the main `docker-compose.yml` file.

echo "Launching GPUPhot stack with $NUM_WORKERS worker(s)..."

# Start the services in detached mode.
# --scale: Overrides the replica count for the `gpuphot_worker` service to match
# the calculated number of workers.
docker compose up -d --scale gpuphot_worker=$NUM_WORKERS


# --- 3. Assign GPU IDs to Workers ---
# This loop iterates through the launched worker containers to assign a unique
# GPU_ID to each one.

# Loop from 0 to (NUM_WORKERS - 1).
for i in $(seq 0 $((NUM_WORKERS-1))); do
  # This command executes a shell command inside a specific worker container.
  # It targets containers by name (e.g., gpuphot_worker_1, gpuphot_worker_2).
  # `exec`: Runs a command in a running container.
  # `-e GPU_ID=$i`: Sets the GPU_ID environment variable for the command being executed.
  # `bash -c '...'`: The command to run, which modifies the /etc/environment file
  # to set the GPU_ID for the container's environment.
  docker compose exec -e GPU_ID=$i gpuphot_worker_$((i+1)) bash -c 'sed -i "s/GPU_ID=0/GPU_ID=$GPU_ID/" /etc/environment'
done

echo "GPUPhot launched with $NUM_WORKERS worker(s)."
