#!/bin/bash
# ==============================================================================
# GPUPhot External Worker Launcher
#
# Description:
# This script launches one or more `gpuphot_worker` containers on the current
# machine, configured to connect to a master node at a specified IP address.
# It scales the number of workers based on the number of available GPUs and
# assigns a unique GPU_ID to each worker container.
#
# Usage:
#   ./launch_external_worker.sh <MASTER_IP> [MAX_GPUS]
#
# Arguments:
#   <MASTER_IP>: (Required) The IP address of the master node.
#   [MAX_GPUS]:  (Optional) The maximum number of workers to launch. Defaults
#                to the total number of GPUs detected.
# ==============================================================================

# --- 1. Argument Validation and Parsing ---
# This section validates that the script was called with the required arguments
# and sets the MASTER_IP and MAX_GPUS variables.

# Exit if the first argument (MASTER_IP) is not provided.
if [ -z "$1" ]; then
  echo "Usage: ./launch_external_worker.sh <MASTER_IP> [MAX_GPUS]"
  echo "Example: ./launch_external_worker.sh 192.168.1.50"
  exit 1
fi

# Set MASTER_IP from the first argument.
MASTER_IP=$1
# Set MAX_GPUS from the second argument, or default to the number of GPUs found.
MAX_GPUS=${2:-$(nvidia-smi -L | wc -l)}


# --- 2. GPU Detection and Worker Count Calculation ---
# This section determines the number of worker containers to launch.

# Count the number of NVIDIA GPUs available on the system.
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "No GPUs detected."
  exit 1
fi

# Calculate the number of workers, ensuring it does not exceed the available GPUs
# or the user-specified maximum.
if [ "$NUM_GPUS" -lt "$MAX_GPUS" ]; then
    NUM_WORKERS=$NUM_GPUS
else
    NUM_WORKERS=$MAX_GPUS
fi

echo "Connecting to Cluster Master at: $MASTER_IP"
echo "Launching $NUM_WORKERS workers..."


# --- 3. Launch Docker Compose Services ---
# This section starts the worker containers using the specified compose file.

# Export MASTER_IP as an environment variable, making it available to the
# docker-compose.worker.yml file for service configuration.
export MASTER_IP
# Start the services defined in the worker-specific compose file.
# -f: Specifies the compose file.
# up -d: Creates and starts containers in detached mode.
# --scale: Sets the number of containers for the `gpuphot_worker` service.
docker compose -f docker-compose.worker.yml up -d --scale gpuphot_worker=$NUM_WORKERS


# --- 4. Assign GPU IDs to Workers ---
# This loop iterates through the launched workers to assign a unique GPU_ID
# to each one.

# Loop from 0 to (NUM_WORKERS - 1).
for i in $(seq 0 $((NUM_WORKERS-1))); do
  # This command executes a shell command inside a running `gpuphot_worker` container.
  # `exec`: Runs a command in a running container.
  # `-e GPU_ID=$i`: Sets the GPU_ID environment variable for the command being executed.
  # `gpuphot_worker`: The service name to target. Docker Compose will pick one of the scaled containers.
  # `bash -c '...'`: The command to run, which modifies the /etc/environment file
  # to set the GPU_ID for the container's environment.
  docker compose -f docker-compose.worker.yml exec -e GPU_ID=$i gpuphot_worker bash -c 'sed -i "s/GPU_ID=0/GPU_ID=$GPU_ID/" /etc/environment'
done

echo "External workers active."
