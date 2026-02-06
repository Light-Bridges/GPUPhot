#!/bin/bash
# ==============================================================================
# GPUPhot Main Launcher
#
# Author: Light-Bridges
# Date: 2024-08-02
#
# Description:
# This script is the primary entry point for launching the entire local GPUPhot
# service stack using Docker Compose. It is intended for a single-machine
# setup where the master services and workers run on the same host.
#
# The script automates:
#   1. Detecting the number of available NVIDIA GPUs.
#   2. Launching the Docker Compose stack defined in `docker-compose.yml`.
#   3. Scaling the `gpuphot_worker` service to match the number of available
#      (or requested) GPUs.
#
# Prerequisites:
#   - Docker and Docker Compose must be installed.
#   - The NVIDIA Container Toolkit must be installed to enable GPU access.
#
# Usage:
#   ./launch_gpuphot.sh [MAX_GPUS]
#
# Arguments:
#   [MAX_GPUS]: (Optional) The maximum number of worker containers to launch.
#               If not provided, it defaults to the total number of GPUs
#               detected on the system.
#
# ==============================================================================

# --- GPU DETECTION ---
# This section determines the number of NVIDIA GPUs available on the host system.

# `nvidia-smi -L` lists all available GPUs, one per line.
# `wc -l` counts the number of lines, giving the total GPU count.
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Check if any GPUs were detected. If not, exit with an error.
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "Error: No GPUs detected. Please check your NVIDIA driver installation and the NVIDIA Container Toolkit."
  exit 1
fi

# --- WORKER COUNT CALCULATION ---
# This section determines how many worker containers to launch.

# Set the maximum number of GPUs to use. Defaults to all available GPUs if the
# first script argument ($1) is not provided.
MAX_GPUS=${1:-$NUM_GPUS}

# Calculate the final number of workers. It's the minimum of the available GPUs
# and the maximum requested by the user.
if [ "$NUM_GPUS" -lt "$MAX_GPUS" ]; then
    NUM_WORKERS=$NUM_GPUS
else
    NUM_WORKERS=$MAX_GPUS
fi

echo "Detected $NUM_GPUS GPUs. Launching GPUPhot stack with $NUM_WORKERS worker(s)..."

# --- LAUNCH DOCKER COMPOSE STACK ---
# This command starts all services defined in the `docker-compose.yml` file.

# `docker compose up`: Creates and starts all services.
# `-d`: Detached mode (runs services in the background).
# `--scale gpuphot_worker=$NUM_WORKERS`: Overrides the `replicas` or `scale`
#   setting for the `gpuphot_worker` service, launching the desired number of
#   worker containers.
docker compose up -d --scale gpuphot_worker=$NUM_WORKERS

# --- DYNAMIC GPU ASSIGNMENT (LEGACY - NOT RECOMMENDED) ---
# The block below is a legacy method for assigning GPUs and is not robust.
# The modern, recommended approach is to use the `deploy.reservations.devices`
# section in the `docker-compose.yml` file, which allows Docker to manage GPU
# assignment declaratively and reliably. This block is preserved for historical
# context but should be considered for removal. See FIXERS.md for more details.

# for i in $(seq 0 $((NUM_WORKERS-1))); do
#   # This command is fragile as it assumes a specific container naming scheme
#   # (e.g., project_service_1, project_service_2) which can vary.
#   docker compose exec -e GPU_ID=$i gpuphot_worker_$((i+1)) bash -c 'sed -i "s/GPU_ID=0/GPU_ID=$GPU_ID/" /etc/environment'
# done

echo "GPUPhot stack has been launched."
echo "Use 'docker compose ps' to see the status of all services."
echo "Use 'docker compose logs -f' to follow the logs."
