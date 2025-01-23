#!/bin/bash

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Check if GPUs are detected
if [ $NUM_GPUS -eq 0 ]; then
  echo "No GPUs detected. Please check your NVIDIA drivers and installation."
  exit 1
fi

# Set maximum number of GPUs to use (use all if not specified)
MAX_GPUS=${1:-$NUM_GPUS}
NUM_WORKERS=$(( NUM_GPUS < MAX_GPUS ? NUM_GPUS : MAX_GPUS ))

# Launch Docker Compose with the correct number of workers
docker compose up -d --scale gpuphot_worker=$NUM_WORKERS

# Assign specific GPUs to each worker
for i in $(seq 0 $((NUM_WORKERS-1))); do
  docker compose exec -e GPU_ID=$i gpuphot_worker_$((i+1)) bash -c 'sed -i "s/GPU_ID=0/GPU_ID=$GPU_ID/" /etc/environment'
done

echo "GPUPhot launched with $NUM_WORKERS worker(s)."
