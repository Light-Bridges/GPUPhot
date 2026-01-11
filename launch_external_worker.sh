#!/bin/bash

# launch_external_worker.sh

# 1. Check for arguments
if [ -z "$1" ]; then
  echo "Usage: ./launch_external_worker.sh <MASTER_IP> [MAX_GPUS]"
  echo "Example: ./launch_external_worker.sh 192.168.1.50"
  exit 1
fi

MASTER_IP=$1
MAX_GPUS=${2:-$(nvidia-smi -L | wc -l)}

# 2. Check GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
  echo "No GPUs detected."
  exit 1
fi

# Limit workers if requested
NUM_WORKERS=$(( NUM_GPUS < MAX_GPUS ? NUM_GPUS : MAX_GPUS ))

echo "Connecting to Cluster Master at: $MASTER_IP"
echo "Launching $NUM_WORKERS workers..."

# 3. Launch Docker Compose using the worker file and passing the MASTER_IP
# We use -f to specify the worker file
export MASTER_IP=$MASTER_IP
docker compose -f docker-compose.worker.yml up -d --scale gpuphot_worker=$NUM_WORKERS

# 4. Assign GPU IDs
for i in $(seq 0 $((NUM_WORKERS-1))); do
  # Note: The service name in the worker file is just 'gpuphot_worker'
  # Docker compose usually names containers project_service_index
  CONTAINER_NAME="gpuphot-gpuphot_worker-$((i+1))"

  # Check if container exists (naming might vary depending on folder name)
  # A safer way is using docker compose ps
  docker compose -f docker-compose.worker.yml exec -e GPU_ID=$i gpuphot_worker bash -c 'sed -i "s/GPU_ID=0/GPU_ID=$GPU_ID/" /etc/environment'
done

echo "External workers active."