#!/bin/bash

# Obtener el número de GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Lanzar Docker Compose con el número correcto de workers
docker compose up -d --scale gpuphot_worker=$NUM_GPUS

# Asignar GPUs específicas a cada worker
for i in $(seq 0 $((NUM_GPUS-1))); do
  docker compose exec -e GPU_ID=$i gpuphot_worker_$((i+1)) bash -c 'sed -i "s/GPU_ID=0/GPU_ID=$GPU_ID/" /etc/environment'
done