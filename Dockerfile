# Base image with CUDA
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 as base

# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt requirements-worker.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt -r requirements-worker.txt

COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Worker target
FROM base as worker
CMD ["celery", "-A", "gpuphot_worker.worker_app", "worker", "--loglevel=info"]

# Lab target
FROM base as lab
RUN pip3 install jupyter jupyterlab
WORKDIR /home/jovyan
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]