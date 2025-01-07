# Base image
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 as base

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && \
    apt-get install -y software-properties-common curl python3 python3-dev python3-pip python3-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set working directory
WORKDIR /app

# Create a virtual environment
RUN python3 -m venv venv

# Copy requirements
COPY requirements_3_12.txt requirements-worker.txt ./

# Activate the virtual environment and install requirements
RUN . venv/bin/activate && \
    pip install --no-cache-dir -r requirements_3_12.txt -r requirements-worker.txt

# Copy application code
COPY . .

# Set Python path and virtual environment path
ENV PYTHONPATH=/app VIRTUAL_ENV=/app/venv PATH="/app/venv/bin:$PATH"

# Worker target
FROM base as worker
CMD ["celery", "-A", "gpuphot_worker.worker_app", "worker", "--loglevel=info"]

# Lab target
FROM base as lab
RUN . venv/bin/activate && pip install --no-cache-dir jupyter jupyterlab
WORKDIR /home/jovyan
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]

# Flower target
FROM base as flower
CMD ["celery", "-A", "gpuphot_worker.worker_app", "flower"]


# Descargar e instalar Q3C
FROM postgres:15.0-alpine as q3c_postgres

# Instalar las dependencias necesarias
RUN apk add --no-cache make gcc g++ postgresql-dev wget tar \
    zstd-dev lz4-dev openssl-dev krb5-dev zlib-dev

# Descargar e instalar Q3C
RUN wget https://github.com/segasai/q3c/archive/refs/tags/v2.0.0.tar.gz \
    && tar -xzf v2.0.0.tar.gz \
    && cd q3c-2.0.0 \
    && make \
    && make install

# Limpiar
RUN apk del make gcc g++ postgresql-dev wget tar
