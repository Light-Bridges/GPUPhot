# Base image with CUDA
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-distutils python3.11-dev && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    rm get-pip.py

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
