# Base image
ARG BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04

FROM ${BASE_IMAGE} AS base

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    PYTHONPATH=/app

# Install Python and pip with version check
RUN apt-get update && apt-get upgrade -y && \
    # Instalar dependencias base
    apt-get install --no-install-recommends -y \
    software-properties-common curl python3 build-essential gcc g++ && \
    # Verificar versión de Python y actualizar si es necesario
    { \
    current_py_version=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))" 2>/dev/null || echo "0.0") && \
    if [ "$(printf '%s\n' "3.8" "$current_py_version" | sort -V | head -n1)" != "3.8" ]; then \
        echo "Instalando Python 3.8"; \
        add-apt-repository ppa:deadsnakes/ppa -y && \
        apt-get update && \
        apt-get install -y python3.8 python3.8-dev python3.8-venv && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
        update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2 && \
        ln -sf /usr/bin/python3.8 /usr/bin/python3; \
    fi; \
    } && \
    # Instalar paquetes específicos para Python 3.8
    apt-get install -y --no-install-recommends \
    # python3-scipy \
    # python3-sklearn \
    # python3-matplotlib \
    # python3-pytest-astropy \
    python3-dev \
    python3-pip \
    python3-venv && \
    # Limpiar caché
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Configurar entorno virtual
    python3 -m venv $VIRTUAL_ENV


# Set working directory
WORKDIR /app

# Create a virtual environment
RUN python3 -m venv venv

# Activate the virtual environment and install requirements
RUN pip install --upgrade pip setuptools wheel pipenv

# Copy requirements
COPY requirements-worker.txt ./
RUN pip install --no-cache-dir -r requirements-worker.txt

# Copy requirements
ARG REQUIREMENTS_FILE=requirements_3_12.txt
COPY ${REQUIREMENTS_FILE} ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

## Copy application code
#COPY . .

## Set Python path and virtual environment path
#ENV PYTHONPATH=/app VIRTUAL_ENV=/app/venv PATH="/app/venv/bin:$PATH"

# Worker target
FROM base AS worker

# Copy application code
WORKDIR /app
COPY . .


CMD ["celery", "-A", "gpuphot_worker.worker_app", "worker", "--loglevel=info"]

# Lab target
FROM base AS lab
RUN . venv/bin/activate && pip install --no-cache-dir jupyter jupyterlab


# Copy application code
WORKDIR /app
COPY . .

WORKDIR /home/jovyan

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/home/jovyan/work"]

# Flower target
FROM base AS flower

# Copy application code
WORKDIR /app
COPY . .

CMD ["celery", "-A", "gpuphot_worker.worker_app", "flower"]


# Descargar e instalar Q3C
FROM postgres:15.0-alpine AS q3c_postgres

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


## Profiler target
FROM base AS profiler

# Activar el entorno virtual
ENV PATH="/app/venv/bin:$PATH"

# Install SSH server
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config

# Install rsyslog for SSH logs
RUN apt-get update && apt-get install -y rsyslog
RUN mkdir -p /var/log && touch /var/log/auth.log
RUN chown syslog:adm /var/log/auth.log && chmod 640 /var/log/auth.log
RUN echo "local5.* /var/log/sshd.log" >> /etc/rsyslog.conf
RUN echo "SyslogFacility LOCAL5" >> /etc/ssh/sshd_config
RUN echo "LogLevel VERBOSE" >> /etc/ssh/sshd_config


ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# Install NVIDIA Nsight Systems
RUN apt update && \
    apt install -y --no-install-recommends gnupg && \
    . /etc/os-release && \
    UBUNTU_VERSION=$(echo "$VERSION_ID" | tr -d '.') && \
    ARCH=$(dpkg --print-architecture) && \
    REPO_URL="https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_VERSION}/${ARCH}" && \
    # Descargar clave GPG oficial de CUDA (compatible universalmente)
    # Configurar repositorio
    echo "deb ${REPO_URL} /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    # Usar clave GPG oficial de CUDA (compatible universalmente)
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
#    # Metodo usando trusted.gpg.d
#    # Descargar clave manualmente
#    mkdir -p /etc/apt/keyrings && \
#    curl -fsSL "${REPO_URL}/nvidia.pub" | gpg --dearmor -o /etc/apt/keyrings/nvidia-dev-tools.gpg && \
#    # Configurar repositorio seguro
#    echo "deb [signed-by=/etc/apt/keyrings/nvidia-dev-tools.gpg] ${REPO_URL} /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    # Instalar
    apt-get update && \
    apt-get install -y nsight-systems && \
    # Instalar Nsight Systems CLI
    apt-get install -y nsight-systems-cli

# Instalar herramientas para módulos (solo Jetson)
ARG TARGET_ARCH
RUN if [ "$TARGET_ARCH" = "aarch64" ]; then \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        linux-headers-$(uname -r) \
        libncurses-dev && \
    rm -rf /var/lib/apt/lists/* ; \
    fi


# Create directory for profiling scripts
RUN mkdir -p /app/profiling_scripts /app/profiling_results

# Copy profiling scripts
COPY ./profiling_scripts/run_profiling.sh /app/profiling_scripts/
COPY ./profiling_scripts/profile_image_processing.py /app/profiling_scripts/
RUN chmod +x /app/profiling_scripts/run_profiling.sh

# Copy the rest of the code
WORKDIR /app
COPY . .

# Expose SSH port for remote profiling
ARG SSSH_PORT
EXPOSE ${SSSH_PORT}

# Set root password
ARG ROOT_PASSWORD
RUN echo "root:${ROOT_PASSWORD}" | chpasswd

# Crear script de inicio para capturar las variables de entorno
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'mkdir -p /root/.ssh' >> /entrypoint.sh && \
    echo 'env | grep -v "^HOSTNAME=" | grep -v "^PWD=" | grep -v "^HOME=" | grep -v "^TERM=" | grep -v "^SHLVL=" > /root/.ssh/environment' >> /entrypoint.sh && \
    echo 'echo "PATH=$PATH" >> /root/.ssh/environment' >> /entrypoint.sh && \
    echo '/usr/sbin/sshd -D' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
