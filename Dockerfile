# Base image
ARG BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04

FROM ${BASE_IMAGE} AS base

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
        wget \
        build-essential \
        gcc \
        g++ \
        ca-certificates \
        pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalar Python y pip
RUN \
    current_py_version=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))" 2>/dev/null || echo "0.0") && \
    echo "Versión actual de Python detectada: $current_py_version" && \
    \
    PYTHON_VERSIONS_TO_TRY="3.12 3.11 3.10 3.8" && \
    MIN_PYTHON_VERSION="3.8" && \
    TARGET_PYTHON_VERSION="" && \
    INSTALL_NEEDED=false && \
    \
    if [ "$(printf '%s\n' "$MIN_PYTHON_VERSION" "$current_py_version" | sort -V | head -n1)" != "$MIN_PYTHON_VERSION" ]; then \
        echo "Actual Python $current_py_version is lower than the minimum required $MIN_PYTHON_VERSION. Trying to install a newer version." ; \
        INSTALL_NEEDED=true ; \
    else \
        echo "Actual Python $current_py_version is higher than the minimum required $MIN_PYTHON_VERSION. No se instalará otra versión." ; \
        TARGET_PYTHON_VERSION=$(echo $current_py_version | cut -d. -f1,2) ; \
        apt-get update && apt-get install -y --no-install-recommends python3-dev python3-pip python3-venv && apt-get clean && rm -rf /var/lib/apt/lists/* ; \
    fi && \
    \
    if [ "$INSTALL_NEEDED" = true ]; then \
        echo "Adding deadsnakes PPA and updating apt..." ; \
        add-apt-repository ppa:deadsnakes/ppa -y && \
        apt-get update && \
        \
        for version in $PYTHON_VERSIONS_TO_TRY; do \
            echo "Trying to install Python $version..." ; \
            if apt-cache show "python$version" > /dev/null 2>&1; then \
                echo "Found Python $version for $(dpkg --print-architecture). Installing..." ; \
                apt-get install -y --no-install-recommends \
                    "python$version" \
                    "python$version-dev" \
                    "python$version-venv" \
                    "python$version-distutils" && \
                if "/usr/bin/python$version" --version > /dev/null 2>&1; then \
                    echo "Configure alternatives for Python $version." ; \
                    update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python$version" 2 && \
                    update-alternatives --install /usr/bin/python python "/usr/bin/python$version" 2 && \
                    ln -sf "/usr/bin/python$version" /usr/bin/python3 && \
                    TARGET_PYTHON_VERSION="$version" && \
                    echo "Python $version instalado y configurado exitosamente." ; \
                    break ; \
                else \
                    echo "ERROR: Failed to install or verify Python $version." ; \
                    # apt-get remove -y "python$version" "python$version-dev" "python$version-venv" "python$version-distutils"; apt-get autoremove -y;
                fi \
            else \
                echo "Python $version not available in the repositories for $(dpkg --print-architecture)." ; \
            fi ; \
        done ; \
        apt-get clean && rm -rf /var/lib/apt/lists/* ; \
    fi && \
    \
    if [ -z "$TARGET_PYTHON_VERSION" ]; then \
        echo "ERROR: Impossible install any of the required Python versions ($PYTHON_VERSIONS_TO_TRY) and the base version ($current_py_version) is lower than the minimum ($MIN_PYTHON_VERSION)." >&2 ; \
        exit 1 ; \
    fi && \
    \
    echo "Verify final of the active Python version:" && \
    python3 --version && \
    \
    echo "Virtual environment creation: $VIRTUAL_ENV with Python $TARGET_PYTHON_VERSION" && \
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
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE} --extra-index-url=https://pypi.nvidia.com


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
