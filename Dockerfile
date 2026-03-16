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

# Optional: force a specific Python version instead of auto-detecting.
# When set (e.g., FORCE_PYTHON_VERSION=3.8), the auto-detection is skipped
# and only that version is installed via deadsnakes PPA.
ARG FORCE_PYTHON_VERSION=""

# If FORCE_PYTHON_VERSION is set, install it and create the venv immediately,
# skipping the entire auto-detection block below.
RUN if [ -n "${FORCE_PYTHON_VERSION}" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa -y && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            "python${FORCE_PYTHON_VERSION}" \
            "python${FORCE_PYTHON_VERSION}-dev" \
            "python${FORCE_PYTHON_VERSION}-venv" \
            "python${FORCE_PYTHON_VERSION}-distutils" && \
        update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${FORCE_PYTHON_VERSION}" 200 && \
        update-alternatives --install /usr/bin/python  python  "/usr/bin/python${FORCE_PYTHON_VERSION}" 200 && \
        python3 -m venv "$VIRTUAL_ENV" && \
        apt-get clean && rm -rf /var/lib/apt/lists/* && \
        echo "Forced Python ${FORCE_PYTHON_VERSION} installed and venv created." ; \
    fi

# Install Python via auto-detection (only runs if FORCE_PYTHON_VERSION is empty)
RUN if [ -n "${FORCE_PYTHON_VERSION}" ]; then echo "Skipping auto-detection (forced=${FORCE_PYTHON_VERSION})." ; exit 0 ; fi && \
    \
    # --- Configuration ---
    PYTHON_VERSIONS_TO_TRY="3.12 3.11 3.10" && \
    MIN_PYTHON_VERSION="3.10" && \
    # Define the specific fallback version
    FALLBACK_PYTHON_VERSION="3.8" && \
    TARGET_PYTHON_VERSION="" && \
    INSTALL_NEEDED=false && \
    PIP_INSTALLED_FOR_TARGET=false && \
    # Ensure software-properties-common is installed for add-apt-repository
    apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    \
    # --- Detect current Python version ---
    current_py_version=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))" 2>/dev/null || echo "0.0") && \
    echo "Detected current Python version: $current_py_version" && \
    echo "Python versions to try: $PYTHON_VERSIONS_TO_TRY" && \
    echo "Minimum required Python version: $MIN_PYTHON_VERSION" && \
    echo "Fallback Python version: $FALLBACK_PYTHON_VERSION" && \
    \
    # --- Initial version check ---
    if [ "$(printf '%s\n' "$MIN_PYTHON_VERSION" "$current_py_version" | sort -V | head -n1)" != "$MIN_PYTHON_VERSION" ]; then \
        echo "Current Python ($current_py_version) is lower than the minimum required ($MIN_PYTHON_VERSION). Attempting to install a newer version." ; \
        INSTALL_NEEDED=true ; \
    else \
        echo "Current Python ($current_py_version) meets or exceeds the minimum required ($MIN_PYTHON_VERSION). Using current version." ; \
        TARGET_PYTHON_VERSION=$(echo $current_py_version | cut -d. -f1,2) ; \
        # Install required packages for the system's python3
        echo "Installing supporting packages for system Python $TARGET_PYTHON_VERSION (python3-dev, python3-pip, python3-venv)..." ; \
        apt-get update && apt-get install -y --no-install-recommends \
            python3-dev \
            python3-pip \
            python3-venv && \
        PIP_INSTALLED_FOR_TARGET=true && \
        apt-get clean && rm -rf /var/lib/apt/lists/* ; \
        # Set system python as the default via alternatives with lower priority
        echo "Configuring alternatives for system Python $TARGET_PYTHON_VERSION." ; \
        update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${TARGET_PYTHON_VERSION}" 40 && \
        update-alternatives --install /usr/bin/python python "/usr/bin/python${TARGET_PYTHON_VERSION}" 40 ; \
    fi && \
    \
    # --- Attempt installation of preferred versions (if needed) ---
    if [ "$INSTALL_NEEDED" = true ]; then \
        echo "Adding deadsnakes PPA and updating apt..." ; \
        # The PPA should already be added if software-properties-common was installed earlier, but `-y` makes it safe
        add-apt-repository ppa:deadsnakes/ppa -y && \
        apt-get update && \
        \
        # Loop to try installing preferred versions
        for version in $PYTHON_VERSIONS_TO_TRY; do \
            echo "Trying to install Python $version..." ; \
            if apt-cache show "python$version" > /dev/null 2>&1; then \
                echo "Python $version found for $(dpkg --print-architecture). Installing..." ; \
                apt-get install -y --no-install-recommends \
                    "python$version" \
                    "python$version-dev" \
                    "python$version-venv" \
                    "python$version-distutils" && \
                if "/usr/bin/python$version" --version > /dev/null 2>&1; then \
                    echo "Configuring alternatives for Python $version." ; \
                    # Give high priority to preferred versions
                    update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python$version" 100 && \
                    update-alternatives --install /usr/bin/python python "/usr/bin/python$version" 100 ; \
                    # Note: pythonX.Y-venv should provide pip implicitly for venv creation
                    TARGET_PYTHON_VERSION="$version" && \
                    PIP_INSTALLED_FOR_TARGET=true && \
                    echo "Python $version installed and configured successfully." ; \
                    # Exit loop if a version was installed successfully
                    break ; \
                else \
                    echo "ERROR: Failed to install or verify Python $version." ; \
                fi \
            else \
                echo "Python $version not available in repositories for $(dpkg --print-architecture)." ; \
            fi ; \
        done ; \
        # Clean apt cache after the installation loop
        apt-get clean && rm -rf /var/lib/apt/lists/* ; \
    fi && \
    \
    # --- Fallback and Final Error Block ---
    # This block only runs if INSTALL_NEEDED was true AND the loop above failed
    if [ "$INSTALL_NEEDED" = true ] && [ -z "$TARGET_PYTHON_VERSION" ]; then \
        echo "WARN: Could not install any of the preferred Python versions ($PYTHON_VERSIONS_TO_TRY)." ; \
        echo "Attempting to install fallback Python version $FALLBACK_PYTHON_VERSION as a last resort..." ; \
        # Ensure apt is updated before the fallback attempt
        apt-get update && \
        version=$FALLBACK_PYTHON_VERSION ; \
        if apt-cache show "python$version" > /dev/null 2>&1; then \
            echo "Fallback Python $version found for $(dpkg --print-architecture). Installing..." ; \
            apt-get install -y --no-install-recommends \
                "python$version" \
                "python$version-dev" \
                "python$version-venv" \
                "python$version-distutils" && \
            if "/usr/bin/python$version" --version > /dev/null 2>&1; then \
                echo "Configuring alternatives for fallback Python $version." ; \
                # Give the fallback a medium priority
                update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python$version" 50 && \
                update-alternatives --install /usr/bin/python python "/usr/bin/python$version" 50 ; \
                # Note: pythonX.Y-venv should provide pip implicitly for venv creation
                TARGET_PYTHON_VERSION="$version" && \
                PIP_INSTALLED_FOR_TARGET=true && \
                echo "Fallback Python $version installed and configured successfully." ; \
            else \
                echo "ERROR: Failed to install or verify fallback Python $version." ; \
            fi \
        else \
            echo "Fallback Python $version not available in repositories for $(dpkg --print-architecture)." ; \
        fi ; \
        # Clean apt cache after the fallback attempt
        apt-get clean && rm -rf /var/lib/apt/lists/* ; \
    fi && \
    \
    # --- Final Check ---
    # This check runs regardless of method. It ensures we have *some* target version.
    if [ -z "$TARGET_PYTHON_VERSION" ]; then \
        echo "CRITICAL ERROR: Failed to install any required Python version." >&2 ; \
        exit 1 ; \
    fi && \
    # --- Final Verification and Venv Creation ---
    echo "Final check of the active Python version pointed to by 'python3':" && \
    python3 --version && \
    echo "Final check of the active Python version pointed to by 'python':" && \
    python --version && \
    \
    # Verify pip is *available* via the selected python3, but DO NOT UPGRADE IT GLOBALLY
    if [ "$PIP_INSTALLED_FOR_TARGET" = true ]; then \
        echo "Verifying system pip availability for Python $TARGET_PYTHON_VERSION..." ; \
        if python3 -m pip --version > /dev/null 2>&1; then \
           echo "System pip command found via 'python3 -m pip'. Proceeding to venv creation." ; \
           # DO NOT UPGRADE SYSTEM PIP HERE: python3 -m pip install --upgrade pip ; \
        else \
           echo "WARN: System pip command not directly found via 'python3 -m pip' even though expected. Venv creation might still succeed using bundled tools." ; \
        fi \
    else \
        echo "Skipping system pip check as it wasn't explicitly installed system-wide for this target." ;\
    fi && \
    \
    echo "Creating virtual environment: $VIRTUAL_ENV with Python $TARGET_PYTHON_VERSION" && \
    # Ensure VIRTUAL_ENV variable is defined previously, e.g., ENV VIRTUAL_ENV=/opt/venv
    # The 'venv' module, provided by pythonX.Y-venv or python3-venv, will install pip inside the venv.
    python3 -m venv "$VIRTUAL_ENV" && \
    echo "Virtual environment created successfully."

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
ARG REQUIREMENTS_FILE=requirements-312.txt
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

COPY initialize_notebooks.sh /usr/local/bin/initialize_notebooks.sh
RUN chmod +x /usr/local/bin/initialize_notebooks.sh

WORKDIR /home/jovyan

ENTRYPOINT ["/usr/local/bin/initialize_notebooks.sh"]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--ServerApp.token=''", "--notebook-dir=/home/jovyan/work"]
#CMD []

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
    && sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config \
    && sed -i 's/#MaxAuthTries 6/MaxAuthTries 20/' /etc/ssh/sshd_config

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
