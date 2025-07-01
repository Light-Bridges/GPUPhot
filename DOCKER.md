# Running GPUPhot with Docker Compose

GPUPhot can be deployed using Docker Compose for distributed processing, easy management of services, and a reproducible environment. This guide explains how to set up and run GPUPhot with Docker Compose.

## Service Architecture

The Docker setup uses a multi-stage `Dockerfile` to create optimized images for each service:

*   **`base`**: A base image containing all common dependencies and the Python virtual environment.
*   **`worker`**: Contains the code for the Celery workers that perform the GPU-based processing.
*   **`lab`**: A JupyterLab environment for interactive development, with GPU access.
*   **`flower`**: The Celery monitoring dashboard.
*   **`q3c_postgres`**: A custom PostgreSQL image that includes the **Q3C** extension for efficient spatial indexing of astronomical coordinates.
*   **`profiler`**: A specialized image for debugging and performance profiling with **NVIDIA Nsight Systems**, which includes an SSH server.

## Prerequisites

*   **Docker and Docker Compose:** Ensure you have both Docker and Docker Compose installed.
    *   Docker version 20.10 or later.
    *   Docker Compose v2 or later (which uses the `docker compose` command).
*   **NVIDIA GPU and Drivers (Required):**
    *   One or more NVIDIA GPUs with compute capability 6.0 or higher.
    *   **NVIDIA Container Toolkit** installed and configured to allow Docker to access GPUs.
    *   NVIDIA drivers version 525.x or later. Verify that `nvidia-smi` runs correctly.
*   **Git:** To clone the repository.

## 1. Clone the Repository

```bash
git clone https://github.com/Light-Bridges/GPUPhot.git
cd GPUPhot
```

## 2. Environment Variables (`.env`)

Create a `.env` file in the project's root directory. This file centralizes all configuration. You can copy the example below and **modify the paths and passwords** to match your setup.

**Complete `.env` Example:**

```dotenv
# ===================================================================
#  General & Docker Build Configuration
# ===================================================================

# Base NVIDIA/CUDA image for building the containers.
# Ensure it is compatible with your drivers and architecture.
# Example x86_64: nvidia/cuda:12.6.3-devel-ubuntu24.04
# Example Jetson: nvcr.io/nvidia/l4t-base:r35.4.1
BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04

# Python requirements file to be used during the build.
# This should match the Python version installed in the base image.
REQUIREMENTS_FILE=requirements_3_12.txt

# ===================================================================
#  GPUPhot Application Settings
# ===================================================================

# Enable more verbose logging (True/False).
GPUPHOT_DEBUG=True

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
GPUPHOT_LOG_LEVEL=DEBUG

# Execution environment (development, production).
GPUPHOT_ENVIRONMENT=development

# ===================================================================
#  Host Path Configuration
#  IMPORTANT: You must change these paths! They should be absolute paths.
# ===================================================================

# Path on the *host* machine where Astrometry.net index files will be cached (requires a lot of space).
ASTROMETRY_CACHE_PATH=/path/on/your/host/to/astrometry_cache

# Path on the *host* machine where instrument configuration files (.json) are located.
INSTRUMENT_CONFIG_PATH=/path/on/your/host/to/instrument_configs

# Path on the *host* machine where astronomical images to be processed are located.
IMAGE_PATH=/path/on/your/host/to/images

# Path on the *host* where JupyterLab notebooks and work files will be saved.
NOTEBOOKS_PATH=/path/on/your/host/to/jupyter_projects

# Path on the *host* where PostgreSQL database files will be stored.
POSTGRES_DATA_PATH=/path/on/your/host/to/postgres_data

# ===================================================================
#  Celery Worker Settings
# ===================================================================

# Name of the instrument to use (must correspond to a .json file in INSTRUMENT_CONFIG_PATH).
INSTRUMENT_NAME=default

# Number of worker processes per GPU. '1' is almost always the optimal value.
CELERY_CONCURRENCY=1

# ===================================================================
#  Services & Port Configuration
# ===================================================================

# --- PostgreSQL (Database) ---
# Password for the database 'admin' user. CHANGE IN PRODUCTION!
POSTGRES_PASSWORD=gpuphot
# Port exposed on the host for PostgreSQL.
POSTGRES_PORT=5432

# --- RabbitMQ (Message Broker) ---
# Port exposed on the host for the RabbitMQ management panel.
RABBITMQ_MANAGEMENT_PORT=15672
# Port exposed on the host for AMQP communication.
RABBITMQ_AMQP_PORT=5672

# --- Redis (Result Backend) ---
# Port exposed on the host for Redis.
REDIS_PORT=6379

# --- JupyterLab ---
# Port exposed on the host for JupyterLab.
JUPYTER_PORT=8888

# --- Flower (Celery Monitor) ---
# Port exposed on the host for Flower.
FLOWER_PORT=5555

# ===================================================================
#  Profiling & Debugging Configuration (Optional)
# ===================================================================

# SSH port for connecting to the profiling container.
PROFILER_SSSH_PORT=2222

# Root password for the SSH session in the profiling container.
ROOT_PASSWORD=gpuphot_profiler

# Path on the *host* where profiler results (Nsight reports) will be saved.
PROFILER_RESULTS_PATH=/path/on/your/host/to/profiling_results

# ===================================================================
#  External Logging Configuration (Optional)
# ===================================================================

# Enable sending logs to Logstash (True/False).
LOGSTASH_LOGGING=False
LOGSTASH_HOST=localhost
LOGSTASH_PORT=5000

# ===================================================================
#  API Keys (Optional)
# ===================================================================

# API key for the Astrometry.net online service (if used).
ASTROMETRY_API_KEY=your_api_key_here
```

**Security Note:** Do not commit your `.env` file to version control systems like Git. Add it to your `.gitignore` file.

## 3. Launching the Services

### 3.A. Standard Launch

To start all main services (database, workers, JupyterLab, etc.) in detached mode, run:

```bash
docker compose up -d
```

This command will use the default configuration for a single `gpuphot_worker` on the GPU with `ID=0`.

### 3.B. Multi-GPU Launch (Recommended for Multiple GPUs)

The `launch_gpuphot.sh` script is designed to detect and launch one `gpuphot_worker` container for each available GPU, assigning each to a specific GPU.

1.  **Make the script executable:**
    ```bash
    chmod +x launch_gpuphot.sh
    ```

2.  **Run the script:**
    ```bash
    ./launch_gpuphot.sh [max_gpus]
    ```
    *   `[max_gpus]` (optional): Limits the number of GPUs to use. If omitted, all detected GPUs will be used.
        *   Example: `./launch_gpuphot.sh 2` will use a maximum of 2 GPUs.

    This script will launch all base services and scale the `gpuphot_worker` service to match the desired number of GPUs.

### 3.C. Launching for Profiling and Debugging

The `docker-compose.yml` includes special `profiler` services for detailed performance analysis with NVIDIA Nsight Systems. These services are not started by default. To launch them, use the `debug` profile:

1.  **Launch profiling services (x86):**
    ```bash
    docker compose --profile debug up -d profiler
    ```
2.  **Launch profiling services (Jetson):**
    ```bash
    # For standard Jetson
    docker compose --profile debug up -d profiler_jetson

    # For Jetson Orin
    docker compose --profile debug up -d profiler_jetson_orin
    ```

Once running, you can connect to the container via SSH to run profiling tools:
```bash
ssh root@localhost -p ${PROFILER_SSSH_PORT:-2222}
# The password is the one defined by ROOT_PASSWORD in your .env file
```

## 4. Accessing Services

*   **JupyterLab:** `http://localhost:${JUPYTER_PORT:-8888}`
    *   *Your work files will be saved in the host directory specified by `NOTEBOOKS_PATH`.*
*   **Flower (Celery Monitor):** `http://localhost:${FLOWER_PORT:-5555}`
*   **RabbitMQ Management:** `http://localhost:${RABBITMQ_MANAGEMENT_PORT:-15672}` (user: `gpuphot`, pass: `gpuphot`)
*   **Profiler SSH:** Connect via SSH to port `${PROFILER_SSSH_PORT:-2222}` (see previous section).

## 5. Initializing the Environment (First-Time Setup)

After launching the services for the first time, you should initialize your JupyterLab environment by generating the example notebooks. These notebooks will guide you through the usage of GPUPhot.

**Run the following command in your terminal:**

```bash
docker compose exec lab /usr/local/bin/initialize_notebooks.sh
```

This script will:
*   Check if the notebooks have already been generated.
*   If not, it will create a set of example `.ipynb` files inside the directory you specified in `NOTEBOOKS_PATH`.
*   If you run it again, it will do nothing, preserving any changes you have made.

After running the command, refresh your JupyterLab browser window. The example notebooks should appear in the file browser.

## 6. Stopping the Services

*   **To stop standard services:**
    ```bash
    docker compose down
    ```

*   **To stop profiling services:**
    ```bash
    docker compose --profile debug down
    ```

This will stop and remove the containers, networks, and volumes.

## Services Overview

| Service                | Description                                                          | Default Port(s) | Default Credentials              | Profile     |
|:-----------------------|:---------------------------------------------------------------------|:----------------|:---------------------------------|:------------|
| `rabbitmq`             | Message broker for Celery.                                           | `15672`, `5672` | `gpuphot` / `gpuphot`            | `default`   |
| `redis`                | Result backend for Celery.                                           | `6379`          | (no auth)                        | `default`   |
| `postgres`             | PostgreSQL database with **Q3C** extension.                          | `5432`          | `admin` / `${POSTGRES_PASSWORD}` | `default`   |
| `beat`                 | Celery's periodic task scheduler.                                    | -               | -                                | `default`   |
| `gpuphot_worker`       | Celery worker that processes images on the GPU.                      | -               | -                                | `default`   |
| `lab`                  | Interactive JupyterLab environment with GPU access.                  | `8888`          | (no token)                       | `default`   |
| `flower`               | Web interface for monitoring Celery workers and tasks.               | `5555`          | (no auth)                        | `default`   |
| `profiler`             | Container with Nsight Systems and SSH for profiling on x86.          | `2222`          | `root` / `${ROOT_PASSWORD}`      | `debug`     |
| `profiler_jetson`      | Profiling container for Jetson platforms.                            | `2222`          | `root` / `${ROOT_PASSWORD}`      | `debug`     |
| `profiler_jetson_orin` | Profiling container for Jetson Orin platforms.                       | `2222`          | `root` / `${ROOT_PASSWORD}`      | `debug`     |

## Customization

*   **`docker-compose.yml`**: You can modify this file to adjust resource limits, add services, or change complex configurations.
*   **`.env`**: The easiest way to customize your setup is by modifying the variables in this file, especially ports and volume paths.
*   **Build Arguments**: You can change the base image (`BASE_IMAGE`) or the requirements file (`REQUIREMENTS_FILE`) during the `build` phase by editing the `.env` file.

## Troubleshooting

*   **GPU Errors (`CUDA_ERROR_NO_DEVICE`)**:
    *   Ensure the **NVIDIA Container Toolkit** is correctly installed.
    *   Verify that `nvidia-smi` works on the host machine.
    *   Check that the `BASE_IMAGE` variable in your `.env` is compatible with your system's architecture (x86_64 vs. aarch64/jetson).
*   **Volume Permission Errors**: Make sure the paths defined in your `.env` file (`ASTROMETRY_CACHE_PATH`, `IMAGE_PATH`, etc.) exist on your host machine and that the user running Docker has read/write permissions for them.
*   **Port Conflicts**: If a port is already in use, you can easily change it by modifying the corresponding variable in your `.env` file (e.g., `JUPYTER_PORT=8889`).