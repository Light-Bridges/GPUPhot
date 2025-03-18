# Running GPUPhot with Docker Compose

GPUPhot can be deployed using Docker Compose for distributed processing, easy management of services, and a reproducible
environment. This guide explains how to set up and run GPUPhot with Docker Compose.

## Prerequisites

* **Docker and Docker Compose:** Ensure that you have both Docker and Docker Compose installed on your system. Refer to
  the official Docker documentation for installation instructions for your specific operating system. This setup has
  been tested with:
    * Docker version 20.10 or later.
    * Docker Compose version 1.29 or later (using the Compose file format version 2 or higher).
* **NVIDIA GPU and Drivers (If using GPU acceleration):**
    * An NVIDIA GPU with compute capability 6.0 or higher.
    * NVIDIA drivers installed and configured correctly. Version 525.x or later is recommended. You should be able to
      run `nvidia-smi` and see your GPU listed.
* **Git:** To clone the repository.

## 1. Clone the Repository

First, clone the GPUPhot repository from GitHub:

```bash
git clone https://github.com/Light-Bridges/GPUPhot.git
cd GPUPhot
```

## 2. Environment Variables (`.env`)

Create a `.env` file in the *root* of the GPUPhot project (the same directory as `docker-compose.yml`). This file will
contain environment variables that configure GPUPhot and its services.

**Important:** The `.env` file is *optional* if you want to use all the default values. However, you *must* set
`ASTROMETRY_CACHE_PATH` to a valid path on your host machine.

Here's a complete example `.env` file with *all* available options and their default values. You only need to include
the variables you want to *change* from their defaults.

```dotenv
#--------------------------------------------------
#  Library Settings
#--------------------------------------------------

# Enable debug mode (True/False).  More verbose logging.
GPUPHOT_DEBUG=True

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
GPUPHOT_LOG_LEVEL=DEBUG

# Environment (development, production)
GPUPHOT_ENVIRONMENT=development

#--------------------------------------------------
#  Astrometry Settings
#--------------------------------------------------

# Path on the *host* machine where astrometry index files will be stored.
# This directory MUST exist and be writable.
ASTROMETRY_CACHE_PATH=/path/to/your/astrometry_cache  # ***CHANGE THIS***

#--------------------------------------------------
#  Worker Settings
#--------------------------------------------------

# Name of the instrument configuration to use (corresponds to a .json file in INSTRUMENT_CONFIG_PATH).
INSTRUMENT_NAME=default

# Base path on the *host* machine where instrument configuration files are located.
INSTRUMENT_CONFIG_PATH=/path/to/your/instrument_configs  #  ***CHANGE THIS if not using the default***

# Base path on the *host* machine where astronomical images are located.
IMAGE_PATH=/path/to/your/images #  ***CHANGE THIS if not using the default***

# Number of Celery worker processes to run per GPU.  Usually 1 is sufficient.
CELERY_CONCURRENCY=1

#--------------------------------------------------
#  Database Settings
#--------------------------------------------------

# Password for the PostgreSQL database.  Change this for production!
POSTGRES_PASSWORD=gpuphot

#--------------------------------------------------
#  Logging Settings (Optional - for Logstash integration)
#--------------------------------------------------

# Enable sending logs to Logstash (True/False).
LOGSTASH_LOGGING=False  # Change to True to enable

# Hostname or IP address of your Logstash server.
LOGSTASH_HOST=localhost

# Port of your Logstash server.
LOGSTASH_PORT=5000
```

**Explanation of Environment Variables:**

* **`GPUPHOT_DEBUG`:** Enables verbose logging for debugging. Set to `False` for normal operation.
* **`GPUPHOT_LOG_LEVEL`:** Sets the logging level. Use `DEBUG` for detailed logs, `INFO` for general information,
  `WARNING` for warnings, `ERROR` for errors, and `CRITICAL` for critical errors.
* **`GPUPHOT_ENVIRONMENT`:** Sets the environment, for example 'development' for local use, and 'production' in the
  deployment.
* **`ASTROMETRY_CACHE_PATH`:**  *Crucially*, this is the directory on your *host* machine where the large astrometry
  index files will be downloaded and stored. You *must* set this to a valid, writable directory. This directory should
  have *plenty of free space* (tens of GB).
* **`INSTRUMENT_NAME`:** The name of the instrument configuration to use. This corresponds to a JSON file (e.g.,
  `default.json`, `my_instrument.json`) in the `INSTRUMENT_CONFIG_PATH`.
* **`INSTRUMENT_CONFIG_PATH`:**  The directory on your *host* machine where your instrument configuration files are
  located. The default is usually fine.
* **`IMAGE_PATH`:** The base directory on your *host* machine where your astronomical images are located.
* **`CELERY_CONCURRENCY`:**  The number of Celery worker processes to run *per GPU*. Usually, `1` is the optimal value,
  as most of the work is done on the GPU. Increasing this beyond 1 will likely *not* improve performance and may lead to
  memory issues.
* **`POSTGRES_PASSWORD`:** The password for the PostgreSQL database.  **Change this for a production environment!**
* **`LOGSTASH_LOGGING`:** Enables sending logs to a Logstash server. This is optional.
* **`LOGSTASH_HOST`:** The hostname or IP address of your Logstash server.
* **`LOGSTASH_PORT`:** The port of your Logstash server.

**Security Note:**  Do *not* commit your `.env` file to version control (e.g., Git), especially if it contains passwords
or other sensitive information. Add `.env` to your `.gitignore` file.

## 3. Starting Services

There are two ways to start the services: automatically using all available GPUs, or manually.

### 3.A. Automatic Multi-GPU Setup (Recommended)

This method uses the `launch_gpuphot.sh` script to automatically detect the available GPUs and configure the Celery
workers accordingly.

1. **Make sure `launch_gpuphot.sh` is executable:**

   ```bash
   chmod +x launch_gpuphot.sh
   ```

2. **Run the script:**

   ```bash
   ./launch_gpuphot.sh [max_gpus]
   ```

    * `[max_gpus]` (optional):  The maximum number of GPUs to use. If omitted, all available GPUs will be used. Example:
      `./launch_gpuphot.sh 2` will use at most 2 GPUs.

   This script will:

    * Detect the number of available NVIDIA GPUs using `nvidia-smi`.
    * Start Docker Compose with the appropriate number of `gpuphot_worker` services (one per GPU, up to `max_gpus`).
    * Set the `GPU_ID` environment variable for each worker to assign it to a specific GPU.

### 3.B. Manual Setup

If you don't want to use the automatic script, or if you want to control the setup more precisely, you can start the
services manually:

1. **Ensure you have a `.env` file (if you need to override default values).**

2. **Run Docker Compose:**
   ```bash
    docker compose up -d
   ```

   This will start all services defined in `docker-compose.yml` in detached mode (in the background).

## 4. Accessing Services

Once the services are running, you can access them through the following URLs:

* **JupyterLab:**  `http://localhost:8888` (No login required by default)
* **Flower (Celery Monitor):** `http://localhost:5555` (No authentication by default)
* **RabbitMQ Management:** `http://localhost:15672` (Default credentials: `gpuphot` / `gpuphot`)

## 5. Stopping Services

To stop all services, run:

```bash
docker compose down
```

This will stop and remove the containers, networks, and volumes defined in `docker-compose.yml`.

## Troubleshooting

* **GPU Issues:**
    * Ensure your NVIDIA drivers are installed and working correctly: `nvidia-smi`
    * Check that Docker and Docker Compose are configured for GPU support. See the official Docker documentation for GPU
      support.
    * Make sure your `docker-compose.yml` file is correctly configured to use GPUs (see the `deploy` section for the
      `gpuphot_worker` service).
* **Permission errors:** Make sure you have the right permissions on the folders defined in `.env`
* **Port Conflicts:** If you have other services running on your machine that use the same ports as GPUPhot (e.g.,
  another JupyterLab instance), you'll need to change the ports in `docker-compose.yml` or stop the conflicting
  services.
* **Out of Memory:** Reduce `CELERY_CONCURRENCY`.

## Services Overview

| Service              | Description                                                                   | Port(s)         | Default Credentials        |
| :------------------- | :---------------------------------------------------------------------------- | :-------------- | :------------------------- |
| `rabbitmq`           | Message broker for Celery.  Handles task distribution to workers.            | 5672, 15672    | gpuphot / gpuphot          |
| `redis`              | Result backend for Celery.  Stores task results temporarily.                 | 6379            | (no authentication)       |
| `celery_beat`        | Celery Beat scheduler.  Schedules periodic tasks.                            | -               | -                          |
| `gpuphot_worker`    | Celery worker that processes astronomical images using the GPU.                | -               | -                          |
| `lab`                | JupyterLab interactive development environment.                               | 8888            | (no token required)       |
| `flower`             | Celery monitoring tool.  Provides a web interface to monitor tasks and workers. | 5555            | (no authentication)       |
| `postgres`           | PostgreSQL database.  Stores image metadata and photometry results.        | 5432            | admin / gpuphot (CHANGE THIS!) |

**Note:** All credentials mentioned are default values.  **You should change these, especially the `POSTGRES_PASSWORD`,
in a production environment.**

## Customizing the Setup

* **`docker-compose.yml`:**  You can modify the `docker-compose.yml` file to:
    * Change port mappings.
    * Add or remove services.
    * Adjust resource limits (CPU, memory).
    * Mount additional volumes.
* **`launch_gpuphot.sh`:**  You can modify this script to:
    * Change the logic for determining the number of workers.
    * Add additional options.

By making these changes, the documentation becomes much more complete and helpful.
