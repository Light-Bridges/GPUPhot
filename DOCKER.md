# Running GPUPhot with Docker Compose

GPUPhot can be deployed using Docker Compose for distributed processing and easy management of services.

## Prerequisites

- Install Docker and Docker Compose on your system.
- Clone this repository.

## Environment Variables (`.env`)

Create a `.env` file in the project root with the following variables:

```
# Library settings
GPUPHOT_DEBUG=True
GPUPHOT_LOG_LEVEL=DEBUG

# Astrometry settings
ASTROMETRY_CACHE_PATH=/path/to/astrometry_cache

# Worker settings
INSTRUMENT_NAME=default
INSTRUMENT_CONFIG_PATH=/path/to/instrument_configs
IMAGE_PATH=/path/to/images

# Database settings
POSTGRES_PASSWORD=gpuphot

# Logging settings (optional)
LOGSTASH_LOGGING=True
LOGSTASH_HOST=10.0.210.30  # Replace with your host IP or domain name.
LOGSTASH_PORT=5000         # Replace with your desired port.
```

## Starting Services with Docker Compose

1. Build and start all services:
   ```
   docker compose up -d
   ```

2. Access services:
    - **JupyterLab**: `http://localhost:8888`
    - **Flower (Celery Monitor)**: `http://localhost:5555`
    - **RabbitMQ Management**: `http://localhost:15672`

3. To stop all services:
   ```
   docker compose down
   ```

# Running GPUPhot with Docker Compose

## Automatic Multi-GPU Setup

GPUPhot includes a script that automatically detects the number of available GPUs and sets up the appropriate number of
workers. To use this feature:

1. Ensure you have the `launch_gpuphot.sh` script in your project root.
2. Make the script executable:
   ```
   chmod +x launch_gpuphot.sh
   ```
3. Run the script:
   ```
   ./launch_gpuphot.sh [max_gpus]
   ```
   You can optionally specify the maximum number of GPUs to use. If not specified, it will use all available GPUs.

This script will:

- Detect the number of available GPUs
- Launch Docker Compose with the correct number of workers (up to the specified maximum)
- Assign specific GPUs to each worker

### How it works

The `launch_gpuphot.sh` script:

1. Counts the number of available GPUs using `nvidia-smi`.
2. Checks if any GPUs are detected and exits if none are found.
3. Determines the number of workers based on available GPUs and the optional maximum specified.
4. Scales the `gpuphot_worker` service in Docker Compose to match the worker count.
5. Assigns each worker to a specific GPU by setting the `GPU_ID` environment variable.

## Manual Setup

If you prefer to set up your environment manually or need more control over the configuration, follow these steps:

1. Create a `.env` file in the project root with necessary environment variables.
2. Run Docker Compose:
   ```
   docker compose up -d
   ```
3. Access services:
    - JupyterLab: http://localhost:8888
    - Flower (Celery Monitor): http://localhost:5555

## Customizing the Setup

You can modify the `launch_gpuphot.sh` script to suit your specific needs. For example, you can limit the number of GPUs
used by passing an argument:

```
./launch_gpuphot.sh 2  # Use a maximum of 2 GPUs
```

## Troubleshooting

If you encounter issues with the automatic setup:

- Ensure NVIDIA drivers are properly installed and `nvidia-smi` is working correctly.
- Check that Docker and Docker Compose are installed and configured for GPU support.
- Verify that your Docker Compose file is set up to use GPUs (usually with the `deploy` section specifying GPU
  requirements).

For more detailed information on GPU allocation and Docker, refer to the official Docker documentation on GPU support.

## Services Overview

### RabbitMQ (Message Broker)

Handles task communication between Celery workers.

### Redis (Result Backend)

Stores task results temporarily.

### Celery Workers (`gpuphot_worker`)

Processes tasks such as image processing.

### PostgreSQL (`postgres`)

Stores processed image statistics and metadata.

### JupyterLab (`lab`)

Provides an interactive environment for running notebooks.

### Flower (`flower`)

Monitors Celery tasks in real-time.
