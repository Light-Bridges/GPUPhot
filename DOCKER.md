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
