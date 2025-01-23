# Installation Guide

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- Docker and Docker Compose (for distributed processing)

## Installing GPUPhot

1. Install system dependencies:
   ```
   bash install.sh
   ```

2. Install GPUPhot:
   ```
   pip install .
   ```

3. Install additional Python dependencies if needed:
   ```
   pip install -r requirements-worker.txt -r requirements_3_12.txt
   ```

## Setting up Docker Compose

1. Create a `.env` file with necessary environment variables (see [DOCKER.md](DOCKER.md) for details).
2. Run Docker Compose:
   ```
   docker compose up -d
   ```

3. Access services:
    - JupyterLab: `http://localhost:8888`
    - Flower (Celery Monitor): `http://localhost:5555`

4. To stop the containers:
   ```
   docker compose down
   ```

## Post-Installation Setup

After installing GPUPhot, it's crucial to set up the astrometry index files. See [USAGE.md](USAGE.md#astrometry-setup)
for instructions on how to download these files.

For more details on deployment and configuration, see [DOCKER.md](DOCKER.md).