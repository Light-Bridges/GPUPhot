# Installation Guide

## Requirements

Before installing GPUPhot, ensure you have the following:

*   **Python:**  Version 3.8, 3.9, 3.10, or 3.11 (3.9 recommended). Python 3.12 is supported with additional dependencies (see below).
*   **CUDA:** Version 11.x or 12.x (tested with 12.2).
*   **NVIDIA Driver:** Version 525.x or later.
*   **GPU:** NVIDIA GPU with compute capability 6.0 or later.  8GB of GPU memory recommended.
*   **Docker:** Version 20.10.x or later.
*   **Docker Compose:** Version 1.29.x or later (version 2).
*   **Operating System:**  Linux (tested on Ubuntu 20.04 and 22.04). Windows support via WSL2. macOS support is experimental.
*   **RAM:** 16GB or more recommended.

**It is highly recommended to use a Python virtual environment (venv or conda) to isolate GPUPhot's dependencies.**

**Creating a virtual environment (venv):**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
.venv\\Scripts\\activate  # On Windows
```

## Installation

1.  **Install System Dependencies:**

    The `install.sh` script attempts to install some required system packages. **Review this script *before* running it to understand the changes it will make to your system.**

    ```bash
    bash install.sh
    ```
    **Alternative (Manual Installation):** If you prefer to install system dependencies manually, consult the `install.sh` script for the list of required packages. The specific packages may vary depending on your Linux distribution.

2.  **Install GPUPhot:**

    Clone the GPUPhot repository and install it using `pip`.  Run these commands from the root directory of the GPUPhot project (where `setup.py` is located):

    ```bash
    git clone https://github.com/Light-Bridges/GPUPhot.git
    cd GPUPhot
    pip install .
    ```
    Use `pip install -e .` for development installations (editable mode).

3.  **Install Additional Dependencies (Optional):**

    *   **For Celery (distributed processing):**
        ```bash
        pip install .[worker]
        # OR
        # pip install -r requirements-worker.txt
        ```

    *   **For Python 3.12:**
        ```bash
        pip install .[py312]
        # OR
        # pip install -r requirements-312.txt
        ```

    *  **For development (testing, documentation):**
        ```bash
        pip install .[dev]
        ```

## Setting up Docker Compose (Optional - for Distributed Processing)

1.  **Create a `.env` file:**

    Create a file named `.env` in the project root directory.  This file contains environment variables that configure GPUPhot.  See [DOCKER.md](DOCKER.md) for a detailed explanation of each variable.  *If you don't create a `.env` file, default values will be used.*

    **Example `.env` file:**
    ```
    # Library settings (Optional, defaults shown)
    GPUPHOT_DEBUG=True
    GPUPHOT_LOG_LEVEL=DEBUG
    GPUPHOT_ENVIRONMENT=development

    # Astrometry settings (REQUIRED)
    ASTROMETRY_CACHE_PATH=/path/to/astrometry_cache  # **CHANGE THIS**

    # Worker settings (Optional, defaults shown)
    INSTRUMENT_NAME=default
    INSTRUMENT_CONFIG_PATH=./gpuphot/instrument_configs
    IMAGE_PATH=~/gpuphot_images

    # Database settings (Optional, defaults shown)
    POSTGRES_PASSWORD=gpuphot

    # Logging settings (optional)
    LOGSTASH_LOGGING=False # Change to True to enable
    LOGSTASH_HOST=localhost
    LOGSTASH_PORT=5000
    ```
    **Important:**  `ASTROMETRY_CACHE_PATH`, `INSTRUMENT_CONFIG_PATH`, and `IMAGE_PATH` must be paths on your *host* machine.

2.  **Run Docker Compose:**

    From the project root directory (where `docker-compose.yml` is located), run:
    ```bash
    docker compose up -d
    ```

3.  **Access Services:**
    *   JupyterLab: `http://localhost:8888`
    *   Flower (Celery Monitor): `http://localhost:5555`
    *   RabbitMQ Management: http://localhost:15672

4.  **Stop Containers:**

    ```bash
    docker compose down
    ```

## Post-Installation Setup

**Download Astrometry Index Files (REQUIRED):**

Before you can use GPUPhot for astrometric calibration, you *must* download the necessary index files.  Run the following Python code *once*:

```python
from gpuphot.utils.astro import get_solver

get_solver()  # This will download the index files to ASTROMETRY_CACHE_PATH
```

This process may take a significant amount of time, depending on your internet connection.

## Troubleshooting

*   **CUDA Errors:** If you encounter CUDA errors, make sure your NVIDIA drivers are correctly installed and that your GPU is compatible with the installed CUDA version.
*   **ImportError:** If you get an `ImportError`, make sure you have activated your virtual environment (if you are using one) and that all required packages are installed.
* **Docker errors:** Verify docker is installed and configured correctly.

If you encounter any other problems, please refer to the [USAGE.md](USAGE.md) and [DOCKER.md](DOCKER.md) files, or open an issue on the GitHub repository.
```