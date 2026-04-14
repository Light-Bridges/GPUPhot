# Installation Guide

## Requirements

Before installing GPUPhot, ensure you have the following:

*   **Python:** Version 3.8 or later (3.12 recommended for production with full GPU acceleration).
*   **NVIDIA GPU:** Compute capability 6.0 or later. 8 GB of VRAM recommended.
*   **NVIDIA Driver:** Version 525.x or later.
*   **CUDA Toolkit:** Version 11.x or 12.x (tested with 12.2 and 12.6).
*   **CuPy:** Installed and working for your CUDA version (see Step 1 below).
*   **Operating System:** Linux (tested on Ubuntu 20.04, 22.04, and 24.04). Windows support via WSL2.
*   **RAM:** 16 GB or more recommended.

Docker and Docker Compose are only required for distributed processing (see the Docker section below).

## Installation

It is highly recommended to use a Python virtual environment to isolate GPUPhot's dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 1: Install CuPy (prerequisite)

GPUPhot requires CuPy for GPU-accelerated array operations. The correct CuPy package depends on your CUDA version:

```bash
# For CUDA 12.x (recommended)
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

Verify that CuPy can access your GPU:

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.runtimeGetVersion()); print(cp.zeros(3))"
```

If this fails, consult the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

### Step 2: Install GPUPhot

From PyPI (when available):

```bash
pip install gpuphot
```

Or from source:

```bash
git clone https://github.com/Light-Bridges/GPUPhot.git
cd GPUPhot
pip install .
```

Use `pip install -e .` for development installations (editable mode).

### Step 3 (optional): Install RAPIDS cuML for GPU-accelerated cross-matching

RAPIDS cuML provides GPU-accelerated nearest-neighbor searches used for catalog cross-matching and star clustering. If not installed, GPUPhot falls back to CPU implementations (scipy KDTree, scikit-learn) automatically.

cuML is available for x86_64 with CUDA 11.x/12.x. It is **not available** on ARM/Jetson platforms.

```bash
pip install cuml-cu12 pylibraft-cu12 --extra-index-url=https://pypi.nvidia.com
```

After installing cuML, run the calibration tool once to find the optimal source-count
window for your GPU and set `GPUPHOT_CUML_MIN_SOURCES` / `GPUPHOT_CUML_MAX_SOURCES`
in your `.env`.  See [CUML_CALIBRATION.md](CUML_CALIBRATION.md) for the full procedure.

### Step 4 (optional): Install development tools

For running tests and building documentation:

```bash
pip install gpuphot[dev]
```

## Verify the Installation

```python
import gpuphot
from gpuphot.image_processor import create_processor

proc = create_processor('default')
print("GPUPhot is ready.")
```

## Post-Installation Setup

**Download Astrometry Index Files (REQUIRED):**

Before you can use GPUPhot for astrometric calibration, you must download the necessary index files. Run the following Python code once:

```python
from gpuphot.utils.astro import get_solver

get_solver()  # Downloads index files to ASTROMETRY_CACHE_PATH
```

This process may take a significant amount of time depending on your internet connection. Set the `ASTROMETRY_CACHE_PATH` environment variable to control where the files are stored.

## Deployment Modes

GPUPhot supports two deployment modes:

### Standalone Library (recommended for integration)

Import `gpuphot` directly in your Python scripts or pipeline. This is the simplest option and requires only Python, CuPy, and the core dependencies.

```python
from gpuphot.image_processor import create_processor

proc = create_processor('my_instrument')
df, header = proc.process_image(image_data, fits_header)
```

### Distributed Processing with Docker Compose

For high-throughput survey operations, GPUPhot can be deployed as a containerized service with Celery workers, RabbitMQ, PostgreSQL, and JupyterLab.

1.  **Create a `.env` file:**

    Create a file named `.env` in the project root. See [DOCKER.md](DOCKER.md) for all available variables. Example:

    ```
    ASTROMETRY_CACHE_PATH=/path/to/astrometry_cache
    INSTRUMENT_NAME=default
    INSTRUMENT_CONFIG_PATH=./gpuphot/instrument_configs
    IMAGE_PATH=~/gpuphot_images
    POSTGRES_PASSWORD=gpuphot
    ```

2.  **Run Docker Compose:**

    ```bash
    docker compose up -d
    ```

3.  **Access Services:**
    *   JupyterLab: `http://localhost:8888`
    *   Flower (Celery Monitor): `http://localhost:5555`
    *   RabbitMQ Management: `http://localhost:15672`

4.  **Stop:**

    ```bash
    docker compose down
    ```

## Platform-Specific Notes

### Pinned requirements for reproducible deployments

The `setup.py` uses flexible version ranges for broad compatibility. For reproducible Docker deployments, use the pinned requirements files:

| File | Environment |
|------|-------------|
| `requirements.txt` | x86_64, Python 3.8, CUDA 12.x |
| `requirements-312.txt` | x86_64, Python 3.12, CUDA 12.x |
| `requirements_jetson.txt` | Jetson ARM64, Python 3.8, CUDA 11.x |
| `requirements_jetson_312.txt` | Jetson ARM64, Python 3.12, CUDA 12.x |
| `requirements-worker.txt` | Additional Celery/Redis dependencies for Docker workers |

### Jetson / ARM64

On NVIDIA Jetson platforms, use `cupy-cuda11x` (JetPack 5.x) or `cupy-cuda12x` (JetPack 6.x). RAPIDS cuML is not available on ARM, so CPU fallbacks are used for cross-matching and clustering.

## Troubleshooting

*   **CuPy import fails:** Verify your NVIDIA driver and CUDA toolkit versions match the CuPy package. Run `nvidia-smi` to check the driver version.
*   **CUDA out of memory:** GPUPhot manages GPU memory with CuPy pools. For large images (>100 MP), 16 GB of VRAM or more is recommended.
*   **ImportError for gpuphot:** Make sure your virtual environment is activated and that `pip install gpuphot` completed without errors.
*   **Docker errors:** Verify Docker and the NVIDIA Container Toolkit are installed correctly.

If you encounter any other problems, refer to [USAGE.md](USAGE.md) and [DOCKER.md](DOCKER.md), or open an issue on the [GitHub repository](https://github.com/Light-Bridges/GPUPhot/issues).
