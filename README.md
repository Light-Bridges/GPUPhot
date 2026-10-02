
# GPUPhot: GPU-Accelerated Photometry and Astrometry

[![PyPI](https://img.shields.io/pypi/v/gpuphot.svg)](https://pypi.org/project/gpuphot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23098402.svg)](https://doi.org/10.5281/zenodo.23098402)
[![Documentation Status](https://readthedocs.org/projects/gpuphot/badge/?version=latest)](https://gpuphot.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2609.32375-b31b1b.svg)](https://arxiv.org/abs/2609.32375)


GPUPhot is a Python library designed for high-performance photometry and astrometry of astronomical images. It leverages the power of NVIDIA GPUs (via CuPy) for accelerated computation and Celery for distributed processing, enabling fast and scalable analysis of large FITS image datasets.

## Key Features

*   **GPU Acceleration:**  Utilizes CuPy for significant speed improvements over CPU-based photometry.
*   **Automated Photometry:**  Performs aperture photometry with automatic source detection, background estimation, and PSF fitting.
*   **Astrometry:**  Integrates with `astrometry.net` (via the `astrometry` Python package) for accurate WCS calibration.
*   **Distributed Processing:**  Uses Celery to distribute image processing tasks across multiple CPU cores, GPUs, or even multiple machines.
*   **Docker Compose Deployment:**  Provides a ready-to-use Docker Compose setup for easy deployment and management of all necessary services (Celery workers, RabbitMQ, Redis, PostgreSQL, JupyterLab, Flower).
*   **Instrument-Specific Configurations:** Supports different telescope/camera setups through customizable JSON configuration files.
*   **JupyterLab Integration:**  Includes a JupyterLab environment for interactive data analysis and exploration.
*   **Database Integration:** Stores the photometric and astrometric results into a PostgreSQL database.

## Table of Contents

*   [Installation](#installation)
*   [Configuration & Data Management](#configuration--data-management)
*   [Docker Compose](#docker-compose)
*   [Scaling with GPUs](#scaling-with-gpus)
*   [Quick Start](#quick-start)
*   [Usage](#usage)
*   [Instrument Configuration](#instrument-configuration)
*   [Astrometry Setup](#astrometry-setup)
*   [Reproducibility & Benchmarks](#reproducibility--benchmarks)
*   [Citation](#citation)
*   [Contributing](#contributing)
*   [License](#license)

## Installation

> [!IMPORTANT]
> **Hardware Requirement:** GPUPhot strictly requires an **NVIDIA GPU** (Compute Capability $\ge$ 6.0) with proprietary NVIDIA drivers and the CUDA toolkit. Other GPU architectures and vendors (such as **Intel Arc**, **AMD Radeon / ROCm**, or **Apple Silicon**) are **not currently supported**.

GPUPhot supports two deployment modalities:

### 1. Standalone Python Library (PyPI)
For direct use in Python scripts, Jupyter notebooks, or integration into existing observatory pipelines:

```bash
# 1. Install CuPy matching your CUDA version (e.g., CUDA 12.x)
pip install cupy-cuda12x

# 2. Install GPUPhot
pip install gpuphot
```

### 2. Distributed Microservices Stack (Docker Compose)
For high-throughput, unattended queue-driven operations at robotic observatories, deploy the containerized cluster (Celery workers, RabbitMQ broker, Redis backend, PostgreSQL/Q3C database, Flower dashboard, JupyterLab):

```bash
docker compose up -d
```

See [INSTALL.md](INSTALL.md) for full prerequisites and [DOCKER.md](DOCKER.md) for container orchestration details. If developing or running the Celery worker service locally outside Docker, install via `pip install -e .[worker]`.

## Configuration & Data Management

**Crucial Step:** GPUPhot runs inside a container. To access your files (images and configs) stored on your host machine, you must map your local folders to the container's expected paths.

1.  Create a `.env` file in the project root (you can copy `.env.example` if available).
2.  Define your local paths in the `.env` file:

```bash
# .env file example

# HOST PATH: Where your FITS/NPY images are located on your PC
IMAGE_PATH=/home/user/raw_data

# HOST PATH: Where your instrument JSON configs are located
INSTRUMENT_CONFIG_PATH=/home/user/gpuphot_configs

# HOST PATH: Where astrometry indices should be stored/cached
ASTROMETRY_CACHE_PATH=./astrometry_cache
```

**Directory Mapping Reference:**

| Variable | Your Host Path (Example) | Container Internal Path | usage in Jupyter/Python |
| :--- | :--- | :--- | :--- |
| `IMAGE_PATH` | `/home/user/images` | `/data/images` | `open_image_file('my_image.fits')` * |
| `INSTRUMENT_CONFIG_PATH` | `/home/user/configs` | `/data/instrument_configs` | Managed by ConfigParser |

*\*Note: When running code inside Jupyter/Docker, paths are relative to `/data/images`.*

## GPU Crossmatch Calibration (optional)

If you have installed cuML (x86_64 only), you can enable GPU-accelerated catalog
cross-matching.  Because GPU efficiency depends on the number of sources, GPUPhot
needs GPU-specific thresholds to decide when to use GPU vs. CPU:

```bash
# .env
GPUPHOT_USE_CUML_CROSSMATCH=0     # 0 = adaptive (recommended)
GPUPHOT_CUML_MIN_SOURCES=3258     # set by the calibration tool
GPUPHOT_CUML_MAX_SOURCES=13549    # set by the calibration tool
```

Run the calibration tool once to find the right values for your GPU:

```bash
docker exec gpuphotfinal-profiler-1 \
    python3 /app/benchmarks/benchmark_cuml_crossover.py \
    --logspace 25 100 200000 --auto-refine
```

See [CUML_CALIBRATION.md](CUML_CALIBRATION.md) for the full guide.

## Docker Compose

The recommended way to deploy GPUPhot is using Docker Compose.  This provides a self-contained environment with all the necessary services.

See [DOCKER.md](DOCKER.md) for detailed instructions. To start the system:

```bash
docker compose up -d
```

## Scaling with GPUs

GPUPhot allows you to easily scale processing across all available GPUs on your machine. We provide a helper script to manage this automatically.

**Using the launch script:**

```bash
# Make the script executable
chmod +x launch_gpuphot.sh

# Launch workers (auto-detects number of GPUs and assigns one worker per GPU)
./launch_gpuphot.sh

# Or force a specific number of workers (e.g., 2)
./launch_gpuphot.sh 2
```

This script ensures that each Docker worker is assigned a unique `GPU_ID` to prevent resource contention.

## Quick Start

This example shows how to process a single FITS image using the standalone Python library:

```python
from astropy.io import fits
from gpuphot.image_processor import create_processor

# 1. Load the image data and header using Astropy
with fits.open('path/to/your/image.fits') as hdul:
    imdata = hdul[0].data
    imheader = hdul[0].header

# 2. Create an ImageProcessor instance ('default' loads default.json)
processor = create_processor('default')

# 3. Process the image
phot_df, hwcs = processor.process_image(imdata, imheader)

# 4. Results: phot_df is a pandas DataFrame, hwcs is the updated FITS header
print(phot_df)
print(f"Plate solution: CRVAL1={hwcs.get('CRVAL1')}, CRVAL2={hwcs.get('CRVAL2')}")
```

> **Note for Docker / Distributed Worker deployments:** When running inside the containerized microservices stack, `gpuphot_worker.utils.open_image_file` is also available to automatically resolve paths relative to `/data/images` and handle `.npy` files.

**Important Notes:**

*   **Astrometry Index Files:**  *Before* running the example above, you *must* download the astrometry index files.  See [Astrometry Setup](#astrometry-setup).

## Usage

For more detailed usage examples, including how to use Celery for distributed processing, see [USAGE.md](USAGE.md).

## Instrument Configuration

GPUPhot uses instrument-specific configuration files (JSON format). You can map header keywords or **force specific values** (like Gain or Read Noise) to override incorrect headers.

See [USAGE.md](USAGE.md#1-instrument-configuration) or the **Instrument Configuration Notebook** in JupyterLab for details.

## Astrometry Setup

To enable astrometric calibration, you need to download the `astrometry.net` index files. See [USAGE.md](USAGE.md#2-astrometry-setup) for instructions.

## Citation & Academic Use

If you use **GPUPhot** in scientific research or publications, please cite the framework paper and reference the Zenodo archive and Astrophysics Source Code Library (ASCL) record:

- **Framework & Distributed Pipeline (Paper):**  
  Lemes-Perera, S., Alarcon, M. R., Serra-Ricart, M., & Caballero-Gil, P. (2026).  
  *"GPUPHOT: A Python Framework for High-Performance GPU-Accelerated Photometry and Distributed Astronomical Data Reduction"*, Submitted to Astronomy and Computing. arXiv:2609.32375 [astro-ph.IM]. [doi:10.48550/arXiv.2609.32375](https://doi.org/10.48550/arXiv.2609.32375).

- **Kernel-Based Algorithms (Companion Paper):**  
  Alarcon, M. R., Lemes-Perera, S., Serra-Ricart, M., & Licandro, J. (2026).  
  *"GPUPHOT: Kernel-Based Algorithms for Point-Source Detection and Photometry with a Spatially Variable PSF"*, The Planetary Science Journal (in preparation).

- **Software Archive (Zenodo):**  
  Lemes-Perera, S., & Alarcon, M. R. (2026).  
  *Light-Bridges/GPUPhot: GPUPhot v1.0.1*. Zenodo. [doi:10.5281/zenodo.23098402](https://doi.org/10.5281/zenodo.23098402).

- **ASCL Indexing:**  
  GPUPhot is registered in the [Astrophysics Source Code Library](https://ascl.net/) (`ascl:XXXX.XXX`) and indexed by NASA ADS (`YYYYascl.soft...S`).

```bibtex
@article{gpuphot2026,
  author        = {Lemes-Perera, Samuel and Alarcon, Miguel R. and Serra-Ricart, Miquel and Caballero-Gil, Pino},
  title         = {{GPUPHOT: A Python Framework for High-Performance GPU-Accelerated Photometry and Distributed Astronomical Data Reduction}},
  journal       = {arXiv preprint arXiv:2609.32375},
  year          = {2026},
  eprint        = {2609.32375},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM},
  doi           = {10.48550/arXiv.2609.32375},
  note          = {Submitted to Astronomy and Computing}
}

@article{gpuphot_algorithms2026,
  author        = {Alarcon, Miguel R. and Lemes-Perera, Samuel and Serra-Ricart, Miquel and Licandro, Javier},
  title         = {{GPUPHOT: Kernel-Based Algorithms for Point-Source Detection and Photometry with a Spatially Variable PSF}},
  journal       = {The Planetary Science Journal},
  year          = {2026},
  note          = {In preparation}
}

@software{gpuphot_zenodo,
  author        = {Lemes-Perera, Samuel and Alarcon, Miguel R.},
  title         = {{Light-Bridges/GPUPhot: GPUPhot v1.0.1}},
  month         = oct,
  year          = {2026},
  publisher     = {Zenodo},
  version       = {v1.0.1},
  doi           = {10.5281/zenodo.23098402},
  url           = {https://doi.org/10.5281/zenodo.23098402}
}

@software{gpuphot_ascl,
  author        = {Lemes-Perera, Samuel and Alarcon, Miguel R. and Serra-Ricart, Miquel and Caballero-Gil, Pino and Licandro, Javier},
  title         = {{GPUPhot: A GPU-Accelerated Framework for Astronomical Photometry and Astrometry}},
  howpublished  = {Astrophysics Source Code Library},
  year          = {2026},
  note          = {ascl:XXXX.XXX}
}
```

## Reproducibility & Benchmarks

The full empirical benchmark campaign, raw execution telemetry, hardware inventories, and automated generation scripts for all manuscript tables and figures are permanently archived in the [v1.0.0 Release](https://github.com/Light-Bridges/GPUPhot/releases/tag/v1.0.0).

To clone this exact benchmark-reproducible state:

```bash
git clone --branch v1.0.0 https://github.com/Light-Bridges/GPUPhot.git
```

For detailed descriptions of the telemetry datasets, controls, and calibration measurements across architectures (A100, H100, L40S, RTX 3090/3060/3050Ti, Jetson Orin), see [benchmarks/data/README.md](benchmarks/data/README.md).

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPUPhot is released under the [MIT License](LICENSE).