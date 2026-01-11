
# GPUPhot: GPU-Accelerated Photometry and Astrometry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[//]: # ([![Build Status]&#40;https://github.com/Light-Bridges/GPUPhot/actions/workflows/tests.yml/badge.svg&#41;]&#40;https://github.com/Light-Bridges/GPUPhot/actions/workflows/tests.yml&#41;  <!-- Add this if you have CI -->)

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
*   [Contributing](#contributing)
*   [License](#license)

## Installation

See the detailed installation instructions in [INSTALL.md](INSTALL.md).  Briefly, you have two main options:

1.  **Using Docker Compose (Recommended):** This is the easiest way to get started, as it provides a complete, pre-configured environment.
2.  **Manual Installation:**  This gives you more control but requires more setup.

## Configuration & Data Management

**Crucial Step:** GPUPhot runs inside a container. To access your files (images and configs) stored on your host machine, you must map your local folders to the container's expected paths.

1.  Create a `.env` file in the project root (you can copy `env.example` if available).
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
chmod +x launch_workers.sh

# Launch workers (auto-detects number of GPUs and assigns one worker per GPU)
./launch_workers.sh

# Or force a specific number of workers (e.g., 2)
./launch_workers.sh 2
```

This script ensures that each Docker worker is assigned a unique `GPU_ID` to prevent resource contention.

## Quick Start

This example shows how to process a single FITS image. 

**Prerequisite:** Ensure your image is located inside the folder defined by `IMAGE_PATH` in your `.env` file.

```python
from gpuphot.image_processor import create_processor
from gpuphot_worker.utils import open_image_file

# 1. Load the image data and header.
#    The path must be relative to the mounted /data/images directory.
image_filename = 'session_01/target_A.fits' 

try:
    imdata, imheader = open_image_file(image_filename)
except ValueError as e:
    print(f"Error opening image: {e}")
    print("Hint: Check if the file exists in your mapped IMAGE_PATH folder.")
    exit(1)

# 2. Create an ImageProcessor instance.
#    'default' uses the default configuration.
processor = create_processor('default')

# 3. Process the image.
try:
    phot_df, hwcs = processor.process_image(imdata, imheader)
except Exception as e:
    print(f"Error during image processing: {e}")
    exit(1)

# 4. Print the results (a Pandas DataFrame).
print(phot_df)

# The updated FITS header (with WCS information) is in 'hwcs'.
```

**Important Notes:**

*   **Astrometry Index Files:**  *Before* running the example above, you *must* download the astrometry index files.  See [Astrometry Setup](#astrometry-setup).

## Usage

For more detailed usage examples, including how to use Celery for distributed processing, see [USAGE.md](USAGE.md).

## Instrument Configuration

GPUPhot uses instrument-specific configuration files (JSON format). You can map header keywords or **force specific values** (like Gain or Read Noise) to override incorrect headers.

See [USAGE.md](USAGE.md#1-instrument-configuration) or the **Instrument Configuration Notebook** in JupyterLab for details.

## Astrometry Setup

To enable astrometric calibration, you need to download the `astrometry.net` index files. See [USAGE.md](USAGE.md#2-astrometry-setup) for instructions.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPUPhot is released under the [MIT License](LICENSE).