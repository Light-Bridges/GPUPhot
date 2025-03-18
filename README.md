
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
*  **Database Integration:** Stores the photometric and astrometric results into a postgreSQL database.

## Table of Contents

*   [Installation](#installation)
*   [Quick Start](#quick-start)
*   [Docker Compose](#docker-compose)
*   [Usage](#usage)
*   [Instrument Configuration](#instrument-configuration)
*   [Astrometry Setup](#astrometry-setup)
*   [Contributing](#contributing)
*   [License](#license)

## Installation

See the detailed installation instructions in [INSTALL.md](INSTALL.md).  Briefly, you have two main options:

1.  **Using Docker Compose (Recommended):** This is the easiest way to get started, as it provides a complete, pre-configured environment.
2.  **Manual Installation:**  This gives you more control but requires more setup.

**Quick Installation (using pip):**

For a basic, non-distributed installation (without Celery), you can install GPUPhot using pip:

```bash
git clone https://github.com/Light-Bridges/GPUPhot.git
cd GPUPhot
pip install .
```
**Important**: This will install only the core `gpuphot` library. To use Celery, you need additional packages.

## Docker Compose

The recommended way to deploy GPUPhot is using Docker Compose.  This provides a self-contained environment with all the necessary services, including:

*   Celery workers (for distributed processing)
*   RabbitMQ (message broker for Celery)
*   Redis (result backend for Celery)
*   PostgreSQL (database for storing results)
*   JupyterLab (interactive development environment)
*   Flower (Celery task monitor)

See [DOCKER.md](DOCKER.md) for detailed instructions on setting up and running GPUPhot with Docker Compose.

## Quick Start

This example shows how to process a single FITS image:

```python
from gpuphot.image_processor import create_processor
from gpuphot_worker.utils import open_image_file

# 1.  Load the image data and header.
try:
    imdata, imheader = open_image_file('path/to/your/image.fits')  # Replace with your image path
except ValueError as e:
    print(f"Error opening image: {e}")
    exit(1)

# 2. Create an ImageProcessor instance.
#    'default' uses the default configuration.  You can specify a different
#    instrument configuration file (e.g., 'my_telescope').
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
* **Replace Placeholders**: You need to use a real image

## Usage

For more detailed usage examples, including how to use Celery for distributed processing, see [USAGE.md](USAGE.md).

## Instrument Configuration

GPUPhot uses instrument-specific configuration files (JSON format) to adapt the processing pipeline to different telescopes and cameras. See [USAGE.md](USAGE.md#1-instrument-configuration) for details.

## Astrometry Setup

To enable astrometric calibration, you need to download the `astrometry.net` index files. See [USAGE.md](USAGE.md#2-astrometry-setup) for instructions.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPUPhot is released under the [MIT License](LICENSE).