Installation Guide
==================

Requirements
------------

Before installing GPUPhot, ensure your system meets the following prerequisites:

* **Python:** Version 3.8 or later (Python 3.12 recommended for production).
* **NVIDIA GPU:** Compute capability 6.0 or higher (Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper). At least 8 GB of VRAM is recommended.
* **NVIDIA Driver:** Version 525.x or later. Verify via ``nvidia-smi``.
* **CUDA Toolkit:** Version 11.x or 12.x (tested with 12.2, 12.4, and 12.6).
* **Operating System:** Linux (tested on Ubuntu 20.04, 22.04, and 24.04). Windows is supported via WSL2.
* **RAM:** 16 GB or more recommended.

Virtual Environment
-------------------

It is strongly recommended to install GPUPhot inside an isolated virtual environment:

.. code-block:: bash

    python3 -m venv .venv
    source .venv/bin/activate

Step 1: Install CuPy (Prerequisite)
-----------------------------------

GPUPhot relies on `CuPy <https://cupy.dev/>`_ for GPU-accelerated numerical operations. The appropriate CuPy wheel depends on your installed CUDA version:

For CUDA 12.x:
~~~~~~~~~~~~~~

.. code-block:: bash

    pip install cupy-cuda12x

For CUDA 11.x:
~~~~~~~~~~~~~~

.. code-block:: bash

    pip install cupy-cuda11x

Verify that CuPy can detect and communicate with your GPU:

.. code-block:: bash

    python -c "import cupy as cp; print('CUDA Driver/Runtime:', cp.cuda.runtime.runtimeGetVersion()); print('Device 0:', cp.cuda.Device(0))"

If this command fails, please consult the official `CuPy Installation Guide <https://docs.cupy.dev/en/stable/install.html>`_.

Step 2: Install GPUPhot
-----------------------

From PyPI:
~~~~~~~~~~

.. code-block:: bash

    pip install gpuphot

From Source:
~~~~~~~~~~~~

Clone the repository and install in standard or editable mode:

.. code-block:: bash

    git clone https://github.com/Light-Bridges/GPUPhot.git
    cd GPUPhot
    pip install .

For development (including documentation and test tools):

.. code-block:: bash

    pip install -e .[dev]

For running the distributed worker service (Celery, Redis, PostgreSQL/Q3C) locally:

.. code-block:: bash

    pip install -e .[worker]

Step 3: Astrometry Setup (Index Files)
--------------------------------------

To enable astrometric plate solving, the `astrometry.net` solver requires astrometric index files. Run the following once to initialize the solver and download default indices:

.. code-block:: python

    from gpuphot.utils.astro import get_solver
    get_solver()

Alternatively, if you already have local astrometry index files, configure their directory location using the environment variable:

.. code-block:: bash

    export ASTROMETRY_CACHE_PATH="/path/to/astrometry_cache"

Step 4: Distributed Processing via Docker Compose (Optional)
------------------------------------------------------------

If you plan to run GPUPhot as a distributed service with Celery workers, RabbitMQ, PostgreSQL, and Flower:

1. Configure your local paths in ``.env``:

   .. code-block:: bash

       cp .env.example .env

2. Start the full service stack:

   .. code-block:: bash

       docker compose up -d

3. Access the service dashboards:

   * **JupyterLab:** ``http://localhost:8888``
   * **Flower (Celery Monitor):** ``http://localhost:5555``

Troubleshooting
---------------

CuPy fails to initialize:
   Ensure that ``nvidia-smi`` runs correctly on the host. If using Docker, ensure the NVIDIA Container Toolkit is installed and configured as the default runtime.

Astrometry solver timeout:
   Ensure an internet connection is available on the first solver run to retrieve index files, or provide local index files via ``ASTROMETRY_CACHE_PATH``.
