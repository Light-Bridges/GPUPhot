Installation
============

Requirements
------------
- Python 3.8+
- CUDA-compatible GPU
- Docker and Docker Compose (for distributed processing)

Installing from PyPI
--------------------
::

    pip install gpuphot

Installing from source
----------------------
::

    git clone https://github.com/Light-Bridges/GPUPhot.git
    cd gpuphot
    bash install.sh
    pip install -r requirements.txt
    pip install .

Setting up Docker workers
-------------------------
1. Create a `.env` file in the project root with necessary environment variables.

2. Run Docker Compose:
   ::

       docker compose up -d

3. Access services:
   - JupyterLab: http://localhost:8888
   - Flower (Celery Monitor): http://localhost:5555

For more detailed installation instructions and environment setup, see the INSTALL.md file in the project root.

Post-Installation Setup
-----------------------
After installing GPUPhot, it's crucial to set up the astrometry index files:

::

    from gpuphot.utils.astro import get_solver
    get_solver()

This will download the necessary astrometry index files, which may take some time depending on your internet connection.
