GPUPhot Documentation
=====================

**GPUPhot** is a high-performance Python library for GPU-accelerated photometry and astrometry of astronomical CCD and CMOS images. It utilizes CuPy for array computations on NVIDIA GPUs and integrates Celery for scalable, distributed processing across heterogeneous multi-GPU systems.

Key Features
------------

* **GPU Acceleration:** High-throughput numerical operations executed on NVIDIA GPUs via CuPy.
* **Automated Photometry:** Accurate source detection, background modeling, and PSF extraction (empirical, analytical, and PCA-based).
* **Astrometric Calibration:** Native integration with Astrometry.net for blind and hinted WCS plate solving.
* **Distributed Processing:** Celery-based job queues for processing multi-gigabyte astronomical observation streams across CPU cores and GPU nodes.
* **Astronomical Database Integration:** Storage of frame metadata and calibrated catalogs in PostgreSQL using Q3C spatial indexing.
* **Instrument Config Flexibility:** Robust JSON configurations to handle heterogeneous FITS headers and detector specifications.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   usage
   examples
   citation

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules
   gpuphot_worker

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
