Examples & Interactive Tutorials
=================================

GPUPhot includes a series of Jupyter notebooks that guide you through setup, configuration, execution, and data exploration.

.. toctree::
   :maxdepth: 1
   :caption: Interactive Notebooks:

   notebooks/0_System_Overview_Notebook.ipynb
   notebooks/1_Setup_Notebook.ipynb
   notebooks/2_Instrument_Configuration_Notebook.ipynb
   notebooks/3_Task_Execution_Notebook.ipynb
   notebooks/4_Database_Query_Notebook.ipynb
   notebooks/5_Multi_Instrument_Auto_Detection_Notebook.ipynb
   notebooks/6_cuML_Configuration_Notebook.ipynb

Overview of Tutorials
---------------------

1. **System Overview (`0_System_Overview_Notebook.ipynb`)**:
   High-level walkthrough of the GPUPhot distributed architecture (Celery, RabbitMQ, PostgreSQL/Q3C, and GPU workers).

2. **Initial Setup (`1_Setup_Notebook.ipynb`)**:
   Step-by-step guidance on setting up astrometry index files, verifying CUDA/CuPy availability, and initializing directories.

3. **Instrument Configuration (`2_Instrument_Configuration_Notebook.ipynb`)**:
   How to inspect headers, customize JSON configuration files, set forced values, and map keywords.

4. **Task Execution (`3_Task_Execution_Notebook.ipynb`)**:
   Hands-on examples of processing single FITS images and running asynchronous batch directory scans via Celery.

5. **Database Querying (`4_Database_Query_Notebook.ipynb`)**:
   Tutorial on querying `imastats` and `imaphot` tables using Q3C spatial indexing and built-in Python search helpers.

6. **Multi-Instrument Auto-Detection (`5_Multi_Instrument_Auto_Detection_Notebook.ipynb`)**:
   Demonstrates dynamic instrument detection and automated dispatch from FITS header metadata.

7. **cuML Configuration (`6_cuML_Configuration_Notebook.ipynb`)**:
   Benchmarks and configuration guide for GPU-accelerated cuML k-means clustering and crossmatching.
