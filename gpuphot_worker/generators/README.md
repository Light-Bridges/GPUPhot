# Notebook Generators

**Version:** 1.0
**Date:** 2024-08-02

## 1. Overview

This directory contains a series of Python scripts responsible for programmatically generating the Jupyter notebooks found in the top-level `/notebooks` directory.

Each script in this folder uses the `nbformat` library to construct a notebook, cell by cell, and then save it as a `.ipynb` file. This approach ensures that the tutorial notebooks provided with `GPUPhot` are consistent, version-controlled, and can be easily regenerated or modified.

These scripts are primarily intended to be executed by the `initialize_notebooks.sh` script, which populates the JupyterLab working directory with a fresh set of tutorial notebooks.

## 2. Directory Contents

The scripts are numbered to reflect the intended order of use for the final notebooks:

-   `0_System_Overview_Notebook.py`: Generates a notebook that provides a high-level overview of the `GPUPhot` system architecture and its services (RabbitMQ, Celery, PostgreSQL, etc.).
-   `1_Setup_Notebook.py`: Creates a notebook to guide the user through the initial, one-time setup steps, such as downloading astrometry index files.
-   `2_Instrument_Configuration_Notebook.py`: Generates a notebook that explains how to create and manage instrument configuration files.
-   `3_Task_Execution_Notebook.py`: Creates a notebook with examples on how to execute image processing tasks using the Celery worker.
-   `4_Database_Query_Notebook.py`: Generates a notebook showing how to query the PostgreSQL database to retrieve and analyze processing results.
-   `5_Multi_Instrument_Auto_Detection_Notebook.py`: Generates a notebook demonstrating automatic instrument detection and configuration dispatch from FITS headers.
-   `6_cuML_Configuration_Notebook.py`: Generates a notebook providing benchmarks and instructions for configuring GPU-accelerated cuML algorithms.

## 3. Usage

While direct manual execution is not the primary use case, the scripts can be run individually from the command line. This is useful for development or for regenerating a single notebook.

Each script accepts an optional `--output-dir` argument to specify where the generated `.ipynb` file should be saved. If not provided, it defaults to the project's root `/notebooks` directory.

### Example: Regenerating a Single Notebook

To regenerate the "Task Execution" notebook and place it in the default `/notebooks` directory, you would run the following command from the project's root directory:

```bash
python gpuphot_worker/generators/3_Task_Execution_Notebook.py
```

To place it in a different directory, for example, a custom `my_notebooks` folder:

```bash
python gpuphot_worker/generators/3_Task_Execution_Notebook.py --output-dir ./my_notebooks
```

### Initializing All Notebooks

To regenerate all notebooks at once, use the provided shell script, which executes all generator scripts in the correct order:

```bash
bash ./initialize_notebooks.sh
```
