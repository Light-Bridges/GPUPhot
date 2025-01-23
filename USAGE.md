# Usage Guide

## Instrument Configuration

GPUPhot uses instrument-specific configuration files to interpret image headers and set processing parameters. These files are typically named after the instrument (e.g., `default.json`) and contain three main sections:

1. `header_keywords`: Mapping of GPUPhot's internal keywords to the actual FITS header keywords used by the instrument.
2. `camera_specs`: Default values for camera specifications when not provided in the image header.
3. `processing_params`: Parameters controlling the image processing pipeline.

You can find an example configuration file, `default.json`, in the repository:

[View default.json](https://github.com/Light-Bridges/GPUPHOt/blob/main/gpuphot/instrument_configs/default.json)

Example structure of a configuration file:

```
{
  "header_keywords": {
    "exposure_time": "EXPT1",
    "filter": "FILTER",
    // ... (other header keywords)
  },
  "camera_specs": {
    "pxsize": null,
    "gain": null,
    // ... (other camera specifications)
  },
  "processing_params": {
    "tile_section": 1000,
    "center_factor": 0.7,
    // ... (other processing parameters)
  }
}
```

To use a specific instrument configuration:

```
from gpuphot.image_processor import create_processor

processor = create_processor('your_instrument_name', '/path/to/config/directory')
```

## Astrometry Setup

Before using GPUPhot, it's crucial to download the astrometry index files. These files are necessary for the astrometric calculations performed by the library.

To force the download of these index files, run the following code:

```
from gpuphot.utils.astro import get_solver

# Force download of index files
get_solver()
```

This process may take some time depending on your internet connection. It's recommended to run this once before starting to use GPUPhot in your project.

## Basic Usage

To use GPUPhot, import the package:

```python
import gpuphot
```

## Example usage (replace with actual API calls)

```python
processor = gpuphot.get_processor()
result = processor.process_image('path/to/image.fits')
print(result)
```

## Using Celery Tasks

```python
from gpuphot_worker.tasks import process_directory_task

task = process_directory_task.delay('path/to/directory')
result = task.get()
print(result)
```

## Docker Compose Environment

When using the Docker Compose setup:

1. Place your images in the directory specified by `IMAGE_BASE_PATH` in your `.env` file.
2. Use JupyterLab at `http://localhost:8888` to interact with GPUPhot.
3. Monitor tasks using Flower at `http://localhost:5555`.

For more detailed examples, see the Jupyter notebooks in the `notebooks` directory.
