# GPUPhot Usage Guide

This guide explains how to use GPUPhot for astronomical image processing, covering instrument configuration, astrometry setup, basic usage, Celery task execution, and interacting with the Docker Compose environment.

## Table of Contents

*   [1. Instrument Configuration](#1-instrument-configuration)
    *   [1.1 Configuration File Structure](#11-configuration-file-structure)
    *   [1.2 Creating and Modifying Configurations](#12-creating-and-modifying-configurations)
    *   [1.3 Loading a Configuration](#13-loading-a-configuration)
*   [2. Astrometry Setup](#2-astrometry-setup)
    *   [2.1 Solver strategy & timeouts](#21-solver-strategy--timeouts)
    *   [2.2 Handling astrometry failures](#22-handling-astrometry-failures)
*   [3. Basic Usage](#3-basic-usage)
    *   [3.1. Processing a Single Image](#31-processing-a-single-image)
    *   [3.2. Understanding the Output](#32-understanding-the-output)
*   [4. Using Celery Tasks (Distributed Processing)](#4-using-celery-tasks-distributed-processing)
    *   [4.1. Processing a Single Image with Celery](#41-processing-a-single-image-with-celery)
    *   [4.2. Processing a Directory of Images](#42-processing-a-directory-of-images)
    *   [4.3. Monitoring Tasks with Flower](#43-monitoring-tasks-with-flower)
    *   [4.4. Retrieving Results](#44-retrieving-results)
*   [5. Docker Compose Environment](#5-docker-compose-environment)
*   [6. Advanced Usage](#6-advanced-usage)
    *   [6.1. Image Reduction](#61-image-reduction)
    *   [6.2. Custom processing parameters](#62-custom-processing-parameters)
    *   [6.3. Using a Custom Catalog Source](#63-using-a-custom-catalog-source)
    *   [6.4. GPU Crossmatch Calibration (cuML)](#64-gpu-crossmatch-calibration-cuml)
*   [7. Error Handling](#7-error-handling)

## 1. Instrument Configuration

GPUPhot uses instrument-specific configuration files to adapt the image processing pipeline to different telescopes, cameras, and detectors. These files define how to:

*   Interpret FITS header keywords.
*   Set default camera parameters.
*   Adjust processing parameters.

Configuration files are JSON files (e.g., `default.json`, `my_telescope.json`) located in the directory specified by the `INSTRUMENT_CONFIG_PATH` environment variable (default: `./gpuphot/instrument_configs`).

### 1.1. Configuration File Structure

Configuration files have three main sections:

*   **`header_keywords`:**  A mapping between GPUPhot's internal keyword names (used consistently within the library) and the actual FITS header keywords used by *your* specific instrument.  This allows GPUPhot to work with FITS files from different sources, even if they use different keyword names for the same information (e.g., exposure time might be `EXPTIME`, `EXPOSURE`, `EXP_TIME`, etc.).

*   **`camera_specs`:** Default values for camera parameters (e.g., pixel size, gain, read noise).  These values are used *only if* the corresponding keyword is *not found* in the FITS header.

*   **`processing_params`:** Parameters that control the image processing pipeline itself (e.g., tiling options, background estimation, PSF fitting).

**Example (`default.json`):**

```json
{
  "header_keywords": {
    "exposure_time": "EXPT1",
    "filter": "FILTER",
    "naxis1": "NAXIS1",
    "naxis2": "NAXIS2",
    "gain" : "GAIN",
     "rdnoise": "RDNOISE"

  },
  "camera_specs": {
    "pxsize": null,
    "gain": null,
    "rdnoise": null
  },
  "processing_params": {
    "tile_section": 1000,
    "center_factor": 0.7,
    "SP_filt": false,
    "pca_method": false
  }
}
```
[View default.json](https://github.com/Light-Bridges/GPUPhot/blob/main/gpuphot/instrument_configs/default.json)


**Explanation:**

*   `"header_keywords"`:  In this example, GPUPhot will look for the exposure time in the FITS header keyword `EXPT1`, and the filter used in the keyword `FILTER`.  You *must* adapt this section to match the keywords used by *your* FITS files.
*   `"camera_specs"`:  `null` means that GPUPhot will *require* these values to be present in the FITS header.  If you set a default value (e.g., `"gain": 2.5`), that value will be used *only if* the `GAIN` keyword (as defined in `header_keywords`) is missing from the header.
*   `"processing_params"`: These are parameters that control details of the image processing.  You can experiment with these to fine-tune the results.

### 1.2. Creating and Modifying Configurations

1.  **Copy `default.json`:** The easiest way to create a new configuration is to copy the `default.json` file and modify it.
    ```bash
    cp /path/to/gpuphot/instrument_configs/default.json /path/to/your/instrument_configs/my_instrument.json
    ```
     Replace `/path/to/your/instrument_configs/` by the correct value.
2.  **Edit the JSON file:** Open the new JSON file in a text editor.
3.  **Modify `header_keywords`:** Change the values on the *right-hand side* of each entry to match the FITS keywords used by your instrument.  *Do not* change the keys on the left-hand side (e.g., `exposure_time`, `filter`).
4.  **Set default `camera_specs` (optional):** If you know the values of certain camera parameters and they are *not* reliably present in your FITS headers, you can set them here.
5.  **Adjust `processing_params` (optional):** Experiment with these parameters to optimize the processing for your specific images.
6. **Save** the changes.

#### Overriding incorrect FITS headers with `forced_values`

Some instruments write incorrect values to FITS headers (e.g. a gain reported
as `1.0` when the actual gain is `0.33 e⁻/ADU`).  `camera_specs` provides a
*fallback* used only when a keyword is absent, but it cannot override a value
that is already present in the header.  Use `forced_values` instead — it
takes priority over both the FITS header and `camera_specs`:

```json
{
  "forced_values": {
    "gain": 0.33,
    "read_noise": 3.5
  }
}
```

Any key listed in `forced_values` will always be used, regardless of what the
FITS header says.  Leave the block empty (`{}`) for instruments whose headers
are reliable.

#### Supported filters & filter mapping

GPUPhot maps the filter name found in the FITS header to one of its internal
canonical codes, which are used to select the correct photometric catalog
and calibration reference.

**Valid internal codes:**

| Code | Description |
|---|---|
| `Lum` | Luminance / white light |
| `Open` | No filter / open |
| `SDSSu` | SDSS *u* band |
| `SDSSg` | SDSS *g* band |
| `SDSSr` | SDSS *r* band |
| `SDSSi` | SDSS *i* band |
| `SDSSzs` | SDSS *z* band |
| `SDSSy` | SDSS *y* band |

The `default.json` config already maps the most common header values:

| Header value(s) | Internal code |
|---|---|
| `Lum`, `L`, `LUM`, `w` | `Lum` |
| `Open`, `Clear`, `C` | `Open` |
| `u`, `u'` | `SDSSu` |
| `g`, `g'`, `B`, `V` | `SDSSg` |
| `r`, `r'`, `R`, `Ha`, `Halpha` | `SDSSr` |
| `i`, `i'`, `I` | `SDSSi` |
| `z`, `z'` | `SDSSzs` |
| `y` | `SDSSy` |

To add a custom filter name used by your instrument, add an entry to the
`filter_map` block in your instrument JSON:

```json
{
  "filter_map": {
    "H-alpha": "SDSSr",
    "OIII": "SDSSg",
    "MyCustomFilter": "Lum"
  }
}
```

Any header value not present in `filter_map` will be passed through unchanged.
If the resulting code is not one of the eight valid internal codes, the image
will be processed but catalog cross-matching may be skipped.

### 1.3. Loading a Configuration

To use a specific configuration, you need to provide the `instrument_name` when creating an `ImageProcessor`:

```python
from gpuphot.image_processor import create_processor

# Using the default configuration
processor = create_processor('default')

# Using a custom configuration
processor = create_processor('my_instrument', '/path/to/your/instrument_configs')

# Using the default instrument
processor = create_processor('default')

```
The `config_dir` argument in `create_processor` is optional. By default, it uses the built-in `instrument_configs/` directory inside the package.

## 2. Astrometry Setup

GPUPhot uses the `astrometry.net` software (wrapped by the `astrometry` Python package) for astrometric calibration.  This requires downloading index files, which can be quite large.

**Download the Index Files (One-Time Setup):**

Run the following Python code *once* to download the necessary index files:

```python
from gpuphot.utils.astro import get_solver

# Force download of index files
get_solver()
```

*   This will download the files to the directory specified by the `ASTROMETRY_CACHE_PATH` environment variable.  Make sure this directory exists and has enough free space.
* This can take a *long time*, depending on your internet connection. This is normal.

> **Important:** Always run `get_solver()` once *outside* any Celery worker before
> starting distributed processing.  Worker tasks have strict time limits, and triggering
> an index file download inside a worker will cause a timeout.

### 2.1 Solver strategy & timeouts

The astrometry pipeline tries three strategies in order, stopping as soon as one succeeds:

1. **Local solver with position hint** — uses the `RA`/`DEC` from the FITS header to narrow the search field.
2. **Local solver without hint** — blind solve over the full index; slower but works when the header coordinates are absent or wrong.
3. **Online fallback** — submits source coordinates to the [Astrometry.net](https://nova.astrometry.net) web API.

Each attempt has an independent timeout controlled by environment variables:

| Variable | Default | Description |
|---|---|---|
| `GPUPHOT_ASTROMETRY_TIMEOUT` | `60` | Seconds allowed for each local solver attempt |
| `GPUPHOT_ASTROMETRY_ONLINE_TIMEOUT` | `70` | Seconds allowed for the online Astrometry.net attempt |
| `ASTROMETRY_API_KEY` | *(none)* | API key for nova.astrometry.net (online fallback only) |

If all three attempts fail, `AstrometrizationTimeoutError` is raised.
If the solver runs to completion but finds no plate solution, `UnableToAstrometrizeError` is raised.

### 2.2 Handling astrometry failures

```python
from gpuphot.exceptions import AstrometrizationTimeoutError, UnableToAstrometrizeError

try:
    result = processor.process_image('image.fits')
except AstrometrizationTimeoutError:
    # All attempts timed out — increase GPUPHOT_ASTROMETRY_TIMEOUT or
    # check that index files are present in ASTROMETRY_CACHE_PATH
    print("Astrometry timed out")
except UnableToAstrometrizeError:
    # Solver ran but found no solution — field may be outside index coverage
    # or the image contains too few / too many sources
    print("No plate solution found")
```

Common causes and fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| Timeout on first image only | Index files being downloaded inside the worker | Run `get_solver()` once before starting workers |
| Consistent timeout on all images | `GPUPHOT_ASTROMETRY_TIMEOUT` too short for your hardware | Increase timeout in `.env` |
| No solution, local solver | Field outside scale/coverage of downloaded index series | Check `ASTROMETRY_CACHE_PATH` contains the correct series (4100 + 5200) |
| No solution, online fallback | `ASTROMETRY_API_KEY` not set or field not in Astrometry.net | Set the API key in `.env` |

## 3. Basic Usage

```python
from gpuphot.image_processor import create_processor
from gpuphot_worker.utils import open_image_file

# 1. Create an ImageProcessor (using the default configuration)
processor = create_processor('default')

# 2. Load an image
imdata, imheader = open_image_file('path/to/your/image.fits')  # Replace with your image path

# 3. Process the image
phot_df, hwcs = processor.process_image(imdata, imheader)

# 4. Print the results
print(phot_df)
```

### 3.1. Processing a Single Image
The `process_image` function performs the photometry. It takes two main arguments:

* `imdata`: The image data as a NumPy array.
* `imheader`: The image FITS header as an `astropy.io.fits.Header` object.

### 3.2. Understanding the Output
The `process_image` function returns a tuple containing:
* `phot_df`: A Pandas DataFrame with the photometry results.
* `hwcs`: The image FITS header, updated.

**`phot_df` Columns:**

| Column    | Description                                                                  |
| :-------- | :--------------------------------------------------------------------------- |
| `xcentroid` | X coordinate of the detected object (in pixels).                             |
| `ycentroid` | Y coordinate of the detected object (in pixels).                             |
| `flux`    | Measured flux of the object (in ADU).                                     |
| `noise`   | Estimated uncertainty in the flux measurement (in ADU).                      |
| `snr`     | Signal-to-noise ratio of the detection.                                  |
| `RA`      | Right Ascension of the object (in degrees, after astrometric calibration).   |
| `DEC`     | Declination of the object (in degrees, after astrometric calibration).      |
| `RAERR`   | Error in the Right Ascension. It will be NaN when no match is found. |
| `DECERR`  | Error in the Declination. It will be NaN when no match is found.  |

## 4. Using Celery Tasks (Distributed Processing)

For processing multiple images, especially large datasets, GPUPhot uses Celery for distributed task execution. This allows you to process images in parallel, taking advantage of multiple CPU cores or even multiple machines.

### 4.1. Processing a Single Image with Celery

```python
from gpuphot_worker.tasks import process_image_task

# Submit the task to the Celery queue
task = process_image_task.delay('path/to/your/image.fits', instrument_name='your_instrument')

# 'task' is now an AsyncResult object.  You can use it to check the status
# of the task and retrieve the results.
print(f"Task ID: {task.id}")

# You can check the status periodically:
# print(f"Task status: {task.status}")

# ... later, when you need the results ...
# result = task.get()  # This will *block* until the task is complete.
# print(result)
```

### 4.2. Processing a Directory of Images

```python
from gpuphot_worker.tasks import process_directory_task

# Process all FITS files in a directory
task = process_directory_task.delay(path='path/to/your/images', filename='*.fits')
print(f"Task ID: {task.id}")
#You can retrieve the result later by: result = task.get()

# Other options:
# task = process_directory_task.delay(path='my_images', instrument_name='my_config', exclude_pattern='bad_image')
# task = process_directory_task.delay(path='my_images', reprocess=True) #Force to reprocess

```

### 4.3. Monitoring Tasks with Flower

Flower provides a web-based interface for monitoring Celery tasks.

1.  **Access Flower:** Open your web browser and go to `http://localhost:5555`.
2.  You'll see a list of tasks, their status (PENDING, STARTED, SUCCESS, FAILURE, RETRY), and other information.
3.  You can click on a task ID to see more details, including the traceback if the task failed.

### 4.4. Retrieving Results

```python
result = task.get()  # blocks until the task completes
print(result)
```

`task.get()` raises `celery.exceptions.TimeoutError` if you pass a `timeout`
argument and the task does not finish in time.

#### Result storage & TTL

GPUPhot uses two independent result stores:

| Store | TTL | Contents |
|---|---|---|
| **Redis** (Celery result backend) | **24 hours** | `task.get()` return value (JSON) |
| **PostgreSQL** (`imaphot` / `imastats`) | Permanent | Full photometry output |

The Redis TTL is set to 24 hours (`result_expires = 86400` in `celeryconfig.py`).
After that window, `task.get()` raises `celery.exceptions.TimeoutError` even
if the task completed successfully — but the data is still in PostgreSQL.

For long-running pipelines, query results from the database rather than
relying on `task.get()` for anything beyond same-session use:

```python
from gpuphot_worker.database_search_utils import search_by_filename

# Retrieve results any time — not subject to Redis TTL
df = search_by_filename('my_image.fits')
```

#### Worker crashes & task reliability

`worker_max_tasks_per_child = 1` means each worker process handles exactly
one task and then restarts.  This prevents VRAM leaks accumulating across
images.

By default, Celery acknowledges a task when it is *received* (not after it
completes).  If a worker crashes mid-task:

- The task is **not** re-queued automatically.
- Any data already written to PostgreSQL (e.g. `imastats` row) remains.
- The `imaphot` rows for that image may be absent or partial.

To detect incomplete results, check whether an image's `id` has rows in
`imaphot` — if `imastats` has the row but `imaphot` does not, the photometry
step did not complete:

```python
from gpuphot_worker.database_search_utils import search_by_filename

df = search_by_filename('my_image.fits')
if df.empty:
    print("No photometry results — task may have crashed")
```

## 5. Docker Compose Environment

When using the provided Docker Compose setup, keep in mind:

1.  **Image Location:** Place your FITS images in the directory you mapped to `/data/images` inside the container (this is controlled by the `IMAGE_PATH` environment variable in your `.env` file).
2.  **JupyterLab:** Access JupyterLab at `http://localhost:8888` to interact with GPUPhot interactively, run notebooks, and analyze results.
3.  **Flower:** Monitor Celery tasks at `http://localhost:5555`.
4. **RabbitMQ**: You can access it using `http://localhost:15672` with the credentials `gpuphot:gpuphot`.

## 6. Advanced Usage
### 6.1. Image Reduction

GPUPhot supports basic image reduction, such as binning and/or cropping, before photometric and astrometric analysis. This is especially useful when the system lacks sufficient resources to process the full image and its size needs to be reduced. You can configure image reduction using the `image_reduction` parameter in your instrument configuration file. The possible settings are:

* **apply_reduction**: Can be set to `always`, `never`, or `on_failure`.
* **binning**: Here you can define the binning factor and method.
    - **factor**: The binning factor.
    - **method**: `sum` or `median`.
* **center:** If the image is to be cropped, the center of the cropped image in pixels.
* **crop_size:** If the image is to be cropped, the size of the cropped area.

Example:
```json
"image_reduction": {
        "apply_reduction": "on_failure",
        "binning": {
            "factor": 2,
            "method": "sum"
        },
        "center": null,
        "crop_size": null
    }
```

### 6.2 Custom processing parameters
In addition to image reduction, other parameters of the processing chain can be configured using the `processing_params` entry of the instrument configuration file. For example:

*   **tile_section**: Size of the tiles.
*   **center_factor**: Fraction of the image center used to get reference stars for PSF modeling.
*   **SP_filt**: Apply a filter to remove salt-and-pepper noise.
*   **pca_method**: Use the PCA method.
*    **CR_filt**: Apply a filter for cosmic rays.
*    **border**: Set a border where no sources will be searched.
*   **tile_section_psf**: Size of the tiles used to model PSF variations across the image.
*   **lum_gmag_coeff**: The coefficient to weight the g magnitude in the luminosity calculation when the filter is `Lum`.
*    **lum_rmag_coeff**: The coefficient to weight the r magnitude in the luminosity calculation when the filter is `Lum`.

```json
"processing_params": {
    "tile_section": 1000,
    "center_factor": 0.7,
    "SP_filt": false,
    "pca_method": false,
    "tile_section_psf": 3000,
    "lum_gmag_coeff": 0.5,
    "lum_rmag_coeff": 0.5
  }
```

### 6.3 Using a Custom Catalog Source

GPUPhot allows you to replace the default Vizier client with a custom function to query local or private astronomical catalogs. This is a powerful feature for integrating `GPUPhot` with your own data infrastructure, such as a PostgreSQL database or a private API.

To enable this, you pass your custom function and its related parameters to `process_image` by grouping them in a dictionary.

#### Example: Using a Custom Search Function

Here is how you would call `process_image` with your custom catalog function. This method makes it clear that these parameters are part of an advanced, self-contained configuration.

```python
from gpuphot.image_processor import create_processor
from gpuphot_worker.utils import open_image_file
from my_project.catalog_search import my_custom_search_function # Your implementation

# 1. Create a processor
processor = create_processor('my_instrument')

# 2. Load image data and header
imdata, imheader = open_image_file('path/to/your/image.fits')

# 3. Define the custom catalog parameters in a dictionary
custom_catalog_kwargs = {
    'custom_vizier_search_func': my_custom_search_function,
    'custom_vizier_timeout': 120  # Optional: 2-minute timeout
}

# 4. Process the image, unpacking the dictionary as keyword arguments
phot_df, hwcs = processor.process_image(
    imdata,
    imheader,
    **custom_catalog_kwargs
)

# 5. Continue with your analysis
if phot_df is not None:
    print(f"Successfully processed image, found {len(phot_df)} sources.")
else:
    print("Image processing failed or returned no data.")

```

#### Implementing the Custom Function

Your custom function is the core of the integration. It must be carefully designed to meet `GPUPhot`'s expectations to ensure seamless operation.

For a complete guide on how to:
-   Correctly define the function signature.
-   Handle parameters like `radius`, `mag_limit`, and `ref_filter`.
-   Connect to a database (e.g., PostgreSQL) safely using connection pools.
-   Map your database columns to `GPUPhot`'s canonical names.
-   Manage errors and timeouts gracefully.

Please refer to the detailed developer documentation:

**[>> Guide for Custom Catalog Integration (CUSTOM_CATALOG.md)](CUSTOM_CATALOG.md)**

### 6.4. GPU Crossmatch Calibration (cuML)

On x86_64 systems with cuML installed, GPUPhot can use GPU-accelerated catalog
cross-matching.  The GPU is only faster for a specific source-count window that
depends on your hardware, so you need to calibrate it once per GPU model.

Run the calibration tool inside the profiler container:

```bash
docker exec gpuphotfinal-profiler-1 \
    python3 /app/benchmarks/benchmark_cuml_crossover.py \
    --logspace 25 100 200000 --auto-refine
```

The tool prints the recommended thresholds at the end.  Copy them to your `.env`:

```bash
GPUPHOT_USE_CUML_CROSSMATCH=0        # 0 = adaptive (use GPU only within the window)
GPUPHOT_CUML_MIN_SOURCES=<MIN>       # from the "robust" recommendation
GPUPHOT_CUML_MAX_SOURCES=<MAX>       # from the "robust" recommendation
```

For the full procedure, convergence options, and a reference table of known GPUs, see
**[CUML_CALIBRATION.md](CUML_CALIBRATION.md)**.

---

## 7. Error Handling

All GPUPhot-specific exceptions inherit from `GPUPhotError`, so you can catch
the whole hierarchy with a single handler or handle individual cases precisely.

```python
from gpuphot.exceptions import (
    GPUPhotError,
    InsufficientStarsError,
    MoffatFitError,
    ImageQualityError,
    UnableToAstrometrizeError,
    AstrometrizationTimeoutError,
    DataValidationError,
)

try:
    result = processor.process_image('image.fits')
except InsufficientStarsError as e:
    # Fewer than 5 isolated stars found — image too sparse for PSF fitting
    print(f"Not enough stars: {e.num_stars} detected")
except MoffatFitError:
    # PSF model could not be fitted (e.g. saturated or trailed stars)
    print("PSF fit failed — check image quality")
except ImageQualityError:
    # Image too crowded or too noisy for reliable photometry
    print("Image quality insufficient")
except AstrometrizationTimeoutError:
    # astrometry.net solver exceeded its time limit
    print("Astrometry timed out — check index files and solver settings")
except UnableToAstrometrizeError:
    # Solver ran to completion but could not find a solution
    print("Astrometry failed — field may be outside index coverage")
except GPUPhotError as e:
    # Catch-all for any other GPUPhot error
    print(f"Pipeline error: {e}")
```

### Exception reference

| Exception | Raised when |
|---|---|
| `GPUPhotError` | Base class — catch-all for all GPUPhot errors |
| `InsufficientStarsError` | Fewer than 5 isolated stars detected in the image |
| `MoffatFitError` | Moffat PSF model cannot be fitted to the reference stars |
| `ImageQualityError` | Image is too crowded or too noisy for reliable photometry |
| `UnableToAstrometrizeError` | astrometry.net solver failed to find a plate solution |
| `AstrometrizationTimeoutError` | astrometry.net solver exceeded its time limit |
| `DataValidationError` | Input data is missing required fields or is malformed |
| `InvalidGroupSizeError` | Star grouping parameters produced an invalid configuration |

CUDA runtime errors (GPU memory exhaustion, illegal address) are handled
internally by the `capture_cuda_exception` decorator and will be retried once
before being re-raised as standard `CUDARuntimeError` exceptions from CuPy.
