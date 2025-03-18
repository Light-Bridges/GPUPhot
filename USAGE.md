# GPUPhot Usage Guide

This guide explains how to use GPUPhot for astronomical image processing, covering instrument configuration, astrometry setup, basic usage, Celery task execution, and interacting with the Docker Compose environment.

## Table of Contents

*   [1. Instrument Configuration](#1-instrument-configuration)
    *   [1.1  Configuration File Structure](#11-configuration-file-structure)
    *   [1.2.  Creating and Modifying Configurations](#12-creating-and-modifying-configurations)
    *   [1.3.  Loading a Configuration](#13-loading-a-configuration)
*   [2. Astrometry Setup](#2-astrometry-setup)
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
     *  [6.1. Image Reduction](#61-image-reduction)
     *  [6.2 Custom processing parameters](#62-custom-processing-parameters)

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
[View default.json](https://github.com/Light-Bridges/GPUPHOt/blob/main/gpuphot/instrument_configs/default.json)


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

### 1.3. Loading a Configuration

To use a specific configuration, you need to provide the `instrument_name` when creating an `ImageProcessor`:

```python
from gpuphot.image_processor import create_processor

# Using the default configuration
processor = create_processor('default')

# Using a custom configuration
processor = create_processor('my_instrument', '/path/to/your/instrument_configs') #INSTRUMENT_CONFIG_BASE_PATH

#Using the instrument name defined in the .env
processor = create_processor()

```
The `config_dir` argument in `create_processor` is optional. By default, uses the value defined by `INSTRUMENT_CONFIG_BASE_PATH`.

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
* This can take a *long time*, depending on your internet connection. It is normal.

## 3. Basic Usage

```python
import gpuphot
from gpuphot_worker.utils import open_image_file

# 1. Create an ImageProcessor (using the default configuration)
processor = gpuphot.get_processor()

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
* `hwcs`: The image fits header, updated.

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
* Using the `task.get()` method.
```python
result = task.get() #This will wait the task is completed
print(result)
```

## 5. Docker Compose Environment

When using the provided Docker Compose setup, keep in mind:

1.  **Image Location:** Place your FITS images in the directory you mapped to `/data/images` inside the container (this is controlled by the `IMAGE_BASE_PATH` environment variable in your `.env` file).
2.  **JupyterLab:** Access JupyterLab at `http://localhost:8888` to interact with GPUPhot interactively, run notebooks, and analyze results.
3.  **Flower:** Monitor Celery tasks at `http://localhost:5555`.
4. **RabbitMQ**: You can acces using `http://localhost:15672` with the credentials `gpuphot:gpuphot`

## 6. Advanced Usage
### 6.1. Image Reduction

GPUPhot supports basic image reduction as binning or/and cropping before photometric and astrometric analisys. This is specially usefull when the system doesn't have enough resources to process the image, and it is needed to reduce the image size. You can configure the image reduction using the `image_reduction` parameter inside your instrument configuration file. The posible setings are:

* **apply_reduction**: It can be set as `always`, `never` or `on_failure`.
* **binning**: Here you can define the binning factor, and the method.
    - **factor**: The binning factor.
    - **method**: sum or median.
* **center:** If the image has to be cropped, the center, in pixels, of the cropped image.
* **crop_size:** If the image has to be cropped, the size of the cropped area.

Example:
```
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
As well as image reduction configuration, some other parameters of the processing chain can be configured using the `processing_params` entry of the instrument configuration file. For example:

*   **tile_section**: Size of the tiles.
*   **center_factor**: Fraction of the center of the image that will be used to get the reference stars to model the PSF.
*   **SP_filt**: Apply a filter to remove salt and pepper noise.
*   **pca_method**: Use the PCA method.
*    **CR_filt**: Apply a filter for cosmic rays.
*    **border**: Set a border where no sources will be searched.
*   **tile_section_psf**: Size of the tiles used to model the variations of the PSF across the image.
*   **lum_gmag_coeff**: The coefficient to weight the g magnitude in the calculation of the luminosity, when the filter is `Lum`.
*    **lum_rmag_coeff**: The coefficient to weight the r magnitude in the calculation of the luminosity, when the filter is `Lum`.
```

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
