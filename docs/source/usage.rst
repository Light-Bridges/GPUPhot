Usage
=====

This guide will help you get started with GPUPhot, a Python library for GPU-accelerated photometry and astrometry.

Basic Usage
-----------

Importing GPUPhot
^^^^^^^^^^^^^^^^^

To use GPUPhot in your Python scripts or Jupyter notebooks:

.. code-block:: python

   import gpuphot

Creating a Processor
^^^^^^^^^^^^^^^^^^^^

Create a processor for a specific instrument:

.. code-block:: python

   from gpuphot.image_processor import create_processor

   processor = create_processor('your_instrument_name', '/path/to/config/directory')

Processing an Image
^^^^^^^^^^^^^^^^^^^

To process an astronomical image:

.. code-block:: python

   from gpuphot_worker.utils import open_image_file

   # Open the image file
   imdata, imheader = open_image_file('/path/to/your/image.fits')

   # Process the image
   phot_df, hwcs = processor.process_image(imdata, imheader)

   print(phot_df)  # Photometry results
   print(hwcs)     # Updated WCS header

Instrument Configuration
------------------------

GPUPhot uses instrument-specific configuration files (JSON format) to interpret image headers and set processing parameters. These files contain:

1. `header_keywords`: Mapping of GPUPhot's internal keywords to FITS header keywords.
2. `camera_specs`: Default values for camera specifications.
3. `processing_params`: Parameters controlling the image processing pipeline.

For an example configuration, see the `default.json` file in the repository.

Astrometry Setup
----------------

Before using GPUPhot, download the necessary astrometry index files:

.. code-block:: python

   from gpuphot.utils.astro import get_solver

   # Force download of index files
   get_solver()

This process may take some time depending on your internet connection.

Using Celery Tasks
------------------

For distributed processing, GPUPhot integrates with Celery:

.. code-block:: python

   from gpuphot_worker.tasks import process_directory_task

   # Process all images in a directory
   task = process_directory_task.delay('/path/to/directory')
   result = task.get()
   print(result)

Advanced Usage
--------------

Image Reduction
^^^^^^^^^^^^^^^

GPUPhot supports various image reduction strategies:

- 'never': No reduction applied (default)
- 'always': Always apply reduction
- 'on_failure': Apply reduction if initial processing fails due to memory issues

These can be configured in the instrument configuration file.

Customizing Processing Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Customizing Processing Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, GPUPhot uses the processing parameters defined in the instrument configuration file (e.g., default.json). However, you can override these parameters for specific processing needs:

.. code-block:: python

   custom_params = {
       'tile_section': 1500,
       'center_factor': 0.8,
       'SP_filt': True
   }
   phot_df, hwcs = processor.process_image(imdata, imheader, processing_params=custom_params)

This approach allows you to temporarily modify processing parameters without changing the instrument configuration file. It's particularly useful for experimentation or when you need to adjust parameters for a specific image or set of images.

Note: Parameters passed this way will override the corresponding values in the instrument configuration file for this specific processing task only. The original configuration remains unchanged for subsequent operations.

For a complete list of available parameters and their meanings, refer to the documentation of the `process_image` method or the instrument configuration file structure.

For more detailed examples and advanced usage scenarios, refer to the Jupyter notebooks in the `notebooks` directory of the project repository.
