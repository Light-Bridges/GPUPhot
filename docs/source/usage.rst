Usage Guide
===========

This guide provides a comprehensive overview of how to use GPUPhot for astronomical image processing, covering instrument configuration, astrometry, single-image analysis, Celery distributed tasks, custom catalogs, cuML acceleration, and database queries.

.. contents:: Table of Contents
   :local:
   :depth: 2

1. Basic Usage (Single Image)
-----------------------------

Processing a single FITS image synchronously with the standalone library:

.. code-block:: python

    from astropy.io import fits
    from gpuphot.image_processor import create_processor

    # 1. Load image and header using Astropy
    with fits.open('session_01/target_A.fits') as hdul:
        imdata = hdul[0].data
        imheader = hdul[0].header

    # 2. Create processor instance ('default' loads default.json)
    processor = create_processor('default')

    # 3. Process the image
    phot_df, hwcs = processor.process_image(imdata, imheader)

    # 4. Inspect results
    print(phot_df.head())
    print(f"Plate solution: CRVAL1={hwcs.get('CRVAL1')}, CRVAL2={hwcs.get('CRVAL2')}")

.. note::
   When running inside the Docker container environment, ``gpuphot_worker.utils.open_image_file`` is also available to resolve paths relative to ``/data/images`` and load ``.npy`` array files.

Understanding the Output
^^^^^^^^^^^^^^^^^^^^^^^^

``process_image`` returns two objects:

1. **``phot_df`` (pandas.DataFrame):**
   Table containing all detected sources and photometric measurements:

   * ``xcentroid``, ``ycentroid``: Subpixel centroid positions (0-indexed).
   * ``flux``, ``fluxerr``: Instrumental flux and Poisson error in electrons.
   * ``mag``, ``magerr``: Calibrated magnitude and photometric uncertainty.
   * ``fwhm``: Full Width at Half Maximum in pixels.
   * ``elongation``: Source ellipticity / elongation ratio.
   * ``RA``, ``DEC``: Astrometric coordinates in decimal degrees (J2000).
   * ``RAERR``, ``DECERR``: Residuals against the reference catalog (degrees).
   * ``snr``: Signal-to-noise ratio.

2. **``hwcs`` (astropy.io.fits.Header):**
   Updated FITS header containing complete WCS keywords (``CRVAL``, ``CRPIX``, ``CD`` matrix, SIP distortion polynomials) and photometric zero-point metadata (``ZP``, ``ZPERR``, ``CATALOG``, ``CATBAND``, ``MAGLIM``).

2. Instrument Configuration
---------------------------

GPUPhot uses JSON configuration files (located in ``INSTRUMENT_CONFIG_BASE_PATH``) to adapt to different telescope optics and detector characteristics:

Sections of Instrument Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``header_keywords``: Maps internal canonical keys (e.g. ``gain``, ``exptime``) to the specific FITS keywords used by your camera (e.g. ``GAIN``, ``EXPTIME``).
* ``camera_specs``: Default fallback hardware values (pixel size, read noise, saturation) applied only if missing from the FITS header.
* ``processing_params``: Algorithmic parameters controlling background estimation, PSF modeling, and source filtering:
  * ``tile_section``: Image tile size for background estimation (e.g. ``1000``).
  * ``tile_section_psf``: Tile size for spatial PSF extraction (e.g. ``2500``).
  * ``center_factor``: Fraction of the image center used for PSF fitting (e.g. ``0.7``).
  * ``border``: Boundary margin in pixels excluded from detection (e.g. ``50``).
  * ``pca_method``: Enable Principal Component Analysis for spatial PSF variations (``true`` / ``false``).
  * ``SP_filt``: Enable salt-and-pepper defect filtering.
  * ``CR_filt``: Enable cosmic ray rejection.
  * ``max_stars_ref``: Max reference stars per tile for calibration (e.g. ``15``).
  * ``zp_maxmag``: Magnitude limit for catalog queries (e.g. ``21``).
* ``forced_values``: Strict overrides that take precedence over both FITS headers and default camera specs.
* ``filter_map``: Translates raw header filter strings to canonical band codes (e.g. ``SDSSr``, ``Lum``).

Parameter Precedence
^^^^^^^^^^^^^^^^^^^^

Parameters are resolved in strict priority order:
1. ``forced_values`` declared in the instrument configuration file.
2. Value extracted from the FITS header via ``header_keywords``.
3. Instrument configuration JSON file values.
4. Base defaults from ``default.json``.
5. Internal Python defaults in ``DefaultConfig.DEFAULT_PROCESSING_PARAMS``.

Overriding Parameters Dynamically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can pass temporary overrides directly to ``process_image`` without editing JSON files:

.. code-block:: python

    custom_params = {
        'tile_section': 1200,
        'center_factor': 0.85,
        'SP_filt': True,
        'pca_method': True
    }
    phot_df, hwcs = processor.process_image(imdata, imheader, **custom_params)

3. Distributed Processing via Celery
------------------------------------

GPUPhot provides asynchronous distributed workflows through Celery tasks in ``gpuphot_worker.tasks``.

Processing a Single Image Asynchronously
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from gpuphot_worker.tasks import process_image_task

    # Queue an image for processing
    task = process_image_task.delay(
        image_path='raw/night_01/target_001.fits',
        instrument_name='my_telescope'
    )

    print(f"Dispatched task ID: {task.id}")

    # Wait for result (optional)
    result = task.get(timeout=300)
    print("Execution summary:", result)

Batch Directory Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from gpuphot_worker.tasks import process_directory_task

    # Scan and queue all matching FITS files in a directory
    task = process_directory_task.delay(
        path='2025-03-15',
        filename='*.fits',
        instrument_name='my_telescope',
        reprocess=False
    )

    # Returns a dictionary mapping file paths to their Celery task IDs
    queued_tasks = task.get()
    print(f"Queued {len(queued_tasks)} images for processing.")

Automatic Memory Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^

When processing large format frames (e.g. 151 MP sensors), if the worker encounters a ``MemoryError``, it can automatically retry with binning or cropping if configured in the instrument JSON:

.. code-block:: json

    "image_reduction": {
      "apply_reduction": "on_failure",
      "binning": {
        "factor": 2,
        "method": "sum"
      },
      "crop_size": null
    }

4. Custom Catalogs (Local Databases)
------------------------------------

To operate in offline environments or query local spatial databases (e.g. PostgreSQL + Q3C) instead of CDS Vizier, supply a custom callable conforming to the search contract:

.. code-block:: python

    def my_local_catalog_search(coocenter, catalog, radius, mag_limit, ref_filter, row_limit, **kwargs):
        # Query local database using astropy SkyCoord and return pandas DataFrame
        # DataFrame must contain: 'gaia_id', 'RA', 'DEC', and reference filter magnitude
        return df_results

    phot_df, hwcs = processor.process_image(
        imdata, imheader,
        custom_vizier_search_func=my_local_catalog_search,
        custom_vizier_timeout=60
    )

5. GPU Crossmatch Optimization (cuML)
-------------------------------------

For catalog matching, GPUPhot uses an adaptive strategy between CPU (``scipy.spatial.cKDTree``) and GPU (`cuML <https://docs.rapids.ai/api/cuml/stable/>`_ NearestNeighbors):

* By default, ``cKDTree`` is used because its $O(N \log N)$ complexity outperforms GPU brute-force $O(N^2)$ for typical astronomical star densities.
* For very dense fields ($>3,000$ sources), cuML provides acceleration on high-end GPUs.
* Configure the threshold window via environment variables:

.. code-block:: bash

    export GPUPHOT_USE_CUML_CROSSMATCH=0     # 0 = adaptive (recommended)
    export GPUPHOT_CUML_MIN_SOURCES=3258     # Min sources to engage GPU
    export GPUPHOT_CUML_MAX_SOURCES=13549    # Max sources before O(N^2) degrades

6. Database Search API
----------------------

GPUPhot automatically records frame metadata in ``imastats`` and photometered sources in ``imaphot`` in PostgreSQL. Use ``gpuphot_worker.database_search_utils`` for queries:

.. code-block:: python

    from gpuphot_worker.database_search_utils import (
        search_by_radec,
        search_by_date_range,
        search_by_filename,
        search_transients
    )

    # Spatial cone search around coordinates (RA, Dec in degrees, radius in degrees)
    images_near_target = search_by_radec(ra=83.82, dec=-5.39, radius=0.5, table='imastats')

    # Date range query
    records = search_by_date_range(start_date='2025-01-01', end_date='2025-06-30')

    # Candidate transients search
    transients = search_transients(date_after='2025-03-01')

7. Error Handling
-----------------

GPUPhot defines an explicit exception hierarchy under ``gpuphot.exceptions.GPUPhotError``:

.. code-block:: python

    from gpuphot.exceptions import (
        GPUPhotError,
        UnableToAstrometrizeError,
        InsufficientStarsError,
        MoffatFitError,
        CatalogQueryError
    )

    try:
        phot_df, hwcs = processor.process_image(imdata, imheader)
    except UnableToAstrometrizeError:
        print("Astrometric plate solving failed (blind & hint attempts exhausted).")
    except InsufficientStarsError:
        print("Not enough isolated reference stars found in field for PSF fit.")
    except GPUPhotError as e:
        print(f"Pipeline error: {e}")
