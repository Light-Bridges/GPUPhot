import hashlib
import inspect
import os
import re
import signal
import tempfile
import threading
import time
from datetime import datetime
from functools import lru_cache

import astrometry
import ephem
import matplotlib.pyplot as plt
try:
    import cupynumeric as np
except ImportError:
    import numpy as np

import nvtx
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from sklearn.linear_model import RANSACRegressor

from .catalog import crossmatch_sources
from ..exceptions import AstrometrizationTimeoutError
from ..instrument_config_parser import HeaderKey
from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


class SingletonSolver:
    """
    A singleton class for managing the astrometry solver.
    Optimized to avoid redundant file existence checks using a marker file.
    """
    _instance = None
    _lock = threading.Lock()
    _solver_initialized_successfully = False  # Internal flag for success status

    # Marker file name
    MARKER_FILENAME = ".astrometry_files_ok.hash"

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    logger.debug("Creating new SingletonSolver instance")
                    cls._instance = super().__new__(cls)
                    # Don't initialize directly here, __init__ is more appropriate
                    # or call an explicit init method if needed before returning
                    # the instance if logic requires it.
                    # In this case, actual initialization happens on the first get_solver() call.
                    cls._instance._initialized = False  # Flag to control initialization
        return cls._instance

    def __init__(self):
        """
        Initializer. Prevents re-initialization if already done.
        """
        # Avoid re-running __init__ on the existing singleton instance
        if hasattr(self, '_initialized') and self._initialized:
            return

        with self._lock:  # Protect the actual initialization
            # Double check inside the lock
            if hasattr(self, '_initialized') and self._initialized:
                return

            logger.debug("Initializing the SingletonSolver instance...")
            self.solver = None
            self.cache_dir = self._determine_cache_dir()
            self.required_files = self._get_required_files(self.cache_dir)
            self.required_files_hash = self._calculate_list_hash(self.required_files)
            self.marker_file_path = os.path.join(self.cache_dir, self.MARKER_FILENAME)

            self.initialize_solver()
            self._initialized = True  # Mark as initialized

    @nvtx.annotate('_determine_cache_dir', category='utils.astro.SingletonSolver')
    def _determine_cache_dir(self):
        """Determines the appropriate cache directory."""
        default_cache = '/data/astrometry_cache'
        env_cache = os.getenv('ASTROMETRY_CACHE_PATH')
        # Be careful with __file__ if using PyInstaller or similar tools
        try:
            # Assumes the script is in a subdirectory like 'utils/astro' relative to the cache
            base_dir = os.path.dirname(os.path.realpath(__file__))
            fallback_cache = os.path.abspath(os.path.join(base_dir, '..', '..', 'astrometry_cache'))
        except NameError:  # __file__ is not defined (e.g., in an interactive REPL)
            fallback_cache = os.path.join(os.getcwd(), 'astrometry_cache')
            logger.warning(f"__file__ not defined, using fallback relative to CWD: {fallback_cache}")

        if os.path.exists(default_cache):
            cache = default_cache
            logger.debug(f"Using default cache directory: {cache}")
        elif env_cache and os.path.exists(env_cache):  # Check if environment variable points to an existing dir
            cache = env_cache
            logger.debug(f"Using cache directory from environment variable: {cache}")
        elif env_cache and not os.path.exists(env_cache):
            logger.warning(
                f"Cache directory from environment variable {env_cache} does not exist. Attempting to create.")
            try:
                os.makedirs(env_cache, exist_ok=True)
                cache = env_cache
                logger.info(f"Created cache directory from environment variable: {env_cache}")
            except Exception as e:
                logger.error(
                    f"Could not create cache directory from environment variable {env_cache}: {e}. Using fallback.")
                cache = fallback_cache  # Go to fallback if creating ENV one fails
                logger.debug(f"Using fallback cache directory: {cache}")
        else:
            cache = fallback_cache
            logger.debug(f"Using fallback cache directory: {cache}")

        # Create the final directory if it doesn't exist (could be the fallback)
        if not os.path.exists(cache):
            logger.warning(f"Cache directory not found at {cache}. Attempting to create.")
            try:
                os.makedirs(cache, exist_ok=True)
                logger.info(f"Cache directory created: {cache}")
            except PermissionError:
                logger.error(f"Cannot create directory {cache}. Check permissions.")
                # Use temporary directory as a last resort
                try:
                    cache = tempfile.mkdtemp(prefix='astrometry_cache_')
                    logger.warning(f"Using temporary directory as cache due to permission error: {cache}")
                except Exception as temp_e:
                    logger.critical(f"Failed to create even a temporary directory: {temp_e}")
                    raise RuntimeError(f"Unable to establish a cache directory. Permissions issues likely.") from temp_e
            except Exception as e:
                logger.error(f"Unexpected error creating directory {cache}: {e}")
                # You might raise an exception here if the cache is critical
                try:
                    cache = tempfile.mkdtemp(prefix='astrometry_cache_')
                    logger.warning(f"Using temporary directory as cache due to unexpected error: {cache}")
                except Exception as temp_e:
                    logger.critical(f"Failed to create even a temporary directory: {temp_e}")
                    raise RuntimeError(
                        f"Unable to establish a cache directory. Unexpected error during creation.") from temp_e

        logger.debug(f"Using final cache directory: {cache}")
        return cache

    @nvtx.annotate('_get_required_files', category='utils.astro.SingletonSolver')
    def _get_required_files(self, cache_dir):
        """Gets the list of required index files."""
        # Assuming astrometry.series_XXXX.index_files exists and works
        try:
            # Make sure the series modules are available in the astrometry package installation
            # These might be under astrometry.util or directly under astrometry depending on version
            # Adjust the import if necessary
            return (
                    astrometry.series_5200.index_files(cache_directory=cache_dir, scales={0, 1, 2, 3, 4, 5, 6})
                    +
                    astrometry.series_4100.index_files(cache_directory=cache_dir, scales={7, 8, 9, 10, 11})
            )
        except AttributeError as e:
            logger.error(
                f"Error getting file list from astrometry. Ensure the library is installed and accessible: {e}")
            # Decide how to handle this: empty list, raise exception?
            # Raising an exception is safer to avoid undefined behavior.
            raise RuntimeError("Could not get the list of required files from astrometry.") from e
        except Exception as e:
            logger.error(f"Unexpected error getting file list from astrometry: {e}")
            raise RuntimeError("Unexpected error getting required files.") from e

    @staticmethod
    @nvtx.annotate('_calculate_list_hash', category='utils.astro.SingletonSolver')
    def _calculate_list_hash(file_list):
        """Calculates a SHA256 hash for a list of file paths."""
        hasher = hashlib.sha256()
        # Sort to ensure the hash is consistent regardless of order
        # Convert Path objects to strings if they exist in the list
        sorted_files = sorted([str(f) for f in file_list])
        for file_path in sorted_files:
            hasher.update(file_path.encode('utf-8'))
        return hasher.hexdigest()

    @nvtx.annotate('_check_marker_file', category='utils.astro.SingletonSolver')
    def _check_marker_file(self):
        """Checks if the marker file exists and contains the correct hash."""
        if not os.path.exists(self.marker_file_path):
            logger.debug("Marker file not found.")
            return False
        try:
            with open(self.marker_file_path, 'r') as f:
                stored_hash = f.read().strip()
            if stored_hash == self.required_files_hash:
                logger.debug("Marker file hash matches. Assuming files exist and are valid.")
                return True
            else:
                logger.warning("Marker file hash MISMATCH. Re-verification of files required.")
                # Delete old/invalid marker
                try:
                    os.remove(self.marker_file_path)
                except OSError as e:
                    logger.error(f"Could not delete invalid marker file {self.marker_file_path}: {e}")
                return False
        except Exception as e:
            logger.error(f"Error reading marker file {self.marker_file_path}: {e}")
            # If there's an error reading, better verify everything again
            return False

    @nvtx.annotate('_create_marker_file', category='utils.astro.SingletonSolver')
    def _create_marker_file(self):
        """Creates the marker file with the current required files hash."""
        try:
            with open(self.marker_file_path, 'w') as f:
                f.write(self.required_files_hash)
            logger.debug(f"Marker file created/updated: {self.marker_file_path}")
        except Exception as e:
            logger.error(f"Error creating marker file {self.marker_file_path}: {e}")

    @nvtx.annotate('_remove_marker_file', category='utils.astro.SingletonSolver')
    def _remove_marker_file(self):
        """Removes the marker file if it exists."""
        if os.path.exists(self.marker_file_path):
            try:
                os.remove(self.marker_file_path)
                logger.debug(f"Marker file removed: {self.marker_file_path}")
            except OSError as e:
                logger.error(f"Error removing marker file {self.marker_file_path}: {e}")

    @nvtx.annotate('initialize_solver', category='utils.astro.SingletonSolver')
    def initialize_solver(self):
        """
        Initialize the astrometry solver, using the marker file for optimization.
        """
        # We already have self.cache_dir, self.required_files, self.required_files_hash, self.marker_file_path from __init__

        # 1. Quick check using the marker file
        if self._check_marker_file():
            # Assume files are okay, try initializing directly
            try:
                logger.debug("Attempting to initialize solver based on marker file.")
                # Pass the list of required file paths (strings or Path objects)
                self.solver = astrometry.Solver(self.required_files)
                logger.info("Solver initialized successfully.")
                # Mark that initialization succeeded globally for the instance
                SingletonSolver._solver_initialized_successfully = True
                return  # Success
            except Exception as e:
                logger.error(
                    f"Error initializing solver even though marker existed: {e}. Proceeding to full verification.")
                # The marker was wrong (maybe files got corrupted without list changing)
                self._remove_marker_file()  # Remove incorrect marker
                # Continue to full verification below...
        else:
            logger.debug("Marker file invalid or not found. Performing full file verification.")

        # 2. Full verification (if marker check failed or didn't exist)
        index_files_exist = self.check_index_files_exist(self.required_files)
        if not index_files_exist:
            logger.warning(
                "Unable to locate all astrometry index files. The solver will attempt to download them now. "
                "This process may take up to an hour or more, depending on your internet speed and the number of missing files (total ~34GB). "
                "If calling this from within a Celery worker (or similar) with a short time limit, this may cause timeouts. "
                "It is recommended to run the initialization (`from gpuphot.utils.astro import get_solver; solver = get_solver()`) "
                "once outside the time-sensitive process to download the index files beforehand."
            )
            # astrometry.Solver initialization will handle the download
            # if the files are missing.

        # 3. Try to initialize the solver (this might trigger downloads)
        try:
            logger.debug(f"Initializing astrometry.Solver with {len(self.required_files)} index files listed.")
            self.solver = astrometry.Solver(self.required_files)
            logger.info("Solver initialized successfully.")
            # If we got here, initialization (and potential download) was successful
            self._create_marker_file()  # Create the marker for future runs
            SingletonSolver._solver_initialized_successfully = True

        except Exception as e:
            SingletonSolver._solver_initialized_successfully = False  # Mark failure
            error_message = str(e)
            logger.error(f"Error initializing the solver: {error_message}")
            self._remove_marker_file()  # Remove marker if initialization failed

            # Attempt to handle corrupted files mentioned in the error
            match = re.search(r'loading\s+"(.*?)"\s+failed', error_message, re.IGNORECASE)
            if match:
                problematic_file = match.group(1)
                # Ensure the file path is absolute if it's relative in the error message
                if not os.path.isabs(problematic_file):
                    problematic_file = os.path.join(self.cache_dir,
                                                    os.path.basename(problematic_file))  # Make a best guess

                logger.warning(f"Error message suggests a problem loading: {problematic_file}")
                if os.path.exists(problematic_file):  # Check if it actually exists before trying to remove
                    logger.warning(f"Attempting to remove potentially problematic file: {problematic_file}")
                    try:
                        os.remove(problematic_file)
                        logger.info(
                            f"Problematic file removed: {problematic_file}. Re-running initialization might be necessary.")
                        # You could retry here, but beware of infinite loops.
                        # Maybe it's better to fail and let the user retry the operation.
                        # raise RuntimeError(f"Removed a potentially corrupt file ({problematic_file}). Please retry the operation.") from e
                    except OSError as remove_error:
                        logger.error(f"Error removing the problematic file {problematic_file}: {remove_error}")
                else:
                    logger.warning(
                        f"The reported problematic file '{problematic_file}' was not found on disk at the expected location.")

            # Re-raise or handle the error more specifically if needed
            # raise RuntimeError("Astrometry.net solver initialization failed.") from e # Optional: re-raise a cleaner error

    @staticmethod
    @nvtx.annotate('check_index_files_exist', category='utils.astro.SingletonSolver')
    def check_index_files_exist(required_files):
        """
        Check if the required index files exist (the potentially slow check).

        :param required_files: List of required index file paths (expecting Path objects or strings).
        :type required_files: list
        :return: True if all files exist and none are partial downloads, False otherwise.
        :rtype: bool
        """
        logger.debug(f"Performing existence check for {len(required_files)} files...")
        all_exist = True
        try:
            for file_path_obj in required_files:
                file_path = str(file_path_obj)  # Ensure it's a string
                # Check for the main file
                if not os.path.exists(file_path):
                    return False
                # Check if a partial download file exists for it
                # Adjust if astrometry uses a different temporary file naming scheme
                elif os.path.exists(f"{file_path}.download"):
                    return False
            return all_exist

        except Exception as e:
            logger.error(f"Error during file verification check: {e}")
            return False  # Assume failure if there's an exception

    @nvtx.annotate('get_solver_instance', category='utils.astro.SingletonSolver')
    def get_solver_instance(self):
        """
        Returns the initialized solver instance.
        Raises RuntimeError if initialization failed or hasn't completed successfully.
        """
        # Check the instance variable `solver` AND the class variable tracking success
        if not self.solver or not SingletonSolver._solver_initialized_successfully:
            # You could attempt re-initialization here if it makes sense, or just fail.
            # logger.warning("Solver is not initialized successfully. Attempting re-initialization.")
            # try:
            #    self.initialize_solver() # Careful with recursion/loops if it fails consistently
            # except Exception as e:
            #    logger.error(f"Re-initialization attempt failed: {e}")
            #    raise RuntimeError("Astrometry solver could not be initialized correctly after retry.") from e

            # Check again after attempting re-initialization
            # if not self.solver or not SingletonSolver._solver_initialized_successfully:
            #    raise RuntimeError("Astrometry solver could not be initialized correctly.")

            # Simpler approach: just raise if not ready.
            raise RuntimeError("Astrometry solver could not be initialized correctly or is not ready.")
        return self.solver


### # @hierarchical_debug(logger)
@nvtx.annotate('get_solver', category='utils.astro')
def get_solver():
    """
    Gets the singleton instance of the astrometry solver.
    Initialization occurs the first time this function is called.

    :return: The initialized astrometry.Solver instance.
    :raises RuntimeError: If the solver fails to initialize.
    """
    instance = SingletonSolver()
    return instance.get_solver_instance()


@nvtx.annotate('get_astrometry_params', category='utils.astro')
def get_astrometry_params(h_wcs, image_shape):
    """
    Get astrometry parameters from WCS header.

    :param h_wcs: WCS header.
    :type h_wcs: dict
    :param image_shape: Shape of the image.
    :type image_shape: tuple
    :return: Center coordinates, field of view, and scale.
    :rtype: tuple
    """
    w = WCS(h_wcs)
    ra, dec = w.all_pix2world(image_shape[1] // 2, image_shape[0] // 2, 1)
    try:
        cd11 = float(h_wcs['CD1_1'][0])
    except:
        cd11 = float(h_wcs['CD1_1'])
    try:
        cd12 = float(h_wcs[HeaderKey.CD1_2.value][0])
    except:
        cd12 = float(h_wcs[HeaderKey.CD1_2.value])
    scale = np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600
    coocenter = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    npx = max(image_shape)
    FOV = 2 * np.sqrt(2 * ((scale * npx / 3600) ** 2))
    return coocenter, FOV, scale


def logodds_callback_100(logodds):
    """
    Callback function for astrometry solver.

    :param logodds: Log-odds of the solution.
    :type logodds: list
    :return: Action to take (STOP or CONTINUE).
    :rtype: astrometry.Action
    """
    # print(logodds)
    if (logodds[0] > 100.0) | (len(logodds) > 2):
        return astrometry.Action.STOP
    else:
        return astrometry.Action.CONTINUE


def handler(signum, frame):
    """
    Signal handler for astrometry timeout.

    :param signum: Signal number.
    :type signum: int
    :param frame: Current stack frame.
    :type frame: frame
    :raises AstrometrizationTimeoutError: If astrometry times out.
    """
    logger.warning("Astrometry step timed out via signal.")
    raise AstrometrizationTimeoutError("Astrometry step exceeded the time limit")


@nvtx.annotate('_extract_wcs_from_solution', category='utils.astro')
def _extract_wcs_from_solution(solution):
    """Helper to safely extract WCS fields from a local astrometry solution."""
    if solution and solution.matches:
        best_match = solution.best_match()
        if hasattr(best_match, 'wcs_fields'):
            # Check if it's callable (method) or attribute
            if callable(best_match.wcs_fields):
                return best_match.wcs_fields()
            else:
                return best_match.wcs_fields
        else:
            logger.warning("Local solution.best_match() object has no 'wcs_fields'.")
            return None
    return None


@nvtx.annotate('_attempt_local_solve', category='utils.astro')
def _attempt_local_solve(solver, star_data_local, size_hint, position_hint, common_params_local):
    """Helper function to perform a single local solver attempt with timeout."""

    timeout = int(os.getenv("GPUPHOT_ASTROMETRY_TIMEOUT", 60))

    logger.debug(f"Local astrometry (timeout: {timeout}s)")

    wcs_header = None
    try:
        signal.alarm(timeout)
        solution = solver.solve(
            **star_data_local,
            size_hint=size_hint,
            position_hint=position_hint,
            **common_params_local
        )
        signal.alarm(0)  # Disable alarm on successful return
        wcs_header = _extract_wcs_from_solution(solution)
        if wcs_header:
            logger.debug(f"Local astrometry successful: Match found.")

    except AstrometrizationTimeoutError:
        logger.warning(f"Local astrometry failed: Timed out.")
    except Exception as e:
        signal.alarm(0)  # Ensure the alarm is off
        logger.warning(f"Local astrometry failed: An error occurred: {e}",
                       exc_info=False)  # Keep exc_info=False unless debugging specific errors here

    return wcs_header


@nvtx.annotate('_attempt_online_solve', category='utils.astro')
def _attempt_online_solve(df_proc, image_width, image_height):
    """Helper function to perform the online Astrometry.net attempt with timeout."""

    online_timeout = int(os.getenv("GPUPHOT_ASTROMETRY_ONLINE_TIMEOUT", 70))

    attempt_name = "Online astrometry with Astrometry.net"
    logger.debug(f"{attempt_name} (overall timeout: {online_timeout}s)")
    wcs_header = None

    try:
        ast_client = AstrometryNet()

        api_key = os.getenv('ASTROMETRY_API_KEY', None)
        if not api_key:
            logger.warning(
                f"{attempt_name} skipped: ASTROMETRY_API_KEY environment variable not set. Astrometry.net will attempt using other configurations.")
        else:
            ast_client.api_key = api_key
            logger.debug(f"{attempt_name}: Using API key from ASTROMETRY_API_KEY environment variable.")

        signal.alarm(online_timeout)  # Set overall timeout for this attempt
        online_wcs_header = ast_client.solve_from_source_list(
            x=df_proc['xcentroid'],
            y=df_proc['ycentroid'],
            image_width=image_width,
            image_height=image_height,
            solve_timeout=online_timeout,
        )
        signal.alarm(0)  # Disable alarm on successful return

        if online_wcs_header:
            logger.debug(f"{attempt_name} successful: Solution found online.")

            # hdu = fits.PrimaryHDU(header=online_wcs_header)
            # hdu.verify('silentfix+warn')
            # wcs = WCS(hdu.header)
            # clean_header = wcs.to_header(relax=True)

            # logger.debug(f"{attempt_name}: Cleaning COMMENT and HISTORY cards from obtained header.")
            # keys_to_remove = ['COMMENT', 'HISTORY']
            #
            # logger.debug(f"{attempt_name}: Cleaning COMMENT and HISTORY cards from obtained header.")
            #
            # clean_header = fits.Header()
            # for result_key in online_wcs_header.keys():
            #     if result_key not in keys_to_remove:
            #         clean_header[result_key] = online_wcs_header[result_key]

            clean_header = copy_header_selectively(online_wcs_header)
            try:
                wcs_header = dict(clean_header)
            except Exception:
                wcs_header = clean_header
        else:
            logger.debug(f"{attempt_name} completed: No solution found online.")

    except AstrometrizationTimeoutError:
        logger.warning(f"{attempt_name} failed: Timed out (overall signal timeout).")
    except Exception as e:
        signal.alarm(0)  # Ensure the alarm is off
        logger.error(f"{attempt_name} failed: An error occurred: {e}",
                     exc_info=True)  # Log full traceback for online errors

    return wcs_header


### # @hierarchical_debug(logger)
@nvtx.annotate('astrometrice2', category='utils.astro')
def astrometrice2(df: pd.DataFrame, scale: float,
                  central_ra: float, central_dec: float,
                  image_shape: tuple,
                  sip_order: int = 3, n_max: int = 500) -> dict:
    """
    Performs astrometry by attempting multiple strategies: local solving
    with position hints, local solving without hints, and online solving
    via Astrometry.net. Uses signal.alarm for timeouts (UNIX only).

    :param df: DataFrame containing detected sources (must include
               'xcentroid' and 'ycentroid' columns).
    :type df: pd.DataFrame
    :param scale: Approximate image scale in arcseconds per pixel (for hints).
    :type scale: float
    :param central_ra: Approximate central right ascension in degrees (for hints).
    :type central_ra: float
    :param central_dec: Approximate central declination in degrees (for hints).
    :type central_dec: float
    :param image_shape: Shape of the image as a tuple (height, width).
    :type image_shape: tuple
    :param sip_order: SIP (Simple Imaging Polynomial) order for local solver. Default is 3.
    :type sip_order: int
    :param n_max: Maximum number of sources (from the top of df) to use. Default is 500.
    :type n_max: int
    :param local_timeout: Timeout in seconds for each local solver attempt. Default is 60.
    :type local_timeout: int
    :param online_timeout: Timeout in seconds for the entire online Astrometry.net attempt
                           (managed by signal.alarm). Default is 70.
    :type online_timeout: int
    :param astrometry_net_timeout: Internal solve timeout in seconds passed *to*
                                   Astrometry.net's solve_from_source_list. Default is 90.
    :type astrometry_net_timeout: int
    :return: Dictionary containing the WCS header fields if successful,
             otherwise an empty dictionary.
    :rtype: dict
    :raises AstrometrizationTimeoutError: If astrometry fails to find a solution after all attempts.
    """
    wcs_header = None
    start_time = time.time()

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Input DataFrame is not valid or empty. Cannot perform astrometry.")
        # Original code returned {}, let's stick to that for no-sources case
        # even though the failure path below raises error for failed attempts.
        # This is a distinction between no sources vs sources but no solution.
        return {}

    # Select top n_max sources, prioritizing by SNR or Flux if available
    if 'snr' in df.columns:
        logger.debug(f"Sorting input DataFrame by 'snr' descending before selecting top {n_max}.")
        df_sorted = df.sort_values(by='snr', ascending=False).dropna().reset_index(drop=True)
    elif 'flux' in df.columns:
        logger.warning("Input DataFrame lacks 'snr' column, sorting by 'flux' descending instead.")
        df_sorted = df.sort_values(by='flux', ascending=False).dropna().reset_index(drop=True)
    else:
        logger.warning(
            f"Input DataFrame lacks both 'snr' and 'flux' columns. Selecting top {n_max} sources without explicit sorting. Astrometry quality may be affected.")
        df_sorted = df

    df_proc = df_sorted.head(n_max).copy()

    if df_proc.empty:
        logger.warning("DataFrame is empty after selecting top n_max sources. Cannot perform astrometry.")
        return {}

    logger.info(f"Starting astrometry using top {len(df_proc)} sources.")

    try:
        solver = get_solver()
    except RuntimeError as e:
        logger.error(f"Failed to obtain solver instance: {e}")
        # If solver initialization failed completely, we cannot proceed.
        raise AstrometrizationTimeoutError("Failed to initialize astrometry solver.") from e

    common_params_local = {
        'solution_parameters': astrometry.SolutionParameters(
            logodds_callback=logodds_callback_100,
            sip_order=sip_order
        )
    }

    star_data_local = None
    try:
        # Check solver signature once to prepare star data
        solve_params = inspect.signature(solver.solve).parameters
        if 'stars_xs' in solve_params and 'stars_ys' in solve_params:
            star_data_local = {'stars_xs': df_proc['xcentroid'], 'stars_ys': df_proc['ycentroid']}
            logger.debug("Prepared star data for local solver using 'stars_xs' and 'stars_ys'.")
        elif 'stars' in solve_params:
            star_data_local = {'stars': df_proc[['xcentroid', 'ycentroid']].values.tolist()}
            logger.debug("Prepared star data for local solver using 'stars' list.")
        else:
            logger.error(
                "Unexpected local solver signature: solver.solve() method parameters not recognized. Skipping local attempts.")
            star_data_local = None  # Ensure local attempts are skipped

    except Exception as e:
        logger.error(f"Failed to inspect local solver parameters: {e}", exc_info=True)
        star_data_local = None  # Ensure local attempts are skipped

    # Set up a signal handler (outside the loop, done once)
    # This handler needs to be defined in the global/module scope or
    # made accessible here. The provided handler is already at module level.
    signal.signal(signal.SIGALRM, handler)

    try:
        # --- Attempt 1: Local solving with a position hint ---
        try:
            if star_data_local:  # Only attempt if star data preparation was successful
                logger.info(f"Attempt 1: Local astrometry with position hint")
                wcs_header = _attempt_local_solve(
                    solver,
                    star_data_local,
                    size_hint=astrometry.SizeHint(
                        lower_arcsec_per_pixel=scale * 0.8,
                        upper_arcsec_per_pixel=scale * 1.
                    ),
                    position_hint=astrometry.PositionHint(
                        ra_deg=central_ra,
                        dec_deg=central_dec,
                        radius_deg=1,
                    ),
                    common_params_local=common_params_local
                )
        except Exception as e:
            logger.warning(f"Error during Attempt 1 (local with hint): {e}", exc_info=True)

        # --- Attempt 2: Local solving without a position hint ---
        try:
            if not wcs_header and star_data_local:  # Only attempt if previously failed AND local attempts are possible
                logger.info(f"Attempt 2: Local astrometry without position hint")
                wcs_header = _attempt_local_solve(
                    solver,
                    star_data_local,
                    size_hint=None,
                    position_hint=None,
                    common_params_local=common_params_local
                )
        except Exception as e:
            logger.warning(f"Error during Attempt 2 (local without hint): {e}", exc_info=True)

        # --- Attempt 3: Online solving with Astrometry.net ---
        try:
            if not wcs_header:  # Only attempt if local attempts failed
                logger.info(f"Attempt 3: Online astrometry with Astrometry.net")

                # Ensure correct image dimensions order (astroquery expects width, height)
                image_height, image_width = image_shape
                logger.debug(f"Using image dimensions for Astrometry.net: width={image_width}, height={image_height}")

                wcs_header = _attempt_online_solve(
                    df_proc=df_proc,
                    image_width=image_width,
                    image_height=image_height,
                )
        except Exception as e:
            logger.warning(f"Error during Attempt 3 (online Astrometry.net): {e}", exc_info=True)


    finally:
        signal.alarm(0)  # Final safety net to disable any pending alarm
        # Consider restoring the default handler if this function is called frequently,
        # and you want to avoid potential side effects on other parts of the application.
        # signal.signal(signal.SIGALRM, signal.SIG_DFL)

    # --- Return Result ---
    elapsed_time = time.time() - start_time
    if wcs_header:
        logger.info(f"Astrometry finished successfully in {elapsed_time:.2f} seconds.")
        return wcs_header
    else:
        logger.warning(f"Astrometry finished: All attempts failed to find a solution after {elapsed_time:.2f} seconds.")
        # The original code returned {}, but also had a signal timeout raise AstrometrizationTimeoutError.
        # The refactored helpers catch and log the timeouts and other errors, returning None.
        # The final failure state should probably rise for consistency with timeout handling.
        # If no WCS is found after all attempts (whether due to timeout, error, or no match), it's a failure.
        raise AstrometrizationTimeoutError("Astrometry failed to find a solution after all attempts.")


@nvtx.annotate('get_zeropoint', category='utils.astro')
def get_zeropoint(df_catalog, df_sources, exptime, center_lims=None, N=50,
                  solar_filter=0.6, dist_thres_px=3, min_snr=30, max_snr=300,
                  plot=False):
    """
    Calculate the zeropoint for photometry.

    :param df_catalog: Catalog dataframe.
    :type df_catalog: pd.DataFrame
    :param df_sources: Sources dataframe.
    :type df_sources: pd.DataFrame
    :param exptime: Exposure time.
    :type exptime: float
    :param center_lims: Limits for center region, optional.
    :type center_lims: tuple or None
    :param N: Number of brightest stars to use, default is 50.
    :type N: int
    :param solar_filter: Solar filter value, default is 0.6.
    :type solar_filter: float
    :param dist_thres_px: Distance threshold in pixels, default is 3.
    :type dist_thres_px: int
    :param min_snr: Minimum signal-to-noise ratio, default is 30.
    :type min_snr: int
    :param max_snr: Maximum signal-to-noise ratio, default is 300.
    :type max_snr: int
    :param plot: Whether to plot the results, default is False.
    :type plot: bool
    :return: Dictionary of zeropoint parameters.
    :rtype: dict
    """
    source_coords_matched_idx, ref_coords_matched_idx = crossmatch_sources(df_sources[['RA', 'DEC']].values,
                                                                           df_catalog[['RA', 'DEC']].values,
                                                                           thres_px=dist_thres_px)

    solar_cat_filt = np.abs(df_catalog['SOLAR'])[ref_coords_matched_idx
                     ] < solar_filter
    source_coords_matched_idx = source_coords_matched_idx[solar_cat_filt]
    ref_coords_matched_idx = ref_coords_matched_idx[solar_cat_filt]
    det_mag = -2.5 * np.log10(df_sources.flux[source_coords_matched_idx].values / exptime)
    cat_mag = df_catalog['MAG'][ref_coords_matched_idx].to_numpy()
    inf_nan_mask = np.isfinite(cat_mag) & np.isfinite(det_mag) & ~np.isnan(
        cat_mag) & ~np.isnan(det_mag)
    cat_mag = cat_mag[inf_nan_mask]
    det_mag = det_mag[inf_nan_mask]
    snr = df_sources['snr'][source_coords_matched_idx][inf_nan_mask].values
    bright_mask = (snr > min_snr) & (snr < max_snr)

    if center_lims is None:
        xmin = np.floor(df_sources['xcentroid'].min())
        xmax = np.ceil(df_sources['xcentroid'].max())
        ymin = np.floor(df_sources['ycentroid'].min())
        ymax = np.ceil(df_sources['ycentroid'].max())
    else:
        xmin, xmax, ymin, ymax = center_lims

    bright_mask = bright_mask & (df_sources['xcentroid'][source_coords_matched_idx][inf_nan_mask].values >= xmin) & \
                  (df_sources['xcentroid'][source_coords_matched_idx][inf_nan_mask].values <= xmax) & \
                  (df_sources['ycentroid'][source_coords_matched_idx][inf_nan_mask].values >= ymin) & \
                  (df_sources['ycentroid'][source_coords_matched_idx][inf_nan_mask].values <= ymax)

    if len(cat_mag[bright_mask]) > N:
        indices_original = np.where(bright_mask)[0]
        m = np.argsort(snr[bright_mask])
        brightest_original = indices_original[m[-N:]]
        brightest_mask = np.zeros(len(bright_mask), dtype=bool)
        brightest_mask[brightest_original] = True
        bright_mask = bright_mask & brightest_mask

    if len(cat_mag[bright_mask]) <= 3:
        zp, ezp, n, min_mag, max_mag = 0, 0, 0, 0, 0

    else:
        y = cat_mag[bright_mask] - det_mag[bright_mask]
        mask = np.abs(y - np.nanmean(y)) < np.nanstd(y)
        if np.sum(mask) > 15:
            reg = RANSACRegressor(random_state=42, residual_threshold=0.05
                                  ).fit(det_mag[bright_mask].reshape([-1, 1])[mask], cat_mag[
                bright_mask].reshape([-1, 1])[mask])
            inlier = reg.inlier_mask_
            if np.sum(inlier) > 10 and np.abs(np.mean(y[mask][inlier]) - np
                    .mean(y[mask])) < np.std(y[mask]):
                zp = np.mean(y[mask][inlier])
                n = np.sum(inlier)
                ezp = np.std(y[mask][inlier]) / np.sqrt(n)
                min_mag = np.min(cat_mag[bright_mask][mask][inlier])
                max_mag = np.max(cat_mag[bright_mask][mask][inlier])
            else:
                zp = np.mean(y[mask])
                n = np.sum(mask)
                ezp = np.std(y[mask]) / np.sqrt(n)
                min_mag = np.min(cat_mag[bright_mask][mask])
                max_mag = np.max(cat_mag[bright_mask][mask])
        else:
            zp = np.mean(y[mask])
            n = np.sum(mask)
            ezp = np.std(y[mask]) / np.sqrt(n)
            min_mag = np.min(cat_mag[bright_mask][mask])
            max_mag = np.max(cat_mag[bright_mask][mask])

    params = {
        'ZP': np.round(zp, 4),
        'EZP': np.round(ezp, 4),
        'CATNSTAR': n,
        'ZPMINMAG': np.round(min_mag, 2),
        'ZPMAXMAG': np.round(max_mag, 2),
        'BVMIN': np.round(0.65 - solar_filter / 2, 2),
        'BVMAX': np.round(0.65 + solar_filter / 2, 2),
    }

    if plot:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)
        ax.plot(cat_mag, cat_mag - det_mag - zp, 'k.', alpha=0.6)
        if np.sum(mask) > 15:
            ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
                    det_mag[bright_mask][mask] - zp, 'b.', alpha=0.1)
            ax.plot(cat_mag[bright_mask][mask][inlier], cat_mag[bright_mask
            ][mask][inlier] - det_mag[bright_mask][mask][inlier] - zp,
                    'r.', label='zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp,
                                                                       n), alpha=0.5)
        else:
            ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
                    det_mag[bright_mask][mask] - zp, 'r.', label=
                    'zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp, n), alpha=0.5)
        ax.set_xlabel('catalog magnitude')
        ax.set_ylabel('error magnitude')
        ax.legend(frameon=False)
        ax.set_ylim(-1.5, 1.5)
        plt.show()

    return params


@nvtx.annotate('get_target_snr', category='utils.astro')
def get_target_snr(dfm: pd.DataFrame, target_ra: float, target_dec: float, dist_thres_px=3) -> float:
    """
    Get the SNR of the target.

    :param dfm: Dataframe with photometry data.
    :type dfm: pd.DataFrame
    :param target_ra: Target right ascension, in degrees.
    :type target_ra: float
    :param target_dec: Target declination, in degrees.
    :type target_dec: float
    :param dist_thres_px: Distance threshold in pixels, default is 3.
    :type dist_thres_px: int
    :return: SNR of the target.
    :rtype: float
    """
    target_coords_np = np.array([[target_ra, target_dec]])
    if 'RA' not in dfm.columns or 'DEC' not in dfm.columns:
        logger.error("DataFrame missing 'RA' or 'DEC' columns.")
        return 0.0
    ref_coords_np = dfm[['RA', 'DEC']].values

    try:
        # Call the main wrapper. It will detect NumPy input and use CPU path.
        _, ref_idx = crossmatch_sources(target_coords_np, ref_coords_np, thres_px=dist_thres_px)
        # ref_idx will be NumPy array because input was NumPy

        if ref_idx.size > 0:
            if 'snr' not in dfm.columns:
                logger.error("DataFrame missing 'snr' column.")
                return 0.0
            target_snr = dfm['snr'].iloc[ref_idx[0]]
        else:
            target_snr = 0.0
    except Exception as e:
        logger.error(f"Error during crossmatch or SNR retrieval in get_target_snr: {e}")
        target_snr = 0.0

    return float(target_snr)


@nvtx.annotate('get_maglim', category='utils.astro')
def get_maglim(mag: np.ndarray, snr: np.ndarray, snr_lim: float) -> float:
    """
    Get the limiting magnitude.

    :param mag: Magnitude values.
    :type mag: numpy.ndarray
    :param snr: SNR values.
    :type snr: numpy.ndarray
    :param snr_lim: SNR limit.
    :type snr_lim: float
    :return: Limiting magnitude.
    :rtype: float
    """
    mask = (snr > 2) & (snr < 15)
    if np.sum(mask) < 3:
        maglim = 0
        logger.warning('Not enough stars to calculate the limiting magnitude.')
    else:
        p = np.polyfit(mag[mask], np.log10(snr[mask]), 1)
        x = np.linspace(12, 24, 100)
        fitted_snr = np.polyval(p, x)
        maglim = x[np.argmin(np.abs(fitted_snr - np.log10(snr_lim)))]
    return np.round(maglim, 2)


@nvtx.annotate('radec_to_moon_sun', category='utils.astro')
def radec_to_moon_sun(ra, dec, site_latitude, site_longitude, site_elevation, date_obs):
    """
    Calculate moon and sun positions relative to a target.

    :param ra: Right ascension of target in degrees.
    :type ra: float
    :param dec: Declination of target in degrees.
    :type dec: float
    :param site_latitude: Latitude of observation site in degrees.
    :type site_latitude: float
    :param site_longitude: Longitude of observation site in degrees.
    :type site_longitude: float
    :param site_elevation: Elevation of observation site in meters.
    :type site_elevation: float
    :param date_obs: Date and time of observation.
    :type date_obs: str
    :return: Tuple of moon altitude, azimuth, distance to target, phase, and sun altitude and azimuth.
    :rtype: tuple
    """
    observer = ephem.Observer()
    observer.lat = np.radians(site_latitude)
    observer.lon = np.radians(site_longitude)
    observer.elevation = site_elevation
    observer.date = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S.%f')
    target = ephem.FixedBody()
    target._ra = np.radians(ra)
    target._dec = np.radians(dec)
    target.compute(observer)

    moon = ephem.Moon()
    sun = ephem.Sun()
    moon.compute(observer)
    sun.compute(observer)
    moon.compute(observer)
    moon_alt = np.degrees(moon.alt)
    moon_az = np.degrees(moon.az)
    distance_to_moon = np.degrees(ephem.separation((target.ra, target.dec), (moon.ra, moon.dec)))
    moon_phase = moon.phase

    sun_alt = np.degrees(sun.alt)
    sun_az = np.degrees(sun.az)

    return round(moon_alt, 2), round(moon_az, 2), round(distance_to_moon, 2), round(moon_phase, 3), round(sun_alt,
                                                                                                          2), round(
        sun_az, 2)


@lru_cache(maxsize=32)
@nvtx.annotate('get_observer_and_frame', category='utils.astro')
def get_observer_and_frame(SITELAT, SITELON, SITEELEV, Date):
    """Create and cache location and AltAz frame objects"""
    Observatory = EarthLocation(lat=SITELAT * u.deg, lon=SITELON * u.deg, height=SITEELEV * u.m)
    aa = AltAz(location=Observatory, obstime=Date)
    return Observatory, aa


@nvtx.annotate('radec_to_altaz', category='utils.astro')
def radec_to_altaz(RA, DEC, SITELAT, SITELON, SITEELEV, Date):
    """
    Convert RA/Dec to altitude and azimuth.

    :param RA: Right ascension in degrees.
    :type RA: float
    :param DEC: Declination in degrees.
    :type DEC: float
    :param SITELAT: Site latitude in degrees.
    :type SITELAT: float
    :param SITELON: Site longitude in degrees.
    :type SITELON: float
    :param SITEELEV: Site elevation in meters.
    :type SITEELEV: float
    :param Date: Observation date and time.
    :type Date: str
    :return: Tuple of azimuth, altitude, airmass, and zenith distance.
    :rtype: tuple
    """
    # Get cached objects
    _, aa = get_observer_and_frame(SITELAT, SITELON, SITEELEV, Date)

    # Create SkyCoord and transform
    coords_deg = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs')
    coords_altaz = coords_deg.transform_to(aa)

    # Extract values directly
    az = coords_altaz.az.deg
    alt = coords_altaz.alt.deg
    airmass = float(coords_altaz.secz)
    zen = coords_altaz.zen.deg

    return round(az, 6), round(alt, 6), round(airmass, 6), round(zen, 6)


@nvtx.annotate('radec_to_gal', category='utils.astro')
def radec_to_gal(RA, DEC):
    """
    Convert RA/Dec to Galactic coordinates.

    :param RA: Right ascension in degrees.
    :type RA: float
    :param DEC: Declination in degrees.
    :type DEC: float
    :return: Tuple of Galactic longitude and latitude.
    :rtype: tuple
    """
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').galactic
    return round(coords_gal.l.deg, 6), round(coords_gal.b.deg, 6)


@nvtx.annotate('radec_to_ecl', category='utils.astro')
def radec_to_ecl(RA, DEC):
    """
    Convert RA/Dec to Ecliptic coordinates.

    :param RA: Right ascension in degrees.
    :type RA: float
    :param DEC: Declination in degrees.
    :type DEC: float
    :return: Tuple of Ecliptic longitude and latitude.
    :rtype: tuple
    """
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').barycentricmeanecliptic
    return round(coords_gal.lon.deg, 6), round(coords_gal.lat.deg, 6)


@nvtx.annotate('date_to_jd', category='utils.astro')
def date_to_jd(dateobs):
    """
    Convert date to Julian Date.

    :param dateobs: Observation date and time.
    :type dateobs: str
    :return: Tuple of Julian Date and Modified Julian Date.
    :rtype: tuple
    """
    Date = Time(dateobs, scale='utc')
    return Date.jd, Date.mjd


@nvtx.annotate('get_ccw', category='utils.astro')
def get_ccw(hwcs):
    """
    Get the counter-clockwise rotation angle from WCS header.

    :param hwcs: WCS header.
    :type hwcs: dict
    :return: Counter-clockwise rotation angle in degrees.
    :rtype: float
    """
    cd11 = hwcs[HeaderKey.CD1_1.value]
    cd12 = hwcs[HeaderKey.CD1_2.value]
    cd21 = hwcs[HeaderKey.CD2_1.value]
    cd22 = hwcs[HeaderKey.CD2_2.value]
    det = cd11 * cd22 - cd12 * cd21
    if det >= 0:
        parity = 1.
    else:
        parity = -1.
    T = parity * cd11 + cd22
    A = parity * cd21 - cd12
    return -np.degrees(np.arctan2(A, T))


@nvtx.annotate('get_scale', category='utils.astro')
def get_scale(hwcs):
    """
    Get the image scale from WCS header.

    :param hwcs: WCS header.
    :type hwcs: dict
    :return: Image scale in arcseconds per pixel.
    :rtype: float
    """
    cd11 = hwcs[HeaderKey.CD1_1.value]
    cd12 = hwcs[HeaderKey.CD1_2.value]
    # cd21 = hwcs[HeaderKey.CD2_1.value]
    # cd22 = hwcs[HeaderKey.CD2_2.value]
    return np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600


@nvtx.annotate('plate_scale_px', category='utils.astro')
def plate_scale_px(microns, focal):
    """
    Calculate plate scale in arcseconds per pixel.

    :param microns: Pixel size in microns.
    :type microns: float
    :param focal: Focal length in mm.
    :type focal: float
    :return: Plate scale in arcseconds per pixel.
    :rtype: float
    """

    # pixel size in microns
    return plate_scale_mm(focal) * microns / 1000  # arcsec/px


@nvtx.annotate('plate_scale_mm', category='utils.astro')
def plate_scale_mm(focal):
    """
    Calculate plate scale in arcseconds per mm.

    :param focal: Focal length in mm.
    :type focal: float
    :return: Plate scale in arcseconds per mm.
    :rtype: float
    """
    # focal length in mm
    return 206265 / focal  # arcsec/mm


def copy_header_selectively(source_header, target_header=None, exclude_keywords=None):
    """
    Copies cards from a source FITS header to a target FITS header,
    excluding specified keywords like 'COMMENT' and 'HISTORY'.
    If target_header is None, a new Header object is created.

    :param source_header: The astropy.io.fits.Header object to copy from.
    :type source_header: astropy.io.fits.Header
    :param target_header: The astropy.io.fits.Header object to copy to.
                          If None, a new one is created.
    :type target_header: astropy.io.fits.Header or None
    :param exclude_keywords: A list or set of uppercase keyword strings to exclude.
                             Defaults to ['COMMENT', 'HISTORY', ''].
                             The empty string '' handles blank keyword cards.
    :type exclude_keywords: list or set or None
    :return: The target_header with cards copied from source_header.
    :rtype: astropy.io.fits.Header
    """
    if target_header is None:
        target_header = fits.Header()

    if exclude_keywords is None:
        # Default keywords to exclude. Note that FITS keywords are case-insensitive
        # but astropy stores them internally as uppercase.
        exclude_keywords = {'COMMENT', 'HISTORY', ''}  # Using a set for faster lookups
    elif isinstance(exclude_keywords, list):
        exclude_keywords = set(kw.upper() for kw in exclude_keywords)  # Ensure uppercase for set
    else:  # Assume it's already a set or compatible
        exclude_keywords = set(kw.upper() for kw in exclude_keywords)

    for card in source_header.cards:
        # card.keyword will be an uppercase string
        if card.keyword not in exclude_keywords:
            try:
                target_header[card.keyword] = (card.value, card.comment)
            except Exception as e:
                logger.debug(f"Warning: Could not copy card '{card.keyword}': {e}. Trying append.")
                try:
                    target_header.append(card.copy(), end=True)
                except Exception as e_append:
                    logger.debug(f"Error: Could not append card '{card.keyword}' after failed set: {e_append}")

    return target_header


if __name__ == '__main__':
    # Example usage (requires df.csv and potentially GPU context)
    # Ensure you have ASTROMETRY_API_KEY set for online attempt
    # and astrometry.net index files accessible for local attempts.
    try:
        df_example = pd.read_csv('/tmp/df.csv')
        # Assume image_shape is known, e.g., (2048, 2048)
        image_shape_example = (2048, 2048)

        # Example parameters (replace with real values)
        example_scale = 0.5081345802919708  # arcsec/px
        example_ra = 255.0  # degrees
        example_dec = 51.0  # degrees

        print("Attempting astrometry...")
        wcs_result = astrometrice2(
            df=df_example,
            scale=example_scale,
            central_ra=example_ra,
            central_dec=example_dec,
            image_shape=image_shape_example,
            sip_order=1,
            n_max=500,
            local_timeout=1000,  # Or a smaller value for testing timeouts
            online_timeout=70,
            astrometry_net_timeout=90
        )
        print("\nAstrometry Result:")
        if wcs_result:
            print("WCS Header Found:")
            for key, value in wcs_result.items():
                print(f"  {key}: {value}")
        else:
            print("No WCS solution found.")

    except AstrometrizationTimeoutError as e:
        print(f"\nAstrometry failed due to timeout or no solution: {e}")
    except FileNotFoundError:
        print("\nError: '/tmp/df.csv' not found. Cannot run example.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during example run: {e}", exc_info=True)
