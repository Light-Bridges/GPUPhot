import json
import os
import warnings
from enum import Enum
from typing import Dict, Any

import jsonc
from astropy.io.fits import Header

from .logger.hierarchical_logging import setup_logger

warnings.simplefilter("once", UserWarning)
logger = setup_logger(__name__)


class HeaderKey(Enum):
    BIASSTD = "BIASSTD"
    CD1_1 = "CD1_1"
    CD1_2 = "CD1_2"
    CD2_1 = "CD2_1"
    CD2_2 = "CD2_2"
    DATE_OBS = "DATE-OBS"
    EXPT1 = "EXPT1"
    FILTER = "FILTER"
    FOCALEN = "FOCALEN"
    GAIN = "GAIN"
    INMODEL = "INMODEL"
    NAXIS1 = "NAXIS1"
    NAXIS2 = "NAXIS2"
    POINTDEC = "POINTDEC"
    POINTRA = "POINTRA"
    PXSIZE = "PXSIZE"
    RDNOISE = "RDNOISE"
    SATLEVEL = "SATLEVEL"
    # SITEALT = "SITEALT"     # TODO: Remove this key when all code use ImageProcessor.process_image
    SITEELEV = "SITEELEV"
    SITELAT = "SITELAT"
    SITELONG = "SITELONG"
    TOTIMA = "TOTIMA"
    XBINNING = "XBINNING"


class ImageReduction(Enum):
    ALWAYS = "always"
    NEVER = "never"
    ON_FAILURE = "on_failure"


class DefaultConfig:
    DEFAULT_HEADER_KEYWORDS = {
        "biasstd": HeaderKey.BIASSTD.value,
        "cd1_1": HeaderKey.CD1_1.value,
        "cd1_2": HeaderKey.CD1_2.value,
        "cd2_1": HeaderKey.CD2_1.value,
        "cd2_2": HeaderKey.CD2_2.value,
        "date_obs": HeaderKey.DATE_OBS.value,
        "exposure_time": HeaderKey.EXPT1.value,
        "filter": HeaderKey.FILTER.value,
        "focalen": HeaderKey.FOCALEN.value,
        "gain": HeaderKey.GAIN.value,
        "inmodel": HeaderKey.INMODEL.value,
        "n_images": HeaderKey.TOTIMA.value,
        "naxis1": HeaderKey.NAXIS1.value,
        "naxis2": HeaderKey.NAXIS2.value,
        "pxsize": HeaderKey.PXSIZE.value,
        "rdnoise": HeaderKey.RDNOISE.value,
        "satlevel": HeaderKey.SATLEVEL.value,
        "site_elevation": HeaderKey.SITEELEV.value,
        "site_latitude": HeaderKey.SITELAT.value,
        "site_longitude": HeaderKey.SITELONG.value,
        # "sitealt": HeaderKey.SITEALT.value,
        "target_dec": HeaderKey.POINTDEC.value,
        "target_ra": HeaderKey.POINTRA.value,
        "xbinning": HeaderKey.XBINNING.value,
    }

    DEFAULT_CAMERA_SPECS = {
        "biasstd": None,
        "cd1_1": None,
        "cd1_2": None,
        "cd2_1": None,
        "cd2_2": None,
        "date_obs": None,
        "exposure_time": None,
        "filter": None,
        "focalen": None,
        "gain": None,
        "inmodel": None,
        "n_images": None,
        "naxis1": None,
        "naxis2": None,
        "pxsize": None,
        "rdnoise": None,
        "satlevel": None,
        "site_elevation": None,
        "site_latitude": None,
        "site_longitude": None,
        # "sitealt": None,
        "target_dec": None,
        "target_ra": None,
        "xbinning": None,
    }

    DEFAULT_PROCESSING_PARAMS = {
        'border': 20,
        'center_factor': 0.7,
        'color_range': 0.6,
        'CR_filt': False,
        'do_pad': True,
        'lum_gmag_coeff': 0.5,
        'lum_rmag_coeff': 0.5,
        'max_stars_ref': 15,
        'min_conv_snr': 300,
        'pca_method': True,
        'SP_filt': True,
        'tile_section': 1000,
        'tile_section_psf': 3000,
        'zp_maxmag': 21,
    }

    DEFAULT_IMAGE_REDUCTION = {
        "apply_reduction": ImageReduction.NEVER.value,
        "binning": {
            "factor": 2,
            "method": "sum"  # sum, median
        },
        "center": None,
        "crop_size": None,
    }

    # Configuration for forced values to override header and default specs
    DEFAULT_FORCED_VALUES = {}

    # Default filter mapping (Header Value -> Internal Code)
    # This covers minimal standard naming. Specific instruments should override this
    # in their JSON file if their filter wheel names are different (e.g., "Red" instead of "SDSSr").
    DEFAULT_FILTER_MAP = {
        # Internal codes as identity (if header matches code, keep it)
        "Lum": "Lum",
        "Open": "Open",
        "SDSSu": "SDSSu",
        "SDSSg": "SDSSg",
        "SDSSr": "SDSSr",
        "SDSSi": "SDSSi",
        "SDSSzs": "SDSSzs",
        "SDSSy": "SDSSy",
        # Common aliases
        "Clear": "Open",
        "L": "Lum",
        "w": "Lum"
    }


class HeaderTranslator:
    def __init__(self, header_keywords, camera_specs, forced_values=None, filter_map=None):
        """
        Initialize the header translator.

        :param header_keywords: Dictionary mapping internal keys to user-specific header keys.
        :type header_keywords: dict
        :param camera_specs: Dictionary with default values for missing keys.
        :type camera_specs: dict
        :param forced_values: Dictionary with values that strictly override any header data.
        :type forced_values: dict or None
        :param filter_map: Dictionary mapping user filter values (from header) to internal standard codes.
        :type filter_map: dict or None
        """
        self.translations = header_keywords
        self.camera_specs = camera_specs
        self.forced_values = forced_values if forced_values is not None else {}
        self.filter_map = filter_map if filter_map is not None else DefaultConfig.DEFAULT_FILTER_MAP.copy()

        self.default_translations = DefaultConfig.DEFAULT_HEADER_KEYWORDS
        self.reverse_translations = {v: k for k, v in self.translations.items()}
        self.warned_keys = set()

        # Identify the internal key used for filters to apply the map later
        self.internal_filter_key = "filter"

    def get_keyword(self, key):
        """
        Retrieve the internal key corresponding to an external key.

        :param key: External key for which to obtain the internal key.
        :type key: str
        :return: Corresponding internal key or the external key if not found.
        :rtype: str
        """
        return self.translations.get(key, self.default_translations.get(key, key))

    def translate_header(self, header):
        """
        Translate a FITS header using defined translations, enforcing forced values,
        translating filter names, and filling in missing values from camera specifications.

        Priority order:
        1. Forced Values (overrides everything)
        2. Header Values (using mapped keys + filter name translation)
        3. Camera Specs (defaults)

        :param header: FITS header to translate.
        :type header: astropy.io.fits.header.Header
        :return: Translated header with internal keys and standard filter names.
        :rtype: astropy.io.fits.header.Header
        :raises TypeError: If the header is not of the correct type.
        """
        if not isinstance(header, Header):
            raise TypeError("Input header must be an astropy.io.fits.header.Header object")

        translated = header.copy()

        # Iterate over our standard internal keys
        for internal_key, standard_fits_key in self.default_translations.items():

            # --- PRIORITY 1: FORCED VALUES ---
            # If a value is explicitly forced in the config, use it and ignore the header.
            if internal_key in self.forced_values:
                forced_val = self.forced_values[internal_key]
                translated[standard_fits_key] = forced_val
                logger.debug(f"Keyword '{standard_fits_key}' was FORCED to value '{forced_val}' by configuration.")
                continue  # Skip to the next key, we are done with this one.

            # --- PRIORITY 2: HEADER VALUES ---
            # Determine which key to look for in the input header (user's key)
            user_key = self.translations.get(internal_key, standard_fits_key)

            # Retrieve value from header if it exists
            value_from_header = None
            if user_key in header:
                value_from_header = header[user_key]
            elif user_key == standard_fits_key and standard_fits_key in header:
                value_from_header = header[standard_fits_key]

            if value_from_header is not None:
                # SPECIAL HANDLING FOR FILTERS:
                # If this is the filter keyword, we must check the filter_map to standardize the name.
                # E.g., Header says "R_Filter" -> Map says "SDSSr".
                if internal_key == self.internal_filter_key:
                    # Normalize string (remove spaces) for lookup
                    raw_val = str(value_from_header).strip()
                    # Look up in map. If not found, keep the raw value.
                    std_val = self.filter_map.get(raw_val, raw_val)

                    if raw_val != std_val:
                        logger.debug(f"Translating filter '{raw_val}' to standard code '{std_val}'.")

                    translated[standard_fits_key] = std_val
                else:
                    # Normal copy for non-filter keys
                    translated[standard_fits_key] = value_from_header

            # --- PRIORITY 3: CAMERA SPECS (DEFAULTS) ---
            else:
                # If not in header, look for a default value in camera specs.
                default_value = self.camera_specs.get(internal_key)

                if default_value is not None:
                    translated[standard_fits_key] = default_value
                    logger.debug(f"Keyword '{user_key}' not found. Added standard keyword '{standard_fits_key}' "
                                 f"with default value '{default_value}' from camera specs.")
                elif internal_key not in self.warned_keys:
                    # If no default value exists, warn the user.
                    warnings.warn(f"Keyword '{user_key}' not found in header and no default value "
                                  f"is defined in camera specs for '{internal_key}'.", UserWarning)
                    self.warned_keys.add(internal_key)

        return translated

    def translate_back_header(self, header):
        """
        Translate a FITS header back to its original form.

        :param header: Translated FITS header.
        :type header: astropy.io.fits.header.Header
        :return: Original header with external keys.
        :rtype: astropy.io.fits.header.Header
        :raises TypeError: If the header is not of the correct type.
        """
        if not isinstance(header, Header):
            raise TypeError("Input header must be an astropy.io.fits.header.Header object")

        original = header.copy()

        for internal_key, default_key in self.default_translations.items():
            if default_key in header:
                user_key = self.translations.get(internal_key, default_key)
                if user_key != default_key:
                    if user_key not in original:
                        original[user_key] = header[default_key]
                    if default_key != user_key:
                        del original[default_key]

        return original


class InstrumentConfigParser:
    def __init__(self, config_dir=None):
        """
        Initialize the instrument configuration parser.

        :param config_dir: Directory where configuration files are located (optional).
                           If not provided, the default directory will be used.
        :type config_dir: str or None
        """
        logger.debug(f'Initializing InstrumentConfigParser with config_dir: {config_dir}')
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'instrument_configs') if not config_dir else config_dir

        # Initialize base configuration with all sections
        from typing import Dict, Any  # Import for Type Hinting
        self.base_config: Dict[str, Any] = {
            "header_keywords": DefaultConfig.DEFAULT_HEADER_KEYWORDS.copy(),
            "camera_specs": DefaultConfig.DEFAULT_CAMERA_SPECS.copy(),
            "processing_params": DefaultConfig.DEFAULT_PROCESSING_PARAMS.copy(),
            "image_reduction": DefaultConfig.DEFAULT_IMAGE_REDUCTION.copy(),
            "forced_values": DefaultConfig.DEFAULT_FORCED_VALUES.copy(),
            "filter_map": DefaultConfig.DEFAULT_FILTER_MAP.copy()
        }

    def _load_json_config(self, file_name):
        """
        Load a JSON configuration file.

        :param file_name: Name of the JSON file to load.
        :type file_name: str
        :return: Dictionary with loaded configuration or an empty dictionary on error.
        :rtype: dict
        """
        file_path = os.path.join(self.config_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return jsonc.loads(content)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Format error in JSON configuration file: '{file_name}'")

            # Split content into lines to display the problematic line
            lines = content.splitlines()
            if 0 < e.lineno <= len(lines):
                error_line = lines[e.lineno - 1]
                # Create a pointer to highlight the exact error column
                pointer = ' ' * (e.colno - 1) + '^'

                logger.error(f"  > Line {e.lineno}, Column {e.colno}: {e.msg}")
                logger.error(f"  > {error_line}")
                logger.error(f"  > {pointer}")
                logger.error("  > Hint: Ensure all strings (both keys and values) use double quotes (\"). "
                             "Also, check for misplaced or trailing commas.")
            else:
                # Fallback in case line/column numbers are out of range
                logger.error(f"  > Error details: {e}")

            return {}
        except Exception as e:
            # Catch-all for any other unexpected errors during file reading
            logger.error(f"Could not read configuration file '{file_name}': {e}")
            return {}

    def get_config(self, instrument_name):
        """
        Obtain configuration for a specific instrument.

        :param instrument_name: Name of the instrument for which to obtain configuration.
                                If 'default', default values will be used.
        :type instrument_name: str
        :return: Dictionary with instrument configuration and header translator.
        :rtype: dict
        """
        # Step 1: Start with base configuration
        config: Dict[str, Any] = self.base_config.copy()

        # Step 2: Update with default.json if it exists
        default_config = self._load_json_config('default.json')
        for section in config:
            if section in default_config:
                config[section].update(default_config[section])

        # Step 3: If an instrument is specified, try to load its configuration
        if instrument_name and instrument_name != 'default':
            instrument_config = self._load_json_config(f"{instrument_name}.json")
            if instrument_config:
                for section in config:
                    if section in instrument_config:
                        config[section].update(instrument_config[section])
            else:
                logger.warning(f"Configuration for {instrument_name} not found. Using default values.")

        # Create and add the header translator, passing the forced_values and filter_map
        config['header_translator'] = HeaderTranslator(
            header_keywords=config['header_keywords'],
            camera_specs=config['camera_specs'],
            forced_values=config.get('forced_values', {}),
            filter_map=config.get('filter_map', {})
        )
        return config

    def generate_config_file(self, instrument_name):
        """
        Generate a JSON configuration file for an instrument.

        :param instrument_name: Name of the instrument for which to generate the configuration file.
        :type instrument_name: str
        """
        file_path = os.path.join(self.config_dir, f"{instrument_name}.json")
        config_data = self.base_config.copy()

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save configuration to a JSON file
        with open(file_path, 'w') as file:
            json.dump(config_data, file, indent=2)

        logger.debug(f"Configuration file generated: {file_path}")

    def generate_default_config(self):
        """
        Generate a default JSON configuration file.
        """
        self.generate_config_file('default')
