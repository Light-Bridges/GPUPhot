import json
import os
import warnings
from enum import Enum

import jsonc
from astropy.io.fits import Header

from .logger.hierarchical_logging import setup_logger

warnings.simplefilter("once", UserWarning)
logger = setup_logger(__name__)


class HeaderKey(Enum):
    EXPT1 = "EXPT1"
    FILTER = "FILTER"
    TOTIMA = "TOTIMA"
    RDNOISE = "RDNOISE"
    SATLEVEL = "SATLEVEL"
    POINTRA = "POINTRA"
    POINTDEC = "POINTDEC"
    SITEELEV = "SITEELEV"
    SITELAT = "SITELAT"
    SITELONG = "SITELONG"
    DATE_OBS = "DATE-OBS"
    INMODEL = "INMODEL"
    PXSIZE = "PXSIZE"
    GAIN = "GAIN"
    NAXIS1 = "NAXIS1"
    NAXIS2 = "NAXIS2"
    FOCALEN = "FOCALEN"
    XBINNING = "XBINNING"
    BIASSTD = "BIASSTD"
    SITEALT = "SITEALT"
    CD1_1 = "CD1_1"
    CD1_2 = "CD1_2"
    CD2_1 = "CD2_1"
    CD2_2 = "CD2_2"


class ImageReduction(Enum):
    NEVER = "never"
    ALWAYS = "always"
    ON_FAILURE = "on_failure"


class DefaultConfig:
    DEFAULT_HEADER_KEYWORDS = {
        "exposure_time": HeaderKey.EXPT1.value,
        "filter": HeaderKey.FILTER.value,
        "n_images": HeaderKey.TOTIMA.value,
        "rdnoise": HeaderKey.RDNOISE.value,
        "satlevel": HeaderKey.SATLEVEL.value,
        "target_ra": HeaderKey.POINTRA.value,
        "target_dec": HeaderKey.POINTDEC.value,
        "site_elevation": HeaderKey.SITEELEV.value,
        "site_latitude": HeaderKey.SITELAT.value,
        "site_longitude": HeaderKey.SITELONG.value,
        "date_obs": HeaderKey.DATE_OBS.value,
        "inmodel": HeaderKey.INMODEL.value,
        "pxsize": HeaderKey.PXSIZE.value,
        "gain": HeaderKey.GAIN.value,
        "naxis1": HeaderKey.NAXIS1.value,
        "naxis2": HeaderKey.NAXIS2.value,
        "focalen": HeaderKey.FOCALEN.value,
        "xbinning": HeaderKey.XBINNING.value,
        "biasstd": HeaderKey.BIASSTD.value,
        "sitealt": HeaderKey.SITEALT.value,
        "cd1_1": HeaderKey.CD1_1.value,
        "cd1_2": HeaderKey.CD1_2.value,
        "cd2_1": HeaderKey.CD2_1.value,
        "cd2_2": HeaderKey.CD2_2.value,
    }

    DEFAULT_CAMERA_SPECS = {
        "exposure_time": None,
        "filter": None,
        "n_images": None,
        "rdnoise": None,
        "satlevel": None,
        "target_ra": None,
        "target_dec": None,
        "site_elevation": None,
        "site_latitude": None,
        "site_longitude": None,
        "date_obs": None,
        "inmodel": None,
        "pxsize": None,
        "gain": None,
        "naxis1": None,
        "naxis2": None,
        "focalen": None,
        "xbinning": None,
        "biasstd": None,
        "sitealt": None,
        "cd1_1": None,
        "cd1_2": None,
        "cd2_1": None,
        "cd2_2": None,
    }

    DEFAULT_PROCESSING_PARAMS = {
        'SP_filt': False,
        'CR_filt': False,
        'border': 20,
        'center_factor': 0.7,
        'pca_method': False,
        'tile_section': 300,
        'astrom': True,
        'tile_section_psf': 3000,
        'do_pad': True,
        'lum_gmag_coeff': 0.5,
        'lum_rmag_coeff': 0.5,
        'color_range': 0.6,
    }

    DEFAULT_IMAGE_REDUCTION = {
        "apply_reduction": ImageReduction.NEVER.value,
        "binning": 2,
        "crop_size": None,
        "center": None
    }


class HeaderTranslator:
    def __init__(self, header_keywords):
        """Initializes the header translator.

        :param header_keywords: Dictionary mapping custom header keys to internal keys.
        """
        self.translations = header_keywords
        self.default_translations = DefaultConfig.DEFAULT_HEADER_KEYWORDS
        self.reverse_translations = {v: k for k, v in self.translations.items()}
        self.warned_keys = set()

    def get_keyword(self, key):
        """Retrieves the internal key corresponding to an external key.

        :param key: External key for which to obtain the internal key.
        :return: Corresponding internal key or the external key if not found.
        """
        return self.translations.get(key, self.default_translations.get(key, key))

    def translate_header(self, header):
        """Translates a FITS header using defined translations.

        :param header: FITS header to translate (type astropy.io.fits.header.Header).
        :return: Translated header with internal keys.
        :raises TypeError: If the header is not of the correct type.
        """
        if not isinstance(header, Header):
            raise TypeError("Input header must be an astropy.io.fits.header.Header object")

        translated = header.copy()

        for internal_key, default_key in self.default_translations.items():
            user_key = self.translations.get(internal_key, default_key)
            if user_key in header:
                value = header[user_key]
                translated[default_key] = value
            else:
                default_value = DefaultConfig.DEFAULT_CAMERA_SPECS.get(internal_key)
                if default_value is not None:
                    translated.setdefault(default_key, default_value)
                    logger.debug(f"Added missing keyword '{default_key}' with default value '{default_value}'.")
                elif internal_key not in self.warned_keys:
                    warnings.warn(f"No default value found in camera specs for '{internal_key}'.", UserWarning)
                    self.warned_keys.add(internal_key)

        return translated

    def translate_back_header(self, header):
        """Translates a FITS header back to its original form.

        :param header: Translated FITS header (type astropy.io.fits.header.Header).
        :return: Original header with external keys.
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
        """Initializes the instrument configuration parser.

        :param config_dir: Directory where configuration files are located (optional).
                           If not provided, the default directory will be used.
        """
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'instrument_configs') if not config_dir else config_dir
        self.base_config = {
            "header_keywords": DefaultConfig.DEFAULT_HEADER_KEYWORDS.copy(),
            "camera_specs": DefaultConfig.DEFAULT_CAMERA_SPECS.copy(),
            "processing_params": DefaultConfig.DEFAULT_PROCESSING_PARAMS.copy(),
            "image_reduction": DefaultConfig.DEFAULT_IMAGE_REDUCTION.copy()
        }

    def _load_json_config(self, file_name):
        """Loads a JSON configuration file.

        :param file_name: Name of the JSON file to load.
        :return: Dictionary with loaded configuration or an empty dictionary on error.
        """
        file_path = os.path.join(self.config_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                return jsonc.load(file)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON in {file_name}: {e}")
            return {}

    def get_config(self, instrument_name):
        """Obtains configuration for a specific instrument.

        :param instrument_name: Name of the instrument for which to obtain configuration.
                                If 'default', default values will be used.
        :return: Dictionary with instrument configuration and header translator.
        """
        # Paso 1: Comenzar con la configuración base de DefaultConfig
        config = self.base_config.copy()

        # Paso 2: Actualizar con default.json si existe
        default_config = self._load_json_config('default.json')
        for section in config:
            if section in default_config:
                config[section].update(default_config[section])

        # Paso 3: Si se especifica un instrumento, intentar cargar su configuración
        if instrument_name and instrument_name != 'default':
            instrument_config = self._load_json_config(f"{instrument_name}.json")
            if instrument_config:
                for section in config:
                    if section in instrument_config:
                        config[section].update(instrument_config[section])
            else:
                logger.debug(f"Configuration for {instrument_name} not found. Using default values.")

        # Crear y añadir el traductor de headers
        config['header_translator'] = HeaderTranslator(config['header_keywords'])
        return config

    def generate_config_file(self, instrument_name):
        """Generates a JSON configuration file for an instrument.

        :param instrument_name: Name of the instrument for which to generate the configuration file.
                               The file will be saved in the configuration directory.
                               If it already exists, it will be overwritten.
        """
        file_path = os.path.join(self.config_dir, f"{instrument_name}.json")
        config_data = self.base_config.copy()

        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Guardar la configuración en un archivo JSON
        with open(file_path, 'w') as file:
            json.dump(config_data, file, indent=2)

        logger.debug(f"Configuration file generated: {file_path}")

    def generate_default_config(self):
        """Generates a default JSON configuration file."""
        self.generate_config_file('default')
