import json
import os
import warnings

import jsonc
from astropy.io.fits import Header

from .logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


class DefaultConfig:
    DEFAULT_HEADER_KEYWORDS = {
        "exposure_time": "EXPT1",
        "filter": "FILTER",
        "n_images": "TOTIMA",
        "rdnoise": "RDNOISE",
        "satlevel": "SATLEVEL",
        "target_ra": "POINTRA",
        "target_dec": "POINTDEC",
        "site_elevation": "SITEELEV",
        "site_latitude": "SITELAT",
        "site_longitude": "SITELONG",
        "date_obs": "DATE-OBS",
        "inmodel": "INMODEL",
        "pxsize": "PXSIZE",
        "gain": "GAIN",
        "naxis1": "NAXIS1",
        "naxis2": "NAXIS2",
        "focalen": "FOCALEN",
        "xbinning": "XBINNING",
        "biasstd": "BIASSTD",
        "sitealt": "SITEALT",
    }

    DEFAULT_CAMERA_SPECS = {
        "pixel_size": 15,
        "gain": 1.0
    }

    DEFAULT_PROCESSING_PARAMS = {
        'SP_filt': True,
        'CR_filt': False,
        'border': 50,
        'center_factor': 0.7,
        'pca_method': True,
        'tile_section': 1000,
        'astrom': True,
        'tile_section_psf': 2500,
        'do_pad': False,
    }


class HeaderTranslator:
    def __init__(self, header_keywords):
        self.translations = header_keywords
        self.default_translations = DefaultConfig.DEFAULT_HEADER_KEYWORDS
        self.reverse_translations = {v: k for k, v in self.translations.items()}

    def get_keyword(self, key):
        return self.translations.get(key, self.default_translations.get(key, key))

    def translate_header(self, header):
        if not isinstance(header, Header):
            raise TypeError("Input header must be an astropy.io.fits.header.Header object")

        translated = Header()
        for key, value in header.items():
            internal_key = self.reverse_translations.get(key)
            if internal_key:
                default_key = self.default_translations[internal_key]
                translated[default_key] = value
            else:
                translated[key] = value

        for internal_key, default_key in self.default_translations.items():
            user_key = self.translations.get(internal_key, default_key)
            if user_key not in header:
                warnings.warn(f"Variable '{user_key}' not found in header. This may cause issues.", UserWarning)

        return translated

    def translate_back_header(self, header):
        if not isinstance(header, Header):
            raise TypeError("Input header must be an astropy.io.fits.header.Header object")

        original = Header()
        for key, value in header.items():
            for internal_key, default_key in self.default_translations.items():
                if key == default_key:
                    user_key = self.translations[internal_key]
                    original[user_key] = value
                    break
            else:
                original[key] = value
        return original


class InstrumentConfigParser:
    def __init__(self, config_dir=None):
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'instrument_configs') if not config_dir else config_dir
        self.base_config = {
            "header_keywords": DefaultConfig.DEFAULT_HEADER_KEYWORDS.copy(),
            "camera_specs": DefaultConfig.DEFAULT_CAMERA_SPECS.copy(),
            "processing_params": DefaultConfig.DEFAULT_PROCESSING_PARAMS.copy()
        }

    def _load_json_config(self, file_name):
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
        file_path = os.path.join(self.config_dir, f"{instrument_name}.json")
        config_data = self.base_config.copy()

        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Guardar la configuración en un archivo JSON
        with open(file_path, 'w') as file:
            json.dump(config_data, file, indent=2)

        logger.debug(f"Configuration file generated: {file_path}")

    def generate_default_config(self):
        self.generate_config_file('default')
