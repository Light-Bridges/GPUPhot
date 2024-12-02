import json
import os

import jsonc


class HeaderTranslator:
    def __init__(self, header_keywords):
        self.translations = header_keywords
        self.default_translations = {
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
            "sitealt": "SITEALT"
        }
        self.reverse_translations = {v: k for k, v in self.translations.items()}

    def get_keyword(self, key):
        return self.translations.get(key, self.default_translations.get(key))

    def translate_header(self, header):
        translated = {}
        for key, value in self.default_translations.items():
            translated_key = self.get_keyword(key)
            if translated_key in header:
                translated[key] = header[translated_key]
        return translated

    def translate_back_header(self, header):
        original = {}
        for key, value in header.items():
            original_key = self.reverse_translations.get(key, key)
            original[original_key] = value
        return original


class InstrumentConfigParser:
    def __init__(self, config_dir=None):
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'instrument_configs') if not config_dir else config_dir
        self.default_config = self._load_config('default')

    def _load_config(self, instrument_name):
        file_path = os.path.join(self.config_dir, f"{instrument_name}.json")
        try:
            with open(file_path, 'r') as file:
                return jsonc.load(file)  # Carga el JSON con comentarios
        except FileNotFoundError:
            if instrument_name != 'default':
                print(f"Configuration for {instrument_name} not found. Using default values.")
                return self.default_config
            else:
                raise FileNotFoundError("Default configuration file not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            raise

    def get_config(self, instrument_name):
        instrument_config = self._load_config(instrument_name)
        config = {**self.default_config, **instrument_config}
        config['header_translator'] = HeaderTranslator(config['header_keywords'])
        return config
