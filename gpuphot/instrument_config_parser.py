import json
import os

class InstrumentConfigParser:
    def __init__(self, config_dir='instrument_configs'):
        self.config_dir = config_dir
        self.default_config = self._load_config('default')

    def _load_config(self, instrument_name):
        file_path = os.path.join(self.config_dir, f"{instrument_name}.json")
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            if instrument_name != 'default':
                print(f"Configuration for {instrument_name} not found. Using default values.")
                return self.default_config
            else:
                raise FileNotFoundError("Default configuration file not found.")

    def get_config(self, instrument_name):
        instrument_config = self._load_config(instrument_name)
        # Combine the instrument configuration with default values
        return {**self.default_config, **instrument_config}