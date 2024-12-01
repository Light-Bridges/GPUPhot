from .instrument_config_parser import InstrumentConfigParser


class ImageProcessor:
    def __init__(self, instrument_name):
        config_parser = InstrumentConfigParser()
        self.config = config_parser.get_config(instrument_name)
        self.processing_params = self.config['processing_params']

    def process_image(self, imdata, imheader, **kwargs):
        # Use the values from the instrument configuration, but allow overriding with kwargs
        params = {**self.processing_params, **kwargs}

        # Here would go the rest of your image processing logic
        # ...
        df_phot = None # Fixme

        return df_phot, imheader

    def get_header_info(self, header):
        return {key: header[value] for key, value in self.config['header_keywords'].items()}


# Function to create the processor
def create_processor(instrument_name):
    return ImageProcessor(instrument_name)


# # Usage:
# processor = create_processor("iKon936")
# df_phot, imheader = processor.process_image(imdata, imheader, astrom=astrom)