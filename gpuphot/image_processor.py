from .instrument_config_parser import InstrumentConfigParser
from .logger.hierarchical_logging import setup_logger, hierarchical_debug
from .phot.photo_gpu import process_image

logger = setup_logger(__name__)


class ImageProcessor:
    def __init__(self, instrument_name):
        config_parser = InstrumentConfigParser()
        self.config = config_parser.get_config(instrument_name)
        self.processing_params = self.config['processing_params']
        self.camera_params = self.config['camera_specs']
        self.header_translator = self.config['header_translator']

    @hierarchical_debug(logger)
    def process_image(self, imdata, imheader, header_descriptions, **kwargs):
        # Traduce el header a las keywords estándar
        translated_header = self.header_translator.translate_header(imheader)
        print(translated_header)

        # Combina los parámetros de procesamiento con los kwargs
        params = {**self.processing_params, **kwargs}

        # Llama a la función process_image con el header traducido
        dfm, processed_header = process_image(imdata, translated_header, header_descriptions, **params)

        # Traduce el header procesado de vuelta a las keywords originales del usuario
        original_header = self.header_translator.translate_back_header(processed_header)
        print(original_header)
        return dfm, original_header

    def get_header_info(self, header):
        return self.header_translator.translate_header(header)


# Function to create the processor
def create_processor(instrument_name):
    return ImageProcessor(instrument_name)


