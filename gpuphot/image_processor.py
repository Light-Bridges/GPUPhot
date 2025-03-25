import nvtx

from .instrument_config_parser import InstrumentConfigParser
from .logger.hierarchical_logging import setup_logger
from .phot.photo_gpu import process_image

logger = setup_logger(__name__)


class ImageProcessor:
    def __init__(self, instrument_name, config_dir=None):
        """
        Initialize the image processor with configuration for a specific instrument.

        :param instrument_name: Name of the instrument for which to configure the processor.
        :type instrument_name: str
        :param config_dir: Directory where configuration files are located (optional).
                           If not provided, the default directory will be used.
        :type config_dir: str or None
        """
        self.instrument_name = instrument_name
        self.config_parser = InstrumentConfigParser(config_dir)
        self.config = self.config_parser.get_config(instrument_name)
        self.processing_params = self.config['processing_params']
        self.camera_params = self.config['camera_specs']
        self.header_translator = self.config['header_translator']

        try:

            extra = {
                'instrument_name': instrument_name,
                'processing_params': self.processing_params,
                'camera_params': self.camera_params,
                'header_keywords': self.config['header_keywords']
            }

            logger.debug(f"Image processor initialized for: {extra}", extra=extra)
        except Exception:
            pass

    ### # @hierarchical_debug(logger)
    @nvtx.annotate('ImageProcessor.process_image', category='image_processor.ImageProcessor')
    def process_image(self, imdata, imheader, header_descriptions=None, **kwargs):
        """
        Process an image using the specified parameters and translate headers.

        :param imdata: The image data to be processed.
        :type imdata: numpy.ndarray
        :param imheader: The original FITS header associated with the image.
        :type imheader: astropy.io.fits.header.Header
        :param header_descriptions: Descriptions of the header keywords.
        :type header_descriptions: dict or None
        :param kwargs: Additional parameters for processing that override default settings.
        :return: A tuple containing the data frame of processed results and the translated original header.
        :rtype: tuple(pandas.DataFrame, astropy.io.fits.header.Header)
        """
        # Traduce el header a las keywords estándar
        translated_header = self.header_translator.translate_header(imheader)

        # Combina los parámetros de procesamiento con los kwargs
        params = {**self.processing_params, **kwargs}

        # Llama a la función process_image con el header traducido
        phot_df, hwcs = process_image(imdata, translated_header, header_descriptions, **params)

        ## Traduce el header procesado de vuelta a las keywords originales del usuario
        # original_hwcs = self.header_translator.translate_back_header(hwcs)

        return phot_df, hwcs

    @nvtx.annotate('get_header_info', category='image_processor.ImageProcessor')
    def get_header_info(self, header):
        """
        Retrieve translated header information.

        :param header: The FITS header to be translated.
        :type header: astropy.io.fits.header.Header
        :return: The translated header with standard keywords.
        :rtype: astropy.io.fits.header.Header
        """
        return self.header_translator.translate_header(header)


# Function to create the processor
@nvtx.annotate('create_processor', category='image_processor')
def create_processor(instrument_name, config_dir=None):
    """
    Create an instance of ImageProcessor for a specified instrument.

    :param instrument_name: Name of the instrument for which to create a processor.
    :type instrument_name: str
    :param config_dir: Directory where configuration files are located (optional).
                       If not provided, the default directory will be used.
    :type config_dir: str or None
    :return: An instance of ImageProcessor configured for the specified instrument.
    :rtype: ImageProcessor
    """
    return ImageProcessor(instrument_name, config_dir)


if __name__ == '__main__':
    test = create_processor('test')
    test.process_image(None, None, None)
    from astropy.io.fits import Header

    print(test.get_header_info(Header()))

    InstrumentConfigParser().generate_config_file('test')
