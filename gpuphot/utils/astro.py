from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
logger = setup_logger(__name__)

@hierarchical_debug(logger)
def plate_scale_px(microns, focal):


    """Calculate the plate scale in arcseconds per pixel.

    :param microns: Pixel size in micrometers.
    :type microns: float
    :param focal: Focal length in millimeters.
    :type focal: float

    
    """

    return plate_scale_mm(focal) * microns / 1000

@hierarchical_debug(logger)
def plate_scale_mm(focal):

    """Calculate the plate scale in arcseconds per millimeter.

    :param focal: Focal length in millimeters.
    :type focal: float

    
    """

    return 206265 / focal
