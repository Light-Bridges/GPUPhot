import logging
import time

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
logger = setup_logger(__name__)

@hierarchical_debug(logger)
def plate_scale_px(microns, focal):


    """
    Calculate the plate scale in arcseconds per pixel.

    Parameters
    ----------
    microns : float
        Pixel size in micrometers.
    focal : float
        Focal length in millimeters.

    Returns
    -------
    float
        Plate scale in arcseconds per pixel.
    """

    return plate_scale_mm(focal) * microns / 1000

@hierarchical_debug(logger)
def plate_scale_mm(focal):

    """
    Calculate the plate scale in arcseconds per millimeter.

    Parameters
    ----------
    focal : float
        Focal length in millimeters.

    Returns
    -------
    float
        Plate scale in arcseconds per millimeter.
    """

    return 206265 / focal
