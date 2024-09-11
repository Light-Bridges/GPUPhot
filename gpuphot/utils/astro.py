import logging
import time

logging.basicConfig(level=logging.DEBUG, format=
'%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import logging

logger = logging.getLogger(__name__)


def plate_scale_px(microns, focal):
    logger.debug(
        f'Iniciando función plate_scale_px(microns={microns}, focal={focal})')
    start_time = time.time()
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
    logger.debug(
        f'Función plate_scale_px completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return plate_scale_mm(focal) * microns / 1000


def plate_scale_mm(focal):
    logger.debug(f'Iniciando función plate_scale_mm(focal={focal})')
    start_time = time.time()
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
    logger.debug(
        f'Función plate_scale_mm completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return 206265 / focal
