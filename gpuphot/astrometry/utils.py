import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_if_header_already_post_processed(header, postprocess):
    """
    Check if a FITS header contains a specific post-processing comment.

    Parameters
    ----------
    header : dict
        FITS header containing metadata and comments.
    postprocess : str
        Post-processing string to check in the header comments.

    Returns
    -------
    bool
        True if the post-processing string is found in the header comments, False otherwise.
    """
    if 'COMMENT' not in header:
        logger.warning('No comments found in header.')
        return False
    comments = list()
    for comment in header['COMMENT']:
        comments.append(comment.strip('   '))
    comments = set(comments)
    return postprocess in comments


def get_scale(hwcs):
    """
    Calculate the pixel scale from the WCS transformation matrix.

    Parameters
    ----------
    hwcs : dict
        Dictionary containing WCS transformation matrix elements.

    Returns
    -------
    float
        Pixel scale in arcseconds per pixel.
    """
    cd11 = hwcs['CD1_1']
    cd12 = hwcs['CD1_2']
    return np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600


def get_ccw(hwcs):
    """
    Calculate the counter-clockwise rotation angle from the WCS transformation matrix.

    Parameters
    ----------
    hwcs : dict
        Dictionary containing WCS transformation matrix elements.

    Returns
    -------
    float
        Counter-clockwise rotation angle in degrees.
    """
    cd11 = hwcs['CD1_1']
    cd12 = hwcs['CD1_2']
    cd21 = hwcs['CD2_1']
    cd22 = hwcs['CD2_2']
    det = cd11 * cd22 - cd12 * cd21
    parity = 1.0 if det >= 0 else -1.0
    T = parity * cd11 + cd22
    A = parity * cd21 - cd12
    return -np.degrees(np.arctan2(A, T))
