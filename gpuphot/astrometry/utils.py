import logging
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_if_header_already_post_processed(header, postprocess):
    logger.debug(
        f'Iniciando función get_if_header_already_post_processed(header={header}, postprocess={postprocess})'
    )
    start_time = time.time()
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
    logger.debug(
        f'Función get_if_header_already_post_processed completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return postprocess in comments
