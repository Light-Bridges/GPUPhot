from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
logger = setup_logger(__name__)

@hierarchical_debug(logger)
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
