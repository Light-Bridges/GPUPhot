import logging

logging.basicConfig(level=logging.DEBUG, format=
'%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
__all__ = ['atlas', 'reduction', 's_util', 'subpixel', 'subpixel_masked']
