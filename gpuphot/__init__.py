import logging

logging.basicConfig(level=logging.DEBUG, format=
'%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from . import astrometry
from . import phot
from . import stats
from . import utils
