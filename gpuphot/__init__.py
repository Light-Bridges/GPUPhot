# SPDX-License-Identifier: MIT
"""
Top-level gpuphot package initializer.

This module performs minor runtime patches (a safe rmtree for TemporaryDirectory)
and exposes commonly used subpackages. It intentionally avoids heavy runtime
initialization (like forcing a specific GPU) to keep import-time side-effects
minimal.
"""

import os
import subprocess
import tempfile
from types import MethodType


def _super_safe_rmtree(cls, name, ignore_errors=False, onerror=None):
    """
    Custom _rmtree for tempfile.TemporaryDirectory that handles:
    1. The top-level directory 'name' being a symbolic link.
    2. Avoids calling shutil.rmtree on the top-level directory if it's a plain directory,
       using 'rm -rf' instead as a fallback due to potential shutil.rmtree bugs/issues.
    """

    # Internal onerror handler (kept simple because we prefer an explicit rm -rf fallback)
    def _internal_onerror(func, path, exc_info):

        # If the caller provided a custom onerror, call it
        if onerror:
            onerror(func, path, exc_info)
        # If we should not ignore errors, re-raise
        elif not ignore_errors:
            exc_type, exc_value, tb = exc_info
            raise exc_value.with_traceback(tb)

    # --- Main logic of _super_safe_rmtree ---
    try:
        # First, check existence using lexists (do not follow symlinks)
        if not os.path.lexists(name):
            return

        # If it's a symlink, unlink it
        if os.path.islink(name):
            os.unlink(name)

        # If it's a directory (and not a symlink), fall back to a shell rm -rf
        elif os.path.isdir(name):
            cmd = ['rm', '-rf', name]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                if not ignore_errors:
                    raise OSError(f"'rm -rf' failed for {name}: {result.stderr}")

        # Otherwise, if it exists, treat it as a regular file and remove
        elif os.path.exists(name):
            os.remove(name)

    except Exception as e:
        if not ignore_errors:
            raise
        # If ignore_errors is True, suppress the exception here


# Apply the super enhanced patch to TemporaryDirectory if available
if hasattr(tempfile.TemporaryDirectory, '_rmtree'):
    tempfile.TemporaryDirectory._rmtree = MethodType(_super_safe_rmtree, tempfile.TemporaryDirectory)


def _auto_pin_blas_kernel():
    """Pin OPENBLAS_CORETYPE from the CPU's capabilities when nobody has.

    OpenBLAS picks its kernel from the host microarchitecture and the kernels
    round differently, which moves the CPU star clustering between local optima
    and shifts the zero point across machines (0.0053 mag measured).  The rule
    below is the measured equivalence group: AVX-512 hosts run SkylakeX, AVX2
    hosts run Haswell — A/B-verified to produce identical zero points; the only
    divergent kernel observed was Cooperlake, which this override avoids.

    OpenBLAS reads the variable when numpy first loads it, so this only works
    if gpuphot is imported before numpy; otherwise it backs off and the runtime
    warning in gpuphot.phot.psf tells the user to export it themselves.  An
    explicitly set value (compose, .env, shell) always wins — this never
    overwrites.
    """
    import sys
    try:
        if os.environ.get("OPENBLAS_CORETYPE", "").strip():
            return  # explicit configuration wins
        if "numpy" in sys.modules:
            return  # too late: OpenBLAS already picked its kernel
        if os.uname().machine != "x86_64":
            return  # ARM never showed cross-machine divergence; leave it be
        with open("/proc/cpuinfo") as fh:
            flags = fh.read()
        if "avx512f" in flags:
            os.environ["OPENBLAS_CORETYPE"] = "SkylakeX"
        elif "avx2" in flags:
            os.environ["OPENBLAS_CORETYPE"] = "Haswell"
    except Exception:
        pass  # never break import over an optimization


_auto_pin_blas_kernel()

# Optional: cupy float64 patch is provided in gpuphot.patch_cupy
# from . import patch_cupy

# Import commonly used subpackages and classes
from . import image_processor
from . import instrument_config_parser
from . import logger
from . import phot
from . import stats
from . import utils
from .image_processor import ImageProcessor, create_processor
from .instrument_config_parser import InstrumentConfigParser

# GPU configuration hints are intentionally commented out to avoid import-time side-effects
# gpu_id = os.environ.get('GPUPHOT_GPU_ID', '0')
# try:
#     from .logger.hierarchical_logging import setup_logger
#     logger = setup_logger(__name__)
#     cp.cuda.Device(int(gpu_id)).use()
#     logger.info(f"Using GPU: {cp.cuda.get_device_id()}")
# except Exception as e:
#     logger.error(f"Could not set the GPU {gpu_id}. Using GPU 0 by default.")
#     cp.cuda.Device(0).use()

__version__ = '1.0.1'

__all__ = [
    'logger',
    'phot',
    'stats',
    'utils',
    'instrument_config_parser',
    'image_processor',
    'ImageProcessor',
    'create_processor',
    'InstrumentConfigParser',
]
