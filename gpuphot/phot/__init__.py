# SPDX-License-Identifier: MIT
"""
gpuphot.phot package

This package exposes photometry-related modules: background estimation,
convolution utilities, PSF utilities, and high-level photometry pipelines.
The submodules are optimized to use CuPy for GPU acceleration with CPU
fallbacks where appropriate.
"""

from . import background
from . import conv
from . import photo_gpu
from . import psf
from . import utils

__all__ = ['background', 'utils', 'conv', 'photo_gpu', 'psf']
