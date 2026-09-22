"""
Shared pytest fixtures and markers for gpuphot tests.

Detects CuPy NVRTC compilation issues (e.g., fp8_e8m0 incompatibility)
that affect certain operations like cp.nanmean, cupyx.scipy.ndimage, etc.
"""

import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Detect if CuPy NVRTC compilation is broken (fp8_e8m0 issue with certain
# CuPy/CUDA toolkit combinations). We test by calling cp.nanmean which
# triggers JIT compilation.
NVRTC_WORKS = False
if HAS_CUPY:
    try:
        arr = cp.array([1.0, 2.0, float('nan')])
        cp.nanmean(arr)
        NVRTC_WORKS = True
        del arr
    except Exception:
        pass

requires_nvrtc = pytest.mark.skipif(
    not NVRTC_WORKS,
    reason="CuPy NVRTC compilation broken (fp8_e8m0 incompatibility)"
)
