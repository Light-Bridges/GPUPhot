# gpuphot.utils

Utility helpers used by the gpuphot pipeline. This package provides a number
of focused helper modules used throughout the project. Each module focuses on
one responsibility and is intentionally small to ease testing and reuse.

Modules
- `astro.py` – Astrometry helpers and solver management. Handles Astrometry.net
  solver initialization, cache location, and utility functions for conversions
  between coordinate systems. Note: initialization may download large index
  files (~10s of GB) and should be done outside of short-lived worker processes.

- `catalog.py` – Catalog queries and coordinate crossmatching. Wraps Vizier
  queries and provides CPU (KDTree) and optional GPU (cuML) implementations for
  nearest-neighbor crossmatching.

- `gpu.py` – GPU memory management helpers. Functions to inspect memory pool
  usage, free memory, move arrays between host and device, and adaptive
  cleanup strategies. These utilities are critical when running within GPU
  memory constrained environments (e.g., small GPUs or multi-tenant nodes).

- `headers.py` – FITS header utilities used to insert astrometry and photometry
  metadata into image headers in a consistent way.

- `timeout.py` – Small, process-based executor wrapper to run functions with a
  timeout. Used to guard network or blocking catalog calls.

Notes & Usage
- Many modules support both NumPy (CPU) and CuPy (GPU) arrays. The
  implementation typically prefers GPU paths when RAPIDS/cuML and CuPy are
  available; otherwise it falls back to CPU implementations.
- Ensure the runtime environment contains the expected GPU libraries when
  running GPU-accelerated code (CuPy, cuML). Where GPU features are optional,
  the code logs warnings and falls back to CPU implementations.
- When you encounter a `FIXERS.md` file in a subpackage, it lists
  non-documentation code improvements that were intentionally deferred during
  the documentation-only pass.
