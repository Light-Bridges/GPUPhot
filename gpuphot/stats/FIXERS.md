# FIXERS for gpuphot/stats

This file lists non-documentation issues discovered while performing the
English documentation pass. These are suggestions to address in a separate
code-fix branch.

- `reduction.py`
  - Long f-strings and log messages exceed 120 columns; consider wrapping and/or helper formatting functions.
  - NVTX decorator shim and fallback may mask import issues; consider making NVTX optional via a runtime flag.

- `subpixel.py` / `subpixel_masked.py`
  - The implementations assume CuPy is available. Add unit tests for CPU fallback behavior (NumPy) and document expected CuPy versions.
  - Consider guarding heavy-memory operations to provide clearer errors when GPU memory is insufficient.

- `s_util.py`
  - No functional issues found; small helper suggests adding a test to validate dtype handling for string and dtype inputs.

- `general`
  - Run the project formatter and linter (Black/isort/flake8) in a separate branch to clean up line-length issues introduced when translating docstrings.

If you'd like, I can prepare a separate branch with a small PR that applies some of these low-risk fixes after the documentation pass is merged.
