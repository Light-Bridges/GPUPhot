# FIXERS for gpuphot/phot

This file lists non-documentation fixes that were discovered or suggested
while performing the documentation pass. These items should be addressed in a
separate code-fix phase. Each entry contains the file, the issue, and a
suggested change.

- file: conv.py
  issue: Inline Spanish comment and an unguarded comment mentioning memory growth
  suggestion: Consider adding a note to the README about memory implications when padding.

- file: psf.py
  issue: Several inline Spanish comments remain (e.g., 'Usar un contexto', 'Liberar') and some
         operations assume `.get()` will always succeed which may raise on OOM.
  suggestion: Review GPU<->CPU transfers and add robust OOM handling and clearer comments.

- file: photo_gpu.py
  issue: Very large file with mixed CPU/GPU paths and many commented code blocks. Some variables
         such as `cov_nan` and `sample_im` are used before declaration in the file, relying on
         top-to-bottom ordering; ensure function ordering is safe or import them from helpers.
  suggestion: Refactor into smaller modules (e.g., phot/processing.py, phot/aperture.py), and add
              explicit unit tests for transfer points and memory management.

- file: photo_gpu.py
  issue: Several places contain Spanish comments and `TODO` notes referencing individuals.
  suggestion: Replace personal TODOs and Spanish comments with neutral, English TODOs and create
              issues in the project tracker referencing them.

- file: psf.py
  issue: GPU implementation of grouping constructs a Python list comprehension using `for gid in valid_group_ids` which may cause device-to-host transfers many times; consider vectorized approach.
  suggestion: Rework grouping to minimize Python-level loops and .item() calls on GPU arrays.

- file: utils.py
  issue: `recompose_from_percentiles` populates border values using slicing that is hard to reason about and might be off-by-one for certain shapes.
  suggestion: Add explicit unit tests covering edge cases where image dimensions are not multiples of block_size.

- file: cosmetics.py
  issue: CR_filter uses `pctile` and loop filling NaNs; consider limiting iterations to avoid infinite loops.
  suggestion: Add iteration cap and warning when fill_nan_fft fails to reduce NaNs.

- file: photo_gpu.py
  issue: Unresolved reference `streams` used when processing 2D images (lines using `with streams[0]:` and `streams[0].synchronize()`). This causes a runtime/static error in a plain Python environment.
  suggestion: Define `streams` (for example create a list of cp.cuda.Stream objects) before use, or guard this block to only execute when streams are initialized. Add a unit test that exercises the 2D path.

- file: photo_gpu.py
  issue: Several long lines and typing constructs using Python 3.10+ union operator (`|`) appear throughout the file; these are type-hints warnings depending on static checker and codebase python-version settings.
  suggestion: Normalize type hints to use typing.Union for broader compatibility or update project pyproject/CI to accept 3.10+ syntax.

- file: photo_gpu.py
  issue: Accidental functional substitutions detected during documentation edits. In particular, changes between `convolve` and `convolve_fft` were observed or reported; swapping these implementations may change edge behavior (padding, normalization, performance) and should NOT be done as part of a documentation pass.
  examples_to_check:
    - `detect_gpu` (expected: `g = convolve(img - sky, gf, origin=(0, 0))`) — ensure it remains `convolve` if a small-kernel direct convolution was intended.
    - `get_sky`, `get_mean_std`, `batch_aperture_photometry` and other functions use either `convolve` or `convolve_fft` depending on expected semantics; audit these call sites for intentional use.
  suggestion: Revert any accidental substitutions to the originally intended function. If `convolve_fft` is required for correctness/performance in a specific call-site, document the reason and add a unit/integration test covering the numerical equivalence or expected behavior change.

If you want, I can open separate pull requests implementing these fixes after you approve them.
