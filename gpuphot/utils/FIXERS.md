# FIXERS for gpuphot/utils

This file lists non-documentation issues and suggested code fixes identified
while performing a documentation-only pass. Do NOT apply these changes now if
we're in documentation-only mode; instead, review and apply them in a
separate code-fix branch.

1) File: `gpu.py`
   - Issue: Static analyzer warnings for `cp.cuda.Stream.null.synchronize()` (unresolved attribute). This appears to be a type-checker false positive in some environments; verify at runtime with the project's CuPy version.
   - Suggestion: Consider using `cp.cuda.get_current_stream().synchronize()` or explicitly acquiring a Stream object. Add unit tests for the memory management paths.

2) File: `gpu.py`
   - Issue: Several long f-strings exceed 120 columns.
   - Suggestion: Reformat long strings or extract helper functions to improve readability.

3) File: `smartgpudecoratorclass.py`
   - Issue: References to MemoryPool methods that may not exist across all CuPy versions (e.g., `largest_free_size`, `n_free_blocks`).
   - Suggestion: Guard attribute access with hasattr(...) and provide documented fallback behavior.

4) File: `catalog.py`
   - Issue: Import-time try/except for cuML does not define `cuNearestNeighbors` in the except branch, causing static analysis warnings.
   - Suggestion: Define a placeholder (e.g., `cuNearestNeighbors = None`) in the except block or move the import inside the GPU-specific function.

5) File: `headers.py`
   - Issue: `delete_header_from` may reference `idx` before assignment when the searched value is not found.
   - Suggestion: Initialize `idx = -1` and check the result before deletion; add tests for headers that lack the searched block.

6) File: `astro.py`
   - Issue: Astrometry index initialization can fail in constrained environments. The solver initialization logic can be improved to surface clearer error messages and support a user-configurable cache path.
   - Suggestion: Document expected cache paths and provide a lighter-weight 'check-only' initialization that reports missing files without attempting downloads.

7) General
   - Issue: Some modules contain long lines and mixed Spanish/English logging lines that were translated but may require additional lint formatting.
   - Suggestion: Run project linter / formatter (e.g., Black + isort + flake8) in a separate branch to avoid mixing formatting changes with documentation edits.

If you want, I can follow-up and open a separate branch to apply these fixes (small, well-scoped PRs) once you finish reviewing the documentation changes.
