# gpuphot package

This package contains the core modules of the GPU-accelerated photometry
pipeline. High-level components:

- `phot/` - Photometry pipeline and GPU-backed processing functions.
- `logger/` - Lightweight hierarchical logging utilities and decorators.
- `utils/` - Utility modules (catalog queries, headers, GPU memory helpers, astro helpers).
- `stats/` - Image stacking, subpixel registration (Guizar-Sicairos), and sigma-clipping utilities.
- `instrument_configs/` - JSON configuration files for instruments. Contains `default.json` with the base configuration; per-instrument overrides are deployed externally via `INSTRUMENT_CONFIG_BASE_PATH`.
- `image_processor.py` - Main `ImageProcessor` class: loads instrument config, delegates to `process_image` in `phot/photo_gpu.py`, and provides a `create_processor` factory.
- `instrument_config_parser.py` - Loads and merges `default.json` with per-instrument configs and provides a `HeaderTranslator`.
- `exceptions.py` - Custom exception hierarchy (`GPUPhotError`, `InsufficientStarsError`, `MoffatFitError`, etc.) used throughout the pipeline.
- `patch_cupy.py` - Optional CuPy patch to default to float64 precision (not imported by default).

For usage examples, consult the top-level `USAGE.md` and `notebooks/`.
