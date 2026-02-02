# gpuphot package

This package contains the core modules of the GPU-accelerated photometry
pipeline. High-level components:

- `phot/` - Photometry pipeline and GPU-backed processing functions.
- `logger/` - Lightweight hierarchical logging utilities and decorators.
- `utils/` - Utility modules (catalog queries, headers, GPU memory helpers, astro helpers).
- `instrument_configs/` - JSON configuration files for instruments (default.json + per-instrument files).
- `image_processor.py` - A small wrapper to instantiate processing flows for a given instrument.
- `instrument_config_parser.py` - Loads and merges `default.json` with per-instrument configs and provides a HeaderTranslator.

This README focuses on what you will find in the package rather than detailed usage
instructions. For usage examples, consult the top-level `USAGE.md` and notebooks.
