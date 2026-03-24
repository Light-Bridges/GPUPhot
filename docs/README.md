# GPUPhot Documentation

API documentation generated with [Sphinx](https://www.sphinx-doc.org/) using
the Read the Docs theme.

## Prerequisites

Install the documentation dependencies (from the project root):

```bash
pip install sphinx sphinx-rtd-theme sphinxcontrib-bibtex sphinx-tippy nbsphinx
```

## Build

From this directory (`docs/`):

```bash
# HTML (recommended)
make html

# Other formats
make latexpdf   # PDF via LaTeX
make help       # list all available targets
```

On Windows (without Make):

```cmd
make.bat html
```

The output is written to `build/html/`. Open `build/html/index.html` in a
browser to view it.

## Rebuild from scratch

```bash
make clean
make html
```

## Structure

```
docs/
  Makefile          # GNU Make driver for Sphinx
  make.bat          # Windows equivalent
  source/
    conf.py         # Sphinx configuration (extensions, theme, paths)
    index.rst       # Documentation entry point
    *.rst           # Module reference pages (auto-generated from docstrings)
    references.bib  # BibTeX references
    _static/        # Custom CSS / images
    _templates/     # Custom Jinja templates
  build/            # Generated output (not committed)
```

## Configuration highlights

- **autodoc + napoleon**: Docstrings are extracted automatically; both
  Google-style and NumPy-style are supported.
- **nbsphinx**: Jupyter notebooks from `notebooks/` can be included;
  they are rendered but never executed during build (`nbsphinx_execute = 'never'`).
- **viewcode**: Adds links from the documentation back to highlighted source
  code.
- **bibtex**: Citations from `references.bib` can be used in `.rst` files.
