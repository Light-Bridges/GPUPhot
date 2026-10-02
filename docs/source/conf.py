import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../gpuphot'))
sys.path.insert(0, os.path.abspath('../../gpuphot_worker'))
sys.path.insert(0, os.path.abspath('../../notebooks'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPUPhot'
copyright = '2025-2026, Samuel Lemes-Perera'
author = 'Samuel Lemes-Perera'

import os
import re
import shutil

# Ensure Logstash and network logging are disabled during doc builds
os.environ['LOGSTASH_LOGGING'] = 'False'

# Ensure pandoc is in PATH (checks system or pypandoc-binary)
if not shutil.which('pandoc'):
    try:
        import pypandoc
        _pandoc_dir = os.path.dirname(pypandoc.get_pandoc_path())
        if _pandoc_dir and _pandoc_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _pandoc_dir + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

_init_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gpuphot', '__init__.py'))
with open(_init_py, encoding='utf-8') as _f:
    release = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", _f.read(), re.M).group(1)
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinx.ext.extlinks',
    'sphinx_tippy',
    'nbsphinx',
]
extlinks = {
    'doi': ('https://dx.doi.org/%s', 'doi:%s'),
}
bibtex_bibfiles = ['references.bib']
# nbsphinx configuration
nbsphinx_allow_errors = True  # Continue building even if there are errors in notebooks
nbsphinx_execute = 'never'    # Do not execute notebooks during build

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Mock heavy/GPU dependencies so Sphinx autodoc builds on ReadTheDocs without physical GPUs
autodoc_mock_imports = [
    'cupy',
    'cupyx',
    'cupy_backends',
    'cuml',
    'pylibraft',
    'nvtx',
    'astrometry',
    'celery',
    'redis',
    'psycopg2',
    'skimage',
    'logstash_async',
    'cupynumeric',
    'sqlalchemy',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

from docutils import nodes
from docutils.parsers.rst import roles

def setting_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    url = f"https://docs.celeryproject.org/en/stable/userguide/configuration.html#{text}"
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []

def sig_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    url = f"https://docs.celeryproject.org/en/stable/userguide/signals.html#{text}"
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []

def setup(app):
    roles.register_local_role('setting', setting_role)
    roles.register_local_role('sig', sig_role)
