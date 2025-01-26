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

project = 'GPU Phot'
copyright = '2025, Samuel Lemes-Perera'
author = 'Samuel Lemes-Perera'
release = '0.1'

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
exclude_patterns = []

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
