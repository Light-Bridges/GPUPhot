import os
import re

from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from gpuphot/__init__.py (single source of truth)
with open(os.path.join(this_directory, 'gpuphot', '__init__.py'), encoding='utf-8') as f:
    version = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.M).group(1)

# Core dependencies: everything EXCEPT CuPy and RAPIDS.
# CuPy must be installed separately by the user because the correct package
# depends on the CUDA version (cupy-cuda11x, cupy-cuda12x, etc.).
install_requires = [
    'astrometry>=3.0',
    'astropy>=5.0',
    'astroquery>=0.4.6',
    'ephem>=4.1',
    'json-with-comments>=1.2',
    'lmfit>=1.2',
    'matplotlib>=3.5',
    'multiprocess>=0.70',
    'nvtx>=0.2.10',
    'numpy>=1.23',
    'pandas>=1.5',
    'python-dotenv>=1.0',
    'python-logstash-async>=2.5',
    'scikit-learn>=1.1',
    'scipy>=1.9',
]

# Optional dependency groups:
# - dev: installed via `pip install gpuphot[dev]` for testing and docs.
# - worker: installed via `pip install -e .[worker]` for running the distributed
#   Celery/Redis/PostgreSQL orchestration layer locally.
#
# Architectural separation:
# The core library 'gpuphot' is strictly decoupled from the distributed
# infrastructure layer. 'gpuphot_worker' provides Celery tasks, database persistence,
# and cluster orchestration, running primarily via Docker containers with
# requirements-worker.txt. 'gpuphot_worker' is excluded from the core PyPI wheel
# so that `pip install gpuphot` installs strictly the scientific library without
# undeclared backend dependencies.
extras_require = {
    'dev': [
        'pytest',
        'sphinx',
        'sphinx_rtd_theme',
        'sphinxcontrib-bibtex',
        'sphinx-tippy',
        'nbsphinx',
    ],
    'worker': [
        'celery>=5.2',
        'redis>=4.5',
        'flower>=2.0',
        'pytz',
        'scikit-image>=0.19',
        'SQLAlchemy>=2.0',
        'psycopg2-binary>=2.9',
        'celery-redbeat>=2.0',
    ],
}

setup(
    name='gpuphot',
    version=version,
    packages=find_packages(exclude=['tests', 'tests.*', 'dev', 'dev.*',
                                    'benchmarks', 'benchmarks.*',
                                    'profiling_scripts', 'profiling_scripts.*',
                                    'gpuphot_worker', 'gpuphot_worker.*']),
    install_requires=install_requires,
    extras_require=extras_require,
    description='A GPU-accelerated library for astronomical photometry and astrometry',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Samuel Lemes-Perera, Miguel R. Alarcon',
    author_email='samuel@lightbridges.es',
    url='https://github.com/Light-Bridges/GPUPhot',
    project_urls={
        'Documentation': 'https://gpuphot.readthedocs.io',
        'Bug Tracker': 'https://github.com/Light-Bridges/GPUPhot/issues',
    },
    license='MIT',
    package_data={
        'gpuphot.instrument_configs': ['*.json', '*.md'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
