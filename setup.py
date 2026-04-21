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

# Optional dependency groups installed via: pip install gpuphot[dev]
# Note: distributed worker dependencies (Celery, Redis, etc.) are installed
# inside Docker containers via requirements-worker.txt, not as pip extras.
extras_require = {
    'dev': [
        'pytest',
        'sphinx',
        'sphinx_rtd_theme',
        'sphinxcontrib-bibtex',
        'sphinx-tippy',
        'nbsphinx',
    ],
}

setup(
    name='gpuphot',
    version=version,
    packages=find_packages(exclude=['tests', 'tests.*', 'dev', 'dev.*',
                                    'benchmarks', 'benchmarks.*',
                                    'profiling_scripts', 'profiling_scripts.*']),
    install_requires=install_requires,
    extras_require=extras_require,
    description='A GPU-accelerated library for astronomical photometry and astrometry',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Samuel Lemes Perera',
    author_email='SamuelLemesPerera@gmail.com',
    url='https://github.com/Light-Bridges/GPUPhot',
    project_urls={
        'Documentation': 'https://gpuphot.readthedocs.io',
        'Bug Tracker': 'https://github.com/Light-Bridges/GPUPhot/issues',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
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
