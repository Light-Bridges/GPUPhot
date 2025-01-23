import os

from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gpuphot',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    description='A GPU-accelerated library for astronomical photometry and astrometry',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Samuel Lemes Perera',
    author_email='SamuelLemesPerera@gmail.com',
    url='https://github.com/Light-Bridges/GPUPhot',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    extras_require={
        'dev': ['pytest', 'sphinx', 'sphinx_rtd_theme'],
    },
)
