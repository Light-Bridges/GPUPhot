from setuptools import setup, find_packages


# Función para leer el archivo requirements.txt
def read_requirements():
    with open('requirements.txt') as reqs_file:
        return reqs_file.read().splitlines()


setup(
    name='GPUPhot',
    version='0.1.0',
    description='A Python library for GPU accelerated photometry and astrometry',
    author='Samuel Lemes Perera',
    author_email='Samuel@lightbridges.es',
    packages=find_packages(),
    install_requires=read_requirements(),
)
