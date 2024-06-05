#!/bin/bash

# Instalar dependencias del sistema operativo
sudo apt-get update
sudo apt-get install -y python3-dev python3-numpy-dev python3-setuptools cython3 python3-pytest-astropy python3-scipy python3-sklearn python3-sklearn-lib python3-matplotlib

# Instalar la librería Python (opcional si deseas automatizar la instalación)
# pip install .
