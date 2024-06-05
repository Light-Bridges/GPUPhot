from setuptools import setup, find_packages

# Leer las dependencias desde requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    install_requires=requirements,
)
