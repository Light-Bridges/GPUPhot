from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(name='gpuphot', version='0.1', packages=find_packages(), install_requires=requirements,
      description='A GPU utility library', author='Samuel Lemes Perera', author_email='SamuelLemesPerera@gmail.com',
      long_description='\nEste paquete requiere dependencias del sistema adicionales.\nPor favor, consulta el README.md para instrucciones de instalación completas.\n')
