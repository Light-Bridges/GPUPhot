from setuptools import setup, find_packages

# Leer el contenido de requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gpuphot',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,  # Usar las dependencias de requirements.txt
    description='A GPU utility library',
    author='Samuel Lemes Perera',
    author_email='SamuelLemesPerera@gmail.com',
    long_description='''
Este paquete requiere dependencias del sistema adicionales.
Por favor, consulta el README.md para instrucciones de instalación completas.
''',
    # Otros parámetros de configuración...
)
