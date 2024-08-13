#!/bin/bash

# Detectar el sistema operativo
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
fi

# Instalar dependencias según el sistema
case $OS in
    "Ubuntu")
        sudo apt-get update
        sudo apt-get install -y $(cat system_requirements/ubuntu.txt)
        ;;
    "Fedora")
        sudo dnf install -y $(cat system_requirements/fedora.txt)
        ;;
    *)
        echo "Sistema operativo no soportado"
        exit 1
        ;;
esac

# Instalar el paquete Python
pip install -r requirements.txt
pip install .