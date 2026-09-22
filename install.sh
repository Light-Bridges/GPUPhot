#!/bin/bash
# ==============================================================================
# GPUPhot System Dependency Installer
#
# Author: Light-Bridges
# Date: 2024-08-02
#
# Description:
# This script automates the installation of system-level dependencies required
# by GPUPhot. It detects the host operating system (currently supporting
# Ubuntu and Fedora) and installs the necessary packages listed in the
# corresponding files within the `system_requirements/` directory.
#
# This script must be run with privileges sufficient to install system
# packages (e.g., using `sudo` or as the root user).
#
# Usage:
#   bash install.sh
#
# ==============================================================================

# --- OS DETECTION ---
# This block identifies the host operating system to determine which package
# manager and dependency list to use.

# Check for the existence of /etc/os-release, a standard file for identifying
# Linux distributions.
if [ -f /etc/os-release ]; then
    # Source the file to load its variables (e.g., NAME, ID, VERSION_ID).
    . /etc/os-release
    OS=$NAME
fi

# --- DEPENDENCY INSTALLATION ---
# This block uses a case statement to execute the appropriate installation
# command based on the detected OS.

case $OS in
    "Ubuntu")
        echo "Detected Ubuntu. Installing dependencies from system_requirements/ubuntu.txt..."
        # Update package lists to ensure we get the latest versions.
        sudo apt-get update
        # Install all packages listed in the ubuntu.txt file.
        # `$(cat ...)` substitutes the content of the file into the command.
        # The -y flag automatically answers "yes" to any confirmation prompts.
        sudo apt-get install -y $(cat system_requirements/ubuntu.txt)
        ;;

    "Fedora")
        echo "Detected Fedora. Installing dependencies from system_requirements/fedora.txt..."
        # Install all packages listed in the fedora.txt file using dnf.
        sudo dnf install -y $(cat system_requirements/fedora.txt)
        ;;

    *)
        # If the OS is not supported, print an error message and exit with a
        # non-zero status code to indicate failure.
        echo "Unsupported operating system: $OS"
        echo "Please manually install the required dependencies for your system."
        exit 1
        ;;
esac

echo "System dependencies installed successfully."

# --- PYTHON PACKAGE INSTALLATION (COMMENTED OUT) ---
# The following lines are commented out but serve as a reference for the
# next steps in a manual installation process. After running this script,
# the user would typically install the Python dependencies and then the
# GPUPhot package itself.

# pip install -r requirements.txt
# pip install .
