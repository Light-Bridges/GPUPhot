import argparse
import os
from pathlib import Path

import nbformat as nbf

# Crear un nuevo notebook
nb_setup = nbf.v4.new_notebook()

# Título y descripción del notebook
nb_setup['cells'].append(nbf.v4.new_markdown_cell(
    "# Initial Setup\n"
    "This notebook guides you through the initial setup steps required to use the GPUPhot system.\n\n"
    "## Step 1: Verify Environment Variables\n"
    "Before proceeding, ensure that the required environment variables are correctly configured. "
    "Run the following code to check the current values of the static environment variables:"
))

# Celda para verificar las variables de entorno estáticas
nb_setup['cells'].append(nbf.v4.new_code_cell(
    "import os\n\n"
    "# Default values for environment variables\n"
    "default_values = {\n"
    "    'ASTROMETRY_CACHE_PATH': './astrometry_cache',\n"
    "    'INSTRUMENT_CONFIG_PATH': './gpuphot/instrument_configs',\n"
    "    'IMAGE_PATH': '~/gpuphot_images',\n"
    "}\n\n"
    "# Check static environment variables\n"
    "print('Current environment variable values (or defaults if not defined):\\n')\n"
    "for var, default in default_values.items():\n"
    "    value = os.getenv(var, default)\n"
    "    print(f'{var}: {value}')\n\n"
    "# Check if any variable is using the default value\n"
    "using_defaults = [var for var, default in default_values.items() if os.getenv(var) is None]\n"
    "if using_defaults:\n"
    "    print(f'\\nNote: The following variables are using default values: {using_defaults}')\n"
    "else:\n"
    "    print('\\nAll required environment variables are defined in the .env file.')"
))

# Sección 2: Descargar índices de astrometría
nb_setup['cells'].append(nbf.v4.new_markdown_cell(
    "## Step 2: Download Astrometry Index Files\n"
    "Before processing astronomical images, you need to download the astrometry index files. "
    "These files are required for the astrometric calibration of the images.\n\n"
    "**Important:** This process uses the `ASTROMETRY_CACHE_PATH` variable from your `.env` file to mount a directory from your host machine. "
    "The code below will confirm the exact host path being used and then begin the download.\n\n"
    "Run the following code to download the index files:"
))

download_cell_code = """import os
from gpuphot.utils.astro import get_solver

def download_astrometry_files():
    \"\"\"
    Checks for the astrometry cache path configuration and triggers the download
    of the index files, providing clear information to the user about the host and
    container paths.
    \"\"\"
    # This is the path INSIDE the Docker container, where the application runs.
    internal_cache_path = '/data/astrometry_cache'

    # This environment variable is passed from the docker-compose.yml
    # to make the HOST path visible inside the container for informational purposes.
    host_cache_path = os.getenv('ASTROMETRY_CACHE_PATH_HOST')

    if host_cache_path:
        print("--- Astrometry Index File Setup ---")
        print(f"The solver will download files to the directory mounted inside this container.")
        print(f"  - Container Path: {internal_cache_path}")
        print(f"  - Host Machine Path: {host_cache_path}\\n")

        print("IMPORTANT:")
        print("The download process requires approximately 34GB of disk space and can take")
        print("over an hour, depending on your internet connection. This only needs to be done once.\\n")

        try:
            # The get_solver() function is already configured to find the
            # cache at the internal path, which is '/data/astrometry_cache'.
            get_solver()
            print("--- Success ---")
            print(f"Astrometry index files are now available and stored on your host machine at: {host_cache_path}")
        except Exception as e:
            print(f"--- Error ---")
            print(f"An error occurred during the solver initialization: {e}")
            print("Please check your internet connection, disk space, and file permissions on the host path.")

    else:
        print("--- Configuration Error ---")
        print("Error: The 'ASTROMETRY_CACHE_PATH_HOST' environment variable is not set inside the container.")
        print("Please ensure it is correctly defined in your docker-compose.yml for the 'lab' service.")
        print("Example for docker-compose.yml:")
        print("  services:")
        print("    lab:")
        print("      environment:")
        print("        - ASTROMETRY_CACHE_PATH_HOST=${ASTROMETRY_CACHE_PATH:-./astrometry_cache}")


# To run the process, simply call the function in your notebook cell.
download_astrometry_files()
"""

# Celda para descargar los índices de astrometría
# Se usa la variable `download_cell_code` definida al inicio
nb_setup['cells'].append(nbf.v4.new_code_cell(download_cell_code))

# Sección 3: Configuración de variables de entorno
nb_setup['cells'].append(nbf.v4.new_markdown_cell(
    "## Step 3: Configure Environment Variables\n"
    "The GPUPhot system relies on several environment variables to function correctly. "
    "These variables control settings such as database connections, file paths, and logging.\n\n"
    "### Example `.env` File\n"
    "Below is an example of a `.env` file with all the available environment variables. "
    "You can copy this template and modify it according to your setup.\n\n"
    "```env\n"
    "# Library settings\n"
    "GPUPHOT_DEBUG=True  # Enable debug mode for detailed logging\n"
    "GPUPHOT_LOG_LEVEL=DEBUG  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)\n"
    "GPUPHOT_ENVIRONMENT=development  # Environment (development, production)\n\n"
    "# Astrometry settings\n"
    "ASTROMETRY_CACHE_PATH=/path/to/astrometry_cache  # Path on the host to store astrometry index files\n\n"
    "# Worker settings\n"
    "INSTRUMENT_NAME=default  # Default instrument configuration\n"
    "INSTRUMENT_CONFIG_PATH=/path/to/instrument_configs  # Path on the host to instrument configuration files\n"
    "IMAGE_PATH=/path/to/images  # Base path on the host for astronomical images\n\n"
    "# Logging settings\n"
    "LOGSTASH_LOGGING=True  # Enable Logstash logging (optional)\n"
    "LOGSTASH_HOST=10.10.10.10  # Logstash server IP or hostname\n"
    "LOGSTASH_PORT=5000  # Logstash server port\n\n"
    "# Database settings\n"
    "POSTGRES_PASSWORD=gpuphot  # Password for the PostgreSQL database\n"
    "```\n\n"
    "### Instructions\n"
    "1. Create a file named `.env` in the root directory of your project.\n"
    "2. Copy the example above into the `.env` file.\n"
    "3. Replace the placeholder paths and values with your actual configuration.\n\n"
    "**Note:** The `.env` file is optional. If you don't provide one, the system will use default values for most settings.\n\n"
    "**Important:** The paths specified in `ASTROMETRY_CACHE_PATH`, `INSTRUMENT_CONFIG_PATH`, and `IMAGE_PATH` must be paths on the **host machine**, not inside the Docker container. These paths will be mounted into the container at runtime."
))

# Sección final: Link to the next notebook
nb_setup['cells'].append(nbf.v4.new_markdown_cell(
    "## Next Steps\n"
    "Once the setup is complete, you can proceed to the following notebooks:\n"
    "- [2. Instrument Configuration](./2_Instrument_Configuration_Notebook.ipynb): Learn how to configure instruments and explore configuration files.\n"
    "- [3. Task Execution](./3_Task_Execution_Notebook.ipynb): Learn how to process images using Celery tasks.\n"
    "- [4. Database Query](./4_Database_Query_Notebook.ipynb): Explore the database and query photometric data.\n"
))

# Guardar el notebook de configuración

parser = argparse.ArgumentParser(description="Generate a Jupyter Notebook.")

# Añadir un argumento opcional '--output-dir'
# Si no se proporciona, se usará el valor 'default'.
parser.add_argument(
    '--output-dir',
    type=str,
    default=os.path.join(Path(__file__).resolve().parent, '..', '..', 'notebooks'),
    help='The directory where the notebook will be saved.'
)

args = parser.parse_args()
output_dir = args.output_dir

output_path_setup = os.path.join(output_dir, "1_Setup_Notebook.ipynb")
os.makedirs(output_dir, exist_ok=True)

with open(output_path_setup, 'w', encoding='utf-8') as f:
    nbf.write(nb_setup, f)

print(f"Setup notebook created successfully: {output_path_setup}")
