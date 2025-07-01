import argparse
import os
from pathlib import Path

import nbformat as nbf

# Crear un nuevo notebook
nb_system = nbf.v4.new_notebook()

# Título y descripción del notebook
nb_system['cells'].append(nbf.v4.new_markdown_cell(
    "# System Overview\n"
    "This notebook provides an overview of the system architecture and services available in the GPUPhot Docker Compose setup.\n\n"
    "## Services Overview\n"
    "The system is composed of the following services:\n\n"
    "1. **RabbitMQ**: Message broker for Celery task queue.\n"
    "   - Ports: `5672` (AMQP) and `15672` (Management UI).\n"
    "2. **Redis**: Backend for Celery task results.\n"
    "   - Port: `6379`.\n"
    "3. **Celery Beat**: Scheduler for periodic tasks.\n"
    "4. **GPUPhot Worker**: Celery worker for processing astronomical images.\n"
    "   - Supports GPU acceleration.\n"
    "5. **JupyterLab**: Interactive environment for running notebooks.\n"
    "   - Port: `8888`.\n"
    "6. **Flower**: Monitoring tool for Celery tasks.\n"
    "   - Port: `5555`.\n"
    "7. **PostgreSQL**: Database for storing processing results.\n"
    "   - Port: `5432`.\n\n"
))

# Sección final: Link to the setup notebook
nb_system['cells'].append(nbf.v4.new_markdown_cell(
    "## Next Steps\n"
    "Before using the system, ensure you have completed the initial setup:\n"
    "- [1. Initial Setup](./1_Setup_Notebook.ipynb): Download astrometry index files and configure the system.\n\n"
    "Once the setup is complete, proceed to the following notebooks:\n"
    "- [2. Instrument Configuration](./2_Instrument_Configuration_Notebook.ipynb): Learn how to configure instruments and explore configuration files.\n"
    "- [3. Task Execution](./3_Task_Execution_Notebook.ipynb): Learn how to process images using Celery tasks.\n"
))
# Guardar el notebook de overview


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

output_path_system = os.path.join(output_dir, "0_System_Overview_Notebook.ipynb")
os.makedirs(output_dir, exist_ok=True)

with open(output_path_system, 'w', encoding='utf-8') as f:
    nbf.write(nb_system, f)

print(f"System overview notebook created successfully: {output_path_system}")
