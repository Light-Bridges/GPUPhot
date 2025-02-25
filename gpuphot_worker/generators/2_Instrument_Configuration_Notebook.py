import os
from pathlib import Path

import nbformat as nbf

# Crear un nuevo notebook
nb_config = nbf.v4.new_notebook()

# Título y descripción del notebook
nb_config['cells'].append(nbf.v4.new_markdown_cell(
    "# Instrument Configuration\n"
    "This notebook demonstrates how to explore, create, and customize instrument configuration files."
))

# Sección 1: Displaying the Default Configuration
nb_config['cells'].append(nbf.v4.new_markdown_cell(
    "## Displaying the Default Configuration\n"
    "The following code displays the default configuration included in the library by directly reading the `default.json` file, including comments."
))

nb_config['cells'].append(nbf.v4.new_code_cell(
    "import os\n\n"
    "# Define the path to the default.json file\n"
    "default_json_path = os.path.join(os.getenv('PYTHONPATH'), 'gpuphot', 'instrument_configs', 'default.json')\n\n"
    "# Load and display the default configuration as plain text\n"
    "try:\n"
    "    with open(default_json_path, 'r') as file:\n"
    "        default_config_content = file.read()\n"
    "    print('Default configuration from default.json (with comments):')\n"
    "    print(default_config_content)\n"
    "except FileNotFoundError:\n"
    "    print(f'Error: The file {default_json_path} does not exist.')\n"
    "except Exception as e:\n"
    "    print(f'Error reading default.json: {e}')"
))

# Sección 2: Creating a Default Configuration File
nb_config['cells'].append(nbf.v4.new_markdown_cell(
    "## Creating a Default Configuration File\n"
    "The following code demonstrates how to generate a default configuration file.\n\n"
    "**Note:** The default configuration is always available, even if no `default.json` file exists. "
    "It is generated internally using `DefaultConfig`. However, you can generate a `default.json` file "
    "to customize the default settings."
))

nb_config['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot.instrument_config_parser import InstrumentConfigParser\n"
    "import os\n\n"
    "# Define the configuration directory\n"
    "INSTRUMENT_CONFIG_BASE_PATH = os.getenv('INSTRUMENT_CONFIG_BASE_PATH', '/data/instrument_configs')\n"
    "print(f'Configuration directory: {INSTRUMENT_CONFIG_BASE_PATH}')\n\n"
    "# Create a default configuration file (optional)\n"
    "config_parser = InstrumentConfigParser(config_dir=INSTRUMENT_CONFIG_BASE_PATH)\n"
    "config_parser.generate_default_config()\n"
    "print('Default configuration file generated.')"
))

# Sección 3: Creating a Custom Configuration File
nb_config['cells'].append(nbf.v4.new_markdown_cell(
    "## Creating a Custom Configuration File\n"
    "The following code demonstrates how to generate a custom configuration file and open it for editing."
))

nb_config['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot.instrument_config_parser import InstrumentConfigParser\n\n"
    "# Create a custom configuration file\n"
    "instrument_name = 'my_instrument'\n"
    "config_parser = InstrumentConfigParser(config_dir=INSTRUMENT_CONFIG_BASE_PATH)\n"
    "config_parser.generate_config_file(instrument_name)\n"
    "print(f'Configuration file for {instrument_name} generated.')\n\n"
    "# Open the file for editing\n"
    "file_path = os.path.join(config_parser.config_dir, f'{instrument_name}.json')\n"
    "with open(file_path, 'r') as file:\n"
    "    content = file.read()\n"
    "print(f'Content of {instrument_name}.json:')\n"
    "print(content)"
))

# Sección 4: Exploring Configuration Files
nb_config['cells'].append(nbf.v4.new_markdown_cell(
    "## Exploring Configuration Files\n"
    "The following code explores the configuration files in `INSTRUMENT_CONFIG_BASE_PATH`."
))

nb_config['cells'].append(nbf.v4.new_code_cell(
    "import os\n"
    "import json\n\n"
    "# Define the configuration directory\n"
    "INSTRUMENT_CONFIG_BASE_PATH = os.getenv('INSTRUMENT_CONFIG_BASE_PATH', '/data/instrument_configs')\n"
    "print(f'Configuration directory: {INSTRUMENT_CONFIG_BASE_PATH}')\n\n"
    "# Explore JSON configuration files\n"
    "for filename in os.listdir(INSTRUMENT_CONFIG_BASE_PATH):\n"
    "    file_path = os.path.join(INSTRUMENT_CONFIG_BASE_PATH, filename)\n"
    "    if os.path.isfile(file_path) and filename.endswith('.json'):\n"
    "        try:\n"
    "            with open(file_path, 'r') as file:\n"
    "                data = json.load(file)\n"
    "                print(f'Content of {filename}:')\n"
    "                print(json.dumps(data, indent=4))\n"
    "                print('=' * 50 + '\\n')\n"
    "        except json.JSONDecodeError:\n"
    "            print(f'Error: {filename} is not a valid JSON file.')\n"
    "        except Exception as e:\n"
    "            print(f'Error reading {filename}: {e}')"
))

# Sección final: Link to the next notebook
nb_config['cells'].append(nbf.v4.new_markdown_cell(
    "## Next Steps\n"
    "Now that you have explored and customized instrument configurations, proceed to the next notebook to learn how to process images:\n"
    "- [3. Task Execution](./3_Task_Execution_Notebook.ipynb)"
))

# Guardar el notebook de configuración
output_dir = os.path.join(Path(__file__).parent.absolute(), '..', '..', 'notebooks')
output_path_config = os.path.join(output_dir, "2_Instrument_Configuration_Notebook.ipynb")
os.makedirs(output_dir, exist_ok=True)

with open(output_path_config, 'w', encoding='utf-8') as f:
    nbf.write(nb_config, f)

print(f"Configuration notebook created successfully: {output_path_config}")
