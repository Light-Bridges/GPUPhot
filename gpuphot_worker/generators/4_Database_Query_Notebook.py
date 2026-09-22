# SPDX-License-Identifier: MIT
"""
Notebook generator: Database Query

Creates a Jupyter notebook that demonstrates how to query the project's
PostgreSQL database for photometric results, light curves and image
statistics. Intended to be executed by the workspace initializer.
"""

import argparse
import os
from pathlib import Path

import nbformat as nbf

# Create a new notebook
nb_db = nbf.v4.new_notebook()

# Title and description of the notebook
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "# Database Query\n"
    "This notebook demonstrates how to query the PostgreSQL database to extract photometric data, light curves, and image statistics.\n\n"
    "## Services Overview\n"
    "The database contains two main tables:\n\n"
    "1. **`imastats`**: Stores image metadata and statistics.\n"
    "   - Columns: `id`, `file_path`, `naxis1`, `naxis2`, `telescop`, `instrume`, `camera`, `filter`, `date_obs`, `exptime`, `object`, `ra`, `dec`, `fwhm`, `maglim`, `header`.\n"
    "2. **`imaphot`**: Stores photometric data for detected objects.\n"
    "   - Columns: `id`, `ra`, `dec`, `flux`, `dflux`, `trans`.\n\n"
    "## Next Steps\n"
    "Proceed to the following sections to explore the database:\n"
    "- [1. Query by Coordinates](#1.-Query-by-Coordinates)\n"
    "- [2. Query by Date Range](#2.-Query-by-Date-Range)\n"
    "- [3. Query by Filename](#3.-Query-by-Filename)\n"
    "- [4. Query Transients](#4.-Query-Transients)\n"
    "- [5. Generate Light Curves](#5.-Generate-Light-Curves)\n"
    "- [6. Explore Image Statistics](#6.-Explore-Image-Statistics)\n"
    "- [7. Connect to Database](#7.-Connect-to-Database)\n"
    "- [8. Query by Multiple Filenames](#8.-Query-by-Multiple-Filenames)\n"
    "- [9. Search Transient Images](#9.-Search-Transient-Images)\n"
))

nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## Database Schema\n"
    "The following code displays the columns available in each table of the database."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.database_search_utils import get_tables_and_columns\n\n"
    "# Get tables and columns from the database\n"
    "tables_and_columns = get_tables_and_columns()\n\n"
    "# Print columns for each table\n"
    "for table, columns in tables_and_columns.items():\n"
    "    print(f'\\nTable: {table}')\n"
    "    print('Columns:')\n"
    "    for column_name, data_type in columns:\n"
    "        print(f'  - {column_name}: {data_type}')"
))

nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## Available Functions\n"
    "The following code lists all the functions available in the `gpuphot_worker.database_search_utils` module."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker import database_search_utils\n"
    "import inspect\n\n"
    "# Get all functions from the database_search_utils module\n"
    "functions = inspect.getmembers(database_search_utils, inspect.isfunction)\n\n"
    "# Print available functions\n"
    "print('Available functions in gpuphot_worker.database_search_utils:')\n"
    "for name, _ in functions:\n"
    "    print(f'  - {name}')"
))

# Section 1: Query by Coordinates
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 1. Query by Coordinates\n"
    "Search for objects or images within a specific region of the sky using Right Ascension (RA) and Declination (Dec)."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "from IPython.display import display\n"
    "from gpuphot_worker.database_search_utils import search_by_radec\n\n"
    "# Example: Search for objects near RA=14.123, Dec=31.345 with a radius of 1 degree\n"
    "ra, dec, radius = 14.123, 31.345, 1.0\n"
    "df_objects = search_by_radec(ra, dec, radius, table='imaphot')\n"
    "df_images = search_by_radec(ra, dec, radius, table='imastats')\n\n"
    "print('Objects found:')\n"
    "display(df_objects.head())\n\n"
    "print('Images found:')\n"
    "display(df_images.head())"
))

# Section 2: Query by Date Range
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 2. Query by Date Range\n"
    "Search for images or objects observed within a specific date range."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.database_search_utils import search_by_date_range\n\n"
    "# Example: Search for images observed between 2024-01-01 and 2024-12-31\n"
    "start_date, end_date = '2024-01-01', '2024-12-31'\n"
    "df_date_results = search_by_date_range(start_date, end_date)\n\n"
    "print('Images found:')\n"
    "df_date_results.head()"
))

# Section 3: Query by Filename
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 3. Query by Filename\n"
    "Search for images or objects by filename or partial filename."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.database_search_utils import search_by_filename\n\n"
    "# Example: Search for images containing 'Mrk352' in the filename\n"
    "filename_part = 'Mrk352'\n"
    "df_file_results = search_by_filename(filename_part)\n\n"
    "print('Images found:')\n"
    "df_file_results.head()"
))

# Section 4: Query Transients
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 4. Query Transients\n"
    "Search for transient objects observed after a specific date."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.database_search_utils import search_transients\n\n"
    "# Example: Search for transients observed after 2024-01-01\n"
    "date_after = '2024-01-01'\n"
    "df_transients = search_transients(date_after)\n\n"
    "print('Transients found:')\n"
    "df_transients.head()"
))

# Section 5: Generate Light Curves
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 5. Generate Light Curves\n"
    "Generate light curves for objects detected in multiple images."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "import pandas as pd\n\n"
    "# Example: Generate a light curve for a specific object\n"
    "object_id = df_objects['id'].iloc[0]  # Use the first object found in the previous query\n"
    "df_light_curve = df_date_results[df_date_results['id'] == object_id]\n\n"
    "# Convert 'date_obs' to datetime if necessary\n"
    "df_light_curve['date_obs'] = pd.to_datetime(df_light_curve['date_obs'])\n\n"
    "# Round dates to seconds (optional: change 'S' to 'T' for minutes)\n"
    "df_light_curve['date_obs'] = df_light_curve['date_obs'].dt.round('S')\n\n"
    "# Group by date and compute mean flux and dflux\n"
    "df_grouped = df_light_curve.groupby('date_obs').agg({'flux': 'mean', 'dflux': 'mean'}).reset_index()\n\n"
    "# Plot the light curve\n"
    "plt.figure(figsize=(10, 5))\n"
    "plt.errorbar(df_grouped['date_obs'], df_grouped['flux'], yerr=df_grouped['dflux'], fmt='o')\n"
    "plt.title(f'Light Curve for Object {object_id}')\n"
    "plt.xlabel('Date')\n"
    "plt.ylabel('Flux')\n"
    "plt.grid(True)\n\n"
    "# Adjust X axis to show only the data range\n"
    "plt.xlim(df_grouped['date_obs'].min(), df_grouped['date_obs'].max())\n"
    "plt.show()"
))

# Section 6: Explore Image Statistics
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 6. Explore Image Statistics\n"
    "Explore statistics for images, such as FWHM, magnitude limit, and exposure time."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "# Example: Plot FWHM vs. Magnitude Limit\n"
    "plt.figure(figsize=(10, 5))\n"
    "plt.scatter(df_images['fwhm'], df_images['maglim'])\n"
    "plt.title('FWHM vs. Magnitude Limit')\n"
    "plt.xlabel('FWHM (arcseconds)')\n"
    "plt.ylabel('Magnitude Limit')\n"
    "plt.grid(True)\n"
    "plt.show()"
))

# Section 7: Connect to Database
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 7. Connect to Database\n"
    "Establish a connection to the PostgreSQL database."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.database_search_utils import connect_to_db\n\n"
    "# Example: Connect to the database\n"
    "conn = connect_to_db()\n"
    "print('Connection established successfully!')"
))

# Section 8: Query by Multiple Filenames
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 8. Query by Multiple Filenames\n"
    "Search for images or objects by multiple filenames."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.database_search_utils import search_by_filenames\n\n"
    "# Example: Search for images with filenames 'image1.fits' and 'image2.fits'\n"
    "filenames = ['image1.fits', 'image2.fits']\n"
    "df_multiple_files = search_by_filenames(filenames)\n\n"
    "print('Images found:')\n"
    "df_multiple_files.head()"
))

# Section 9: Search Transient Images
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## 9. Search Transient Images\n"
    "Search for images containing transient objects."
))

nb_db['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.database_search_utils import search_transient_images\n"
    "from datetime import datetime, timedelta\n\n"
    "# Get current date and subtract a week\n"
    "date_after = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')\n\n"
    "# Example: Search for images containing transients observed in the last week\n"
    "df_transient_images = search_transient_images(date_after)\n\n"
    "print('Transient images found:')\n"
    "df_transient_images.head()"
))

# Final section: Link to the next notebook
nb_db['cells'].append(nbf.v4.new_markdown_cell(
    "## Next Steps\n"
    "If you need to revisit or modify instrument configurations, go back to the previous notebook:\n"
    "- [2. Instrument Configuration](./2_Instrument_Configuration_Notebook.ipynb)\n\n"
    "To learn how to process images, proceed to the next notebook:\n"
    "- [3. Task Execution](./3_Task_Execution_Notebook.ipynb)"
))

# Save the notebook
parser = argparse.ArgumentParser(description="Generate a Jupyter Notebook.")
parser.add_argument(
    '--output-dir',
    type=str,
    default=os.path.join(Path(__file__).resolve().parent, '..', '..', 'notebooks'),
    help='The directory where the notebook will be saved.'
)

args = parser.parse_args()
output_dir = args.output_dir

output_path_db = os.path.join(output_dir, "4_Database_Query_Notebook.ipynb")
os.makedirs(output_dir, exist_ok=True)

with open(output_path_db, 'w', encoding='utf-8') as f:
    nbf.write(nb_db, f)

print(f"Database query notebook created successfully: {output_path_db}")