import os
from pathlib import Path

import nbformat as nbf

# Crear un nuevo notebook
nb_tasks = nbf.v4.new_notebook()

# Título y descripción del notebook
nb_tasks['cells'].append(nbf.v4.new_markdown_cell(
    "# Task Execution\n"
    "This notebook demonstrates how to use Celery tasks to process astronomical images.\n\n"
    "**Note:** This system only processes FITS (.fits) and NumPy (.npy) files. "
    "Other file formats are not supported."
))

# Sección 1: Processing a Single Image
nb_tasks['cells'].append(nbf.v4.new_markdown_cell(
    "## Processing a Single Image\n"
    "The following example demonstrates how to process a single image using `process_image_task`.\n"
    "The image path is relative to `IMAGE_BASE_PATH`.\n\n"
    "Note: The `instrument_name` parameter allows you to specify a custom instrument configuration. "
    "For more details on instrument configurations, see the [Instrument Configuration Notebook](./2_Instrument_Configuration_Notebook.ipynb)."
))

nb_tasks['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.tasks import process_image_task\n\n"
    "# Example 1: Process a single image with the default configuration\n"
    "image_path = 'prered/example_image.fits'  # Relative to IMAGE_BASE_PATH\n"
    "result = process_image_task.delay(image_path)  # No instrument_name provided\n"
    "print(f'Task ID for processing {image_path}: {result.id}')\n\n"
    "# Example 2: Process a single image with a custom instrument configuration\n"
    "instrument_name = 'my_instrument'  # Use a custom instrument configuration\n"
    "result = process_image_task.delay(image_path, instrument_name=instrument_name)\n"
    "print(f'Task ID for processing {image_path} with {instrument_name}: {result.id}')"
))

# Sección 2: Processing a Directory of Images
nb_tasks['cells'].append(nbf.v4.new_markdown_cell(
    "## Processing a Directory of Images\n"
    "The following examples demonstrate how to use `process_directory_task` to process a directory of images.\n"
    "The directory path is relative to `IMAGE_BASE_PATH`.\n\n"
    "**Parameters:**\n"
    "- `path`: Subdirectory to search for images. If `None`, searches in the base image path.\n"
    "- `filename`: Specific filename or pattern to match. Supports partial matches and wildcards.\n"
    "- `instrument_name`: Overrides the default instrument name. See [Instrument Configuration](./2_Instrument_Configuration_Notebook.ipynb) for details.\n"
    "- `exclude_pattern`: Regular expression pattern to exclude certain filenames.\n"
    "- `reprocess`: If `True`, processes all found images; if `False`, skips images already processed."
))

nb_tasks['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.tasks import process_directory_task\n\n"
    "# Example 1: Process all NPY files in the base directory with the default configuration\n"
    "result = process_directory_task.delay(filename='*.npy')  # No path, no instrument_name\n"
    "print(f'Task ID for processing all NPY files: {result.id}')\n\n"
    "# Example 2: Process all FITS files in a specific subdirectory with the default configuration\n"
    "result = process_directory_task.delay(path='prered', filename='*.fits')  # No instrument_name\n"
    "print(f'Task ID for processing FITS files in \"prered\": {result.id}')\n\n"
    "# Example 3: Process files matching a specific pattern in a subdirectory with a custom configuration\n"
    "result = process_directory_task.delay(path='prered', filename='*QHY411*.fits', instrument_name='QHY411MERIS')\n"
    "print(f'Task ID for processing QHY411 files in \"prered\": {result.id}')\n\n"
    "# Example 4: Process files matching a specific pattern in the base directory with a custom configuration\n"
    "result = process_directory_task.delay(filename='*iKon936*.fits', instrument_name='iKon936')\n"
    "print(f'Task ID for processing iKon936 files: {result.id}')\n\n"
    "# Example 5: Process files in a subdirectory, excluding files matching a pattern\n"
    "result = process_directory_task.delay(path='prered', filename='*.fits', exclude_pattern='*bad*')\n"
    "print(f'Task ID for processing FITS files in \"prered\" (excluding \"bad\" files): {result.id}')\n\n"
    "# Example 6: Process files in a subdirectory without reprocessing already processed files\n"
    "result = process_directory_task.delay(path='prered', filename='*.fits', reprocess=False)\n"
    "print(f'Task ID for processing FITS files in \"prered\" (no reprocessing): {result.id}')\n\n"
    "# Example 7: Process files in a subdirectory with a custom configuration and exclude pattern\n"
    "result = process_directory_task.delay(path='prered', filename='*.fits', instrument_name='iKon936', exclude_pattern='*bad*')\n"
    "print(f'Task ID for processing FITS files in \"prered\" with iKon936 config (excluding \"bad\" files): {result.id}')\n\n"
    "# Example 8: Process files in a subdirectory with a custom configuration and force reprocessing\n"
    "result = process_directory_task.delay(path='prered', filename='*.fits', instrument_name='iKon936', reprocess=True)\n"
    "print(f'Task ID for processing FITS files in \"prered\" with iKon936 config (force reprocessing): {result.id}')\n\n"
    "# Example 9: Process files in the base directory with a custom configuration and exclude pattern\n"
    "result = process_directory_task.delay(filename='*.fits', instrument_name='iKon936', exclude_pattern='*bad*')\n"
    "print(f'Task ID for processing FITS files with iKon936 config (excluding \"bad\" files): {result.id}')\n\n"
    "# Example 10: Process files in the base directory with a custom configuration and force reprocessing\n"
    "result = process_directory_task.delay(filename='*.fits', instrument_name='iKon936', reprocess=True)\n"
    "print(f'Task ID for processing FITS files with iKon936 config (force reprocessing): {result.id}')"
))

# Sección 3: Checking Task Status
nb_tasks['cells'].append(nbf.v4.new_markdown_cell(
    "## Checking Task Status\n"
    "After submitting a task, you can check its status using the task ID."
))

nb_tasks['cells'].append(nbf.v4.new_code_cell(
    "from celery.result import AsyncResult\n\n"
    "# Example: Check the status of a task\n"
    "task_id = result.id  # Replace with your task ID\n"
    "task_result = AsyncResult(task_id)\n\n"
    "print(f'Task status: {task_result.status}')\n"
    "print(f'Task result: {task_result.result}')"
))

# Sección 4: Sending Tasks Directly with Celery App
nb_tasks['cells'].append(nbf.v4.new_markdown_cell(
    "## Sending Tasks Directly with Celery App\n"
    "In addition to using the predefined task functions (`process_image_task` and `process_directory_task`), "
    "you can send tasks directly using the Celery application (`app`). This approach provides more flexibility "
    "and control over task submission.\n\n"
    "### Why Use `send_task`?\n"
    "Using `send_task` allows you to integrate task submission directly into your own scripts or systems. "
    "For example, if you have a system for capturing astronomical images, you can automatically send a task "
    "to process each image as soon as it is saved. If the images are saved within the `IMAGE_PATH` directory, "
    "you can use the relative path to the image as the argument for the task.\n\n"
    "### Example: Sending a Task with `send_task`\n"
    "The following example demonstrates how to send a task directly using the `send_task` method. "
    "This method requires the name of the task and the arguments to pass to it."
))

# Celda para enviar una tarea directamente con Celery App
nb_tasks['cells'].append(nbf.v4.new_code_cell(
    "from gpuphot_worker.worker_app import app\n\n"
    "# Example: Send a task to process a single image\n"
    "task_name = 'gpuphot_worker.tasks.process_image_task'  # Name of the task\n"
    "image_path = 'prered/example_image.fits'  # Relative to IMAGE_PATH\n"
    "instrument_name = 'my_instrument'  # Optional: Custom instrument configuration\n\n"
    "# Send the task\n"
    "result = app.send_task(task_name, args=[image_path], kwargs={'instrument_name': instrument_name})\n"
    "print(f'Task ID for processing {image_path} with {instrument_name}: {result.id}')\n\n"
    "# Check the task status\n"
    "from celery.result import AsyncResult\n"
    "task_result = AsyncResult(result.id)\n"
    "print(f'Task status: {task_result.status}')\n"
    "print(f'Task result: {task_result.result}')"
))

# Sección final: Link to the previous and next notebook
nb_tasks['cells'].append(nbf.v4.new_markdown_cell(
    "## Next Steps\n"
    "If you need to revisit or modify instrument configurations, go back to the previous notebook:\n"
    "- [2. Instrument Configuration](./2_Instrument_Configuration_Notebook.ipynb)\n\n"
    "To explore the database and query photometric data, proceed to the next notebook:\n"
    "- [4. Database Query](./4_Database_Query_Notebook.ipynb)"
))

# Guardar el notebook de tareas
output_dir = os.path.join(Path(__file__).parent.absolute(), '..', '..', 'notebooks')
output_path_config = os.path.join(output_dir, "3_Task_Execution_Notebook.ipynb")
os.makedirs(output_dir, exist_ok=True)

with open(output_path_config, 'w', encoding='utf-8') as f:
    nbf.write(nb_tasks, f)

print(f"Configuration notebook created successfully: {output_path_config}")