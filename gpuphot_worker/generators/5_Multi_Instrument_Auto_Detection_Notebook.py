# SPDX-License-Identifier: MIT
"""
Notebook generator: Multi-Instrument Auto-Detection

Generates a Jupyter notebook that demonstrates how to read the instrument
name directly from a FITS header and automatically dispatch the correct
processing configuration — without hard-coding the instrument name per call.

This is the recommended pattern for mixed-instrument pipelines (e.g. a
telescope that alternates between multiple cameras) where each image carries
the instrument name in its own header.
"""

import argparse
import os
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "# Multi-Instrument Auto-Detection\n\n"
    "This notebook demonstrates how to read the instrument name directly from "
    "a FITS header and let GPUPhot automatically select the matching configuration "
    "file — no need to hard-code the instrument name per call.\n\n"
    "**When to use this pattern:**\n"
    "- Your archive contains images from multiple cameras.\n"
    "- Each FITS file carries the instrument name in its header "
    "(`INSTRUME`, `CAMERA`, or similar).\n"
    "- You want a single generic submission loop that works for any instrument.\n\n"
    "**How it works:**  \n"
    "1. Read the FITS header and extract the instrument keyword.  \n"
    "2. Pass it as `instrument_name` to `process_image_task`.  \n"
    "3. The worker looks up `<instrument_name>.json` in `/data/instrument_configs` "
    "and uses that configuration.  \n\n"
    "> See [Instrument Configuration](./2_Instrument_Configuration_Notebook.ipynb) "
    "for details on creating configuration files."
))

# ---------------------------------------------------------------------------
# Section 1: Reading instrument name from header
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 1 — Reading the Instrument Name from a FITS Header\n\n"
    "The standard FITS keyword is `INSTRUME`, but some observatories use "
    "`CAMERA` or custom keywords.  \n"
    "The cell below shows a robust helper that tries common keywords in order."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "from astropy.io import fits\n"
    "import os\n\n"
    "IMAGES_PATH = os.getenv('IMAGE_BASE_PATH', '/data/images')\n"
    "CONFIGS_PATH = os.getenv('INSTRUMENT_CONFIG_BASE_PATH', '/data/instrument_configs')\n\n"
    "# Keywords tried in order — adapt to your observatory's convention\n"
    "_INSTRUMENT_KEYWORDS = ['INSTRUME', 'CAMERA', 'DETECTOR', 'TELESCOP']\n\n"
    "def get_instrument_from_header(fits_path: str, keywords=_INSTRUMENT_KEYWORDS) -> str | None:\n"
    "    \"\"\"Return the first non-empty instrument keyword found in the FITS header.\"\"\"\n"
    "    with fits.open(fits_path) as hdul:\n"
    "        header = hdul[0].header\n"
    "    for kw in keywords:\n"
    "        value = header.get(kw, '').strip()\n"
    "        if value:\n"
    "            return value\n"
    "    return None\n\n"
    "# Quick test — pick any image from the images directory\n"
    "sample_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.fits')][:3]\n"
    "for fname in sample_files:\n"
    "    instr = get_instrument_from_header(os.path.join(IMAGES_PATH, fname))\n"
    "    print(f'{fname:60s}  →  {instr}')"
))

# ---------------------------------------------------------------------------
# Section 2: Check which configs exist
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 2 — Verify Available Instrument Configurations\n\n"
    "Before submitting tasks, confirm that a matching `.json` configuration "
    "exists for every instrument found in your images."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "available_configs = {\n"
    "    os.path.splitext(f)[0]\n"
    "    for f in os.listdir(CONFIGS_PATH)\n"
    "    if f.endswith('.json')\n"
    "}\n"
    "print('Available instrument configs:', sorted(available_configs))\n\n"
    "# Check coverage for all images\n"
    "all_fits = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.fits')]\n"
    "missing = set()\n"
    "for fname in all_fits:\n"
    "    instr = get_instrument_from_header(os.path.join(IMAGES_PATH, fname))\n"
    "    if instr and instr not in available_configs:\n"
    "        missing.add(instr)\n\n"
    "if missing:\n"
    "    print('⚠ Missing configs for:', sorted(missing))\n"
    "    print('Create them in CONFIGS_PATH or see the Instrument Configuration notebook.')\n"
    "else:\n"
    "    print('✓ All instruments have a matching config.')"
))

# ---------------------------------------------------------------------------
# Section 3: Submit a single image with auto-detection
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 3 — Submit a Single Image with Auto-Detected Instrument\n\n"
    "Read the header, extract the instrument name, and submit the task in one go."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "from celery import Celery\n\n"
    "broker  = os.getenv('CELERY_BROKER_URL',  'amqp://gpuphot:gpuphot@rabbitmq:5672/')\n"
    "backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')\n"
    "app = Celery(broker=broker, backend=backend)\n\n"
    "# Choose any image from the list\n"
    "image_file = sample_files[0] if sample_files else 'example.fits'\n"
    "instrument = get_instrument_from_header(os.path.join(IMAGES_PATH, image_file))\n\n"
    "print(f'Image     : {image_file}')\n"
    "print(f'Instrument: {instrument}')\n\n"
    "if instrument and instrument in available_configs:\n"
    "    result = app.send_task(\n"
    "        'gpuphot_worker.tasks.process_image_task',\n"
    "        args=[image_file, instrument],\n"
    "    )\n"
    "    print(f'Task submitted — ID: {result.id}')\n"
    "else:\n"
    "    print('Skipping: instrument not found or no matching config.')"
))

# ---------------------------------------------------------------------------
# Section 4: Batch submission loop
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 4 — Batch Submission for a Mixed-Instrument Archive\n\n"
    "The pattern below iterates over every FITS file in the images directory, "
    "reads each header once, and submits a task only when a matching "
    "configuration exists.  Images without a config are logged and skipped "
    "rather than raising an error."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "submitted, skipped = [], []\n\n"
    "for fname in sorted(all_fits):\n"
    "    full_path = os.path.join(IMAGES_PATH, fname)\n"
    "    instr = get_instrument_from_header(full_path)\n\n"
    "    if not instr:\n"
    "        print(f'[SKIP] {fname}: no instrument keyword found')\n"
    "        skipped.append(fname)\n"
    "        continue\n\n"
    "    if instr not in available_configs:\n"
    "        print(f'[SKIP] {fname}: no config for \"{instr}\"')\n"
    "        skipped.append(fname)\n"
    "        continue\n\n"
    "    result = app.send_task(\n"
    "        'gpuphot_worker.tasks.process_image_task',\n"
    "        args=[fname, instr],\n"
    "    )\n"
    "    submitted.append((fname, instr, result.id))\n"
    "    print(f'[OK]   {fname} ({instr}) → task {result.id}')\n\n"
    "print(f'\\nSubmitted: {len(submitted)}  |  Skipped: {len(skipped)}')"
))

# ---------------------------------------------------------------------------
# Section 5: Monitor batch results
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 5 — Monitor Batch Results\n\n"
    "Poll task states until all submitted tasks are finished.  "
    "Adjust `poll_interval` and `timeout` to match your expected processing time."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "import time\n"
    "from celery.result import AsyncResult\n\n"
    "poll_interval = 10   # seconds between polls\n"
    "timeout       = 600  # maximum wait in seconds\n\n"
    "pending = {task_id: fname for fname, _, task_id in submitted}\n"
    "start   = time.time()\n\n"
    "while pending and (time.time() - start) < timeout:\n"
    "    for task_id in list(pending):\n"
    "        state = AsyncResult(task_id, app=app).state\n"
    "        if state in ('SUCCESS', 'FAILURE', 'REVOKED'):\n"
    "            fname = pending.pop(task_id)\n"
    "            print(f'[{state:7s}] {fname} ({task_id})')\n"
    "    if pending:\n"
    "        time.sleep(poll_interval)\n\n"
    "if pending:\n"
    "    print(f'Timeout reached. Still pending: {list(pending.values())}')\n"
    "else:\n"
    "    elapsed = time.time() - start\n"
    "    print(f'\\nAll tasks finished in {elapsed:.1f}s')"
))

# ---------------------------------------------------------------------------
# Section 6: Filename-based instrument extraction (alternative)
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 6 — Alternative: Extract Instrument from Filename\n\n"
    "If your filenames follow a convention like  \n"
    "`TTT1_QHY411-1_2026-03-09_..._Lum.fits`  \n"
    "you can extract the instrument without opening the file, which is faster "
    "for large archives."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "import re\n\n"
    "# Pattern assumes: <telescope>_<instrument>_<date>_...\n"
    "FILENAME_PATTERN = re.compile(r'^[^_]+_([^_]+)_')\n\n"
    "def get_instrument_from_filename(fname: str) -> str | None:\n"
    "    m = FILENAME_PATTERN.match(os.path.basename(fname))\n"
    "    return m.group(1) if m else None\n\n"
    "# Preview\n"
    "for fname in sorted(all_fits)[:5]:\n"
    "    instr_hdr  = get_instrument_from_header(os.path.join(IMAGES_PATH, fname))\n"
    "    instr_name = get_instrument_from_filename(fname)\n"
    "    match = '✓' if instr_hdr == instr_name else '≠'\n"
    "    print(f'{match}  header={instr_hdr:15s}  filename={instr_name}')"
))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate the Multi-Instrument Auto-Detection notebook."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(Path(__file__).resolve().parent, '..', '..', 'notebooks'),
        help="Directory where the .ipynb file will be written.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir) / "5_Multi_Instrument_Auto_Detection_Notebook.ipynb"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {output_path}")


if __name__ == "__main__":
    main()
