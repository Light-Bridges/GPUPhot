#!/usr/bin/env python3
"""
Direct profiling of gpuphot's core process_image function.

Unlike profile_image_processing.py (which goes through Celery's
process_image_task), this script calls ImageProcessor.process_image
directly — no Celery, no database, no file saving.  This gives a
clean measurement of the GPU photometry pipeline alone.

Usage
-----
::

    python profiling_scripts/profile_process_image.py <image_file> [instrument_name]

    # With nsys:
    nsys profile --trace=cuda,nvtx python profiling_scripts/profile_process_image.py <image_file> iKon936

Exit codes
----------
- 0: process_image completed successfully.
- 1: process_image raised an exception (time is still reported).
- 2: image could not be loaded or arguments are wrong.
"""

import os
import sys
import time
from datetime import timedelta

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, '.env'))


def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_process_image.py <image_file> [instrument_name]")
        print("  image_file:      filename (resolved against IMAGE_BASE_PATH) or absolute path")
        print("  instrument_name: instrument config name (default: from INSTRUMENT_NAME env)")
        sys.exit(2)

    image_arg = sys.argv[1]
    instrument_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('INSTRUMENT_NAME', 'default')

    # Resolve image path
    base_path = os.environ.get('IMAGE_BASE_PATH', '/data/images')
    if os.path.isabs(image_arg):
        image_path = image_arg
    else:
        image_path = os.path.join(base_path, image_arg)

    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(2)

    config_dir = os.environ.get('INSTRUMENT_CONFIG_BASE_PATH',
                                os.environ.get('INSTRUMENT_CONFIG_PATH', None))

    print(f"Image:      {os.path.basename(image_path)}")
    print(f"Instrument: {instrument_name}")
    print(f"Config dir: {config_dir}")
    print(f"Image size: {os.path.getsize(image_path) / 1e6:.1f} MB")
    print(flush=True)

    # --- Load image ---
    from astropy.io import fits
    try:
        with fits.open(image_path) as hdul:
            imdata = hdul[0].data
            imheader = hdul[0].header
        naxis1 = imheader.get('NAXIS1', '?')
        naxis2 = imheader.get('NAXIS2', '?')
        print(f"Loaded:     {naxis1} x {naxis2} ({imdata.dtype})")
    except Exception as e:
        print(f"ERROR: Could not load image: {e}", file=sys.stderr)
        sys.exit(2)

    # --- Create processor ---
    from gpuphot.image_processor import ImageProcessor
    try:
        processor = ImageProcessor(instrument_name, config_dir=config_dir)
    except Exception as e:
        print(f"ERROR: Could not create ImageProcessor: {e}", file=sys.stderr)
        sys.exit(2)

    # --- GPU info helper ---
    def gpu_info():
        """Collect GPU state via nvidia-smi and CuPy."""
        info = {}
        try:
            info['python_version'] = sys.version.split()[0]
        except Exception:
            pass
        try:
            import cupy
            info['cupy_version'] = cupy.__version__
            dev = cupy.cuda.Device(0)
            mem_free, mem_total = dev.mem_info
            info['gpu_mem_total_mb'] = round(mem_total / 1e6, 1)
            info['gpu_mem_used_mb'] = round((mem_total - mem_free) / 1e6, 1)
            info['gpu_mem_free_mb'] = round(mem_free / 1e6, 1)
        except Exception:
            pass
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,power.max_limit,power.default_limit,memory.total,memory.used',
                 '--format=csv,noheader,nounits', '-i', '0'],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                info['gpu_name'] = parts[0]
                info['gpu_uuid'] = parts[1]
                info['gpu_driver'] = parts[2]
                info['gpu_temp_c'] = float(parts[3]) if parts[3] not in ('[N/A]', '') else None
                info['gpu_power_draw_w'] = float(parts[4]) if parts[4] not in ('[N/A]', '') else None
                info['gpu_power_max_w'] = float(parts[5]) if parts[5] not in ('[N/A]', '') else None
                info['gpu_power_default_w'] = float(parts[6]) if parts[6] not in ('[N/A]', '') else None
                info['gpu_mem_total_mb'] = float(parts[7]) if parts[7] not in ('[N/A]', '') else None
                info['gpu_mem_used_mb'] = float(parts[8]) if parts[8] not in ('[N/A]', '') else None
        except Exception:
            pass
        return info

    # --- GPU state before processing ---
    gpu_before = gpu_info()
    if gpu_before.get('gpu_name'):
        print(f"GPU:        {gpu_before.get('gpu_name', '?')}")
        print(f"GPU UUID:   {gpu_before.get('gpu_uuid', '?')}")
        print(f"GPU driver: {gpu_before.get('gpu_driver', '?')}")
        print(f"GPU power:  max={gpu_before.get('gpu_power_max_w', '?')}W default={gpu_before.get('gpu_power_default_w', '?')}W draw={gpu_before.get('gpu_power_draw_w', '?')}W")
        print(f"GPU VRAM:   {gpu_before.get('gpu_mem_used_mb', '?')} / {gpu_before.get('gpu_mem_total_mb', '?')} MB")
        print(f"GPU temp:   {gpu_before.get('gpu_temp_c', '?')} C")
        print(f"CuPy:       {gpu_before.get('cupy_version', '?')}")
        print(f"Python:     {gpu_before.get('python_version', '?')}")

    # --- Run process_image ---
    print(f"Processing...", flush=True)
    success = False
    start_time = time.time()
    try:
        phot_df, hwcs = processor.process_image(imdata, imheader)
        elapsed = time.time() - start_time
        success = True

        n_objects = len(phot_df) if phot_df is not None else 0
        n_transients = 0
        if phot_df is not None and 'RAERR' in phot_df.columns:
            import numpy as np
            n_transients = int(np.isnan(phot_df['RAERR']).sum())

        print(f"Status:     OK")
        print(f"Objects:    {n_objects}")
        print(f"Transients: {n_transients}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Status:     FAILED")
        print(f"Error:      {type(e).__name__}: {e}")

    # --- GPU state after processing ---
    gpu_after = gpu_info()

    # Always report time and GPU memory
    readable_time = str(timedelta(seconds=elapsed))
    print(f"Time:       {readable_time} ({elapsed:.3f}s)")
    if gpu_after.get('gpu_mem_used_mb'):
        print(f"GPU VRAM after: {gpu_after.get('gpu_mem_used_mb', '?')} / {gpu_after.get('gpu_mem_total_mb', '?')} MB")
        print(f"GPU temp after: {gpu_after.get('gpu_temp_c', '?')} C")
        peak_mb = gpu_after.get('gpu_mem_used_mb', 0) - gpu_before.get('gpu_mem_used_mb', 0)
        if peak_mb > 0:
            print(f"GPU VRAM delta: +{peak_mb:.1f} MB")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
