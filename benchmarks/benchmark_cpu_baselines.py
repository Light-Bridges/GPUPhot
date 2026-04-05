#!/usr/bin/env python3
"""
CPU baseline benchmarks for comparison with GPUPhot.

Runs SEP (Python SExtractor) and optionally Photutils on the same benchmark
images used for GPUPhot performance evaluation. Measures wall-clock time for
source detection and aperture photometry.

This script is for the SOFTWARE paper (time comparison only).
Photometric accuracy validation is out of scope (Paper 2, Alarcon et al.).

Usage:
    python benchmarks/benchmark_cpu_baselines.py
    python benchmarks/benchmark_cpu_baselines.py --images-dir /path/to/fits
    python benchmarks/benchmark_cpu_baselines.py --repeats 5 --warmup 1
"""

import argparse
import csv
import glob
import os
import platform
import sys
import time

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# SEP baseline (Python C-extension reimplementation of SExtractor)
# ---------------------------------------------------------------------------

def run_sep_pipeline(data, gain=1.0):
    """
    Run a complete SEP detection + aperture photometry pipeline.

    This mirrors the operations GPUPhot performs:
    background estimation, source detection, aperture photometry.

    Returns (n_sources, elapsed_seconds).
    """
    import sep

    # SEP requires C-contiguous float32/float64 with native byte order
    if data.dtype.byteorder not in ('=', '<' if sys.byteorder == 'little' else '>'):
        data = data.byteswap().newbyteorder()
    data = np.ascontiguousarray(data, dtype=np.float64)

    t0 = time.perf_counter()

    # 1. Background estimation
    bkg = sep.Background(data)
    data_sub = data - bkg

    # 2. Source detection (threshold = 5 sigma, matching GPUPhot default min_snr=5)
    objects = sep.extract(data_sub, thresh=5.0, err=bkg.globalrms)

    # 3. Aperture photometry (radius = 5 px, representative fixed aperture)
    flux, fluxerr, flag = sep.sum_circle(
        data_sub, objects['x'], objects['y'], r=5.0, err=bkg.globalrms, gain=gain
    )

    elapsed = time.perf_counter() - t0
    return len(objects), elapsed


# ---------------------------------------------------------------------------
# Photutils baseline (pure Python / Astropy ecosystem)
# ---------------------------------------------------------------------------

def run_photutils_pipeline(data, gain=1.0):
    """
    Run a complete Photutils detection + aperture photometry pipeline.

    Returns (n_sources, elapsed_seconds) or None if photutils is not installed.
    """
    try:
        from photutils.aperture import CircularAperture, aperture_photometry
        from photutils.background import Background2D, MedianBackground
        from photutils.detection import DAOStarFinder
    except ImportError:
        return None

    data = np.ascontiguousarray(data, dtype=np.float64)

    t0 = time.perf_counter()

    # 1. Background estimation
    bkg = Background2D(data, box_size=64, bkg_estimator=MedianBackground())
    data_sub = data - bkg.background

    # 2. Source detection (threshold = 5 * bkg.background_rms_median, fwhm=5)
    threshold = 5.0 * bkg.background_rms_median
    finder = DAOStarFinder(fwhm=5.0, threshold=threshold)
    sources = finder(data_sub)

    if sources is None or len(sources) == 0:
        elapsed = time.perf_counter() - t0
        return 0, elapsed

    # 3. Aperture photometry
    positions = np.column_stack([sources['xcentroid'], sources['ycentroid']])
    apertures = CircularAperture(positions, r=5.0)
    phot_table = aperture_photometry(data_sub, apertures)

    elapsed = time.perf_counter() - t0
    return len(sources), elapsed


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_image(filepath, repeats=3, warmup=1):
    """Benchmark a single FITS image with all available tools."""
    hdul = fits.open(filepath)
    data = hdul[0].data.astype(np.float64)
    hdr = hdul[0].header
    hdul.close()

    ny, nx = data.shape
    megapixels = nx * ny / 1e6
    gain = float(hdr.get('GAIN', 1.0))
    basename = os.path.basename(filepath)

    results = {
        'filename': basename,
        'nx': nx,
        'ny': ny,
        'megapixels': round(megapixels, 1),
    }

    # --- SEP ---
    # Warmup
    for _ in range(warmup):
        run_sep_pipeline(data, gain=gain)

    sep_times = []
    sep_nsrc = 0
    for _ in range(repeats):
        nsrc, elapsed = run_sep_pipeline(data, gain=gain)
        sep_times.append(elapsed)
        sep_nsrc = nsrc

    results['sep_median_s'] = round(np.median(sep_times), 3)
    results['sep_std_s'] = round(np.std(sep_times), 3)
    results['sep_sources'] = sep_nsrc

    # --- Photutils ---
    res = run_photutils_pipeline(data, gain=gain)
    if res is not None:
        # Warmup
        for _ in range(warmup):
            run_photutils_pipeline(data, gain=gain)

        phot_times = []
        phot_nsrc = 0
        for _ in range(repeats):
            nsrc, elapsed = run_photutils_pipeline(data, gain=gain)
            phot_times.append(elapsed)
            phot_nsrc = nsrc

        results['photutils_median_s'] = round(np.median(phot_times), 3)
        results['photutils_std_s'] = round(np.std(phot_times), 3)
        results['photutils_sources'] = phot_nsrc
    else:
        results['photutils_median_s'] = None
        results['photutils_std_s'] = None
        results['photutils_sources'] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CPU baseline benchmarks (SEP / Photutils) for GPUPhot paper'
    )
    parser.add_argument(
        '--images-dir',
        default=os.path.join(os.path.dirname(__file__), 'benchmark_images'),
        help='Directory with FITS images',
    )
    parser.add_argument('--repeats', type=int, default=3, help='Measured repeats per image')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup runs per image')
    parser.add_argument(
        '--output',
        default=os.path.join(os.path.dirname(__file__), 'cpu_baseline_results.csv'),
        help='Output CSV path',
    )
    args = parser.parse_args()

    fits_files = sorted(glob.glob(os.path.join(args.images_dir, '*.fits')))
    if not fits_files:
        print(f'No FITS files found in {args.images_dir}')
        sys.exit(1)

    print(f'Found {len(fits_files)} images in {args.images_dir}')
    print(f'Repeats: {args.repeats}, Warmup: {args.warmup}')
    print(f'Platform: {platform.node()} / {platform.processor()}')
    print()

    all_results = []
    for filepath in fits_files:
        basename = os.path.basename(filepath)
        print(f'Processing {basename}...', end=' ', flush=True)
        results = benchmark_image(filepath, repeats=args.repeats, warmup=args.warmup)
        all_results.append(results)

        sep_info = f"SEP: {results['sep_median_s']:.3f}s ({results['sep_sources']} src)"
        phot_info = ''
        if results['photutils_median_s'] is not None:
            phot_info = f"  Photutils: {results['photutils_median_s']:.3f}s ({results['photutils_sources']} src)"
        print(f'{sep_info}{phot_info}')

    # Write CSV
    fieldnames = [
        'filename', 'nx', 'ny', 'megapixels',
        'sep_median_s', 'sep_std_s', 'sep_sources',
        'photutils_median_s', 'photutils_std_s', 'photutils_sources',
    ]
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f'\nResults saved to {args.output}')

    # Print summary table
    print(f'\n{"Image":>60}  {"MP":>6}  {"SEP (s)":>8}  {"Phot (s)":>9}  {"SEP src":>8}')
    print('-' * 100)
    for r in all_results:
        phot_str = f"{r['photutils_median_s']:>9.3f}" if r['photutils_median_s'] is not None else '      N/A'
        print(f"{r['filename']:>60}  {r['megapixels']:>6.1f}  {r['sep_median_s']:>8.3f}  {phot_str}  {r['sep_sources']:>8}")


if __name__ == '__main__':
    main()
