#!/usr/bin/env python3
"""
Integrate new cuml ablation results into benchmarks/data/cuml_ablation.csv.

Usage:
    python3 benchmarks/dev/integrate_cuml_ablation.py <new_results.csv>

The new results CSV (from run_cuml_ablation_a100.sh) has format:
    image,mp,cuml,rep,time_s,sources

This script appends the new rows to benchmarks/data/cuml_ablation.csv,
ensuring no duplicates (checks image filename × cuml mode × rep).
"""

import csv
import sys
from pathlib import Path

TARGET = Path('benchmarks/data/cuml_ablation.csv')


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <new_results.csv>')
        sys.exit(1)

    new_file = Path(sys.argv[1])
    if not new_file.exists():
        print(f'ERROR: {new_file} not found')
        sys.exit(1)

    # Load existing
    with open(TARGET) as f:
        reader = csv.DictReader(f)
        existing = list(reader)
        fieldnames = list(reader.fieldnames)

    # Load new rows
    with open(new_file) as f:
        reader = csv.DictReader(f)
        new_rows = list(reader)

    print(f'Existing rows: {len(existing)} (excl. DONE marker)')
    print(f'New rows: {len(new_rows)}')

    # Build existing key set to detect duplicates
    existing_keys = {
        (r['image'], r['cuml'], r['rep'])
        for r in existing
        if r.get('image') and not r['image'].startswith('DONE')
    }

    # Filter new rows
    to_add = []
    skipped = []
    for r in new_rows:
        if not r.get('time_s') or not r.get('image'):
            print(f'  SKIP (empty): {r}')
            continue
        key = (r['image'], r['cuml'], r['rep'])
        if key in existing_keys:
            skipped.append(r)
        else:
            to_add.append(r)

    if skipped:
        print(f'Skipping {len(skipped)} duplicate rows (image+cuml+rep already exists)')

    if not to_add:
        print('Nothing to add.')
        return

    # Remove trailing DONE line if present
    clean_existing = [r for r in existing if r.get('image') != 'DONE']

    # Append new rows
    all_rows = clean_existing + to_add

    with open(TARGET, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        # Restore DONE marker
        f.write('DONE\n')

    print(f'\nAdded {len(to_add)} rows to {TARGET}')
    print(f'Total rows now: {len(all_rows)}')

    # Summary
    print('\nNew data summary:')
    images = {}
    for r in to_add:
        key = (r['image'].split('_')[-2] if '_' in r['image'] else r['image'][:20], r['cuml'])
        images.setdefault(key, []).append(float(r['time_s']))

    for (img, cuml), times in sorted(images.items()):
        print(f'  {img} cuml={cuml}: n={len(times)}  times={[round(t,1) for t in times]}')


if __name__ == '__main__':
    main()
