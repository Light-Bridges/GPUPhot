#!/usr/bin/env python3
"""Adapt an acceptance block to the schema read by build_postfix_latency_csv.py.

WHY THIS EXISTS. The remeasurement runners write the schema the ACCEPTANCE
CRITERION needs (`role`, `load_per_core`, `host_nproc`, controls kept separate),
while the integrator reads the CAMPAIGN schema (`gpu_name`, `profiler_label`).
Converting by hand invites silent mistakes, so the conversion is a script and
states what it does.

WHAT GOES IN AND WHAT DOESN'T. Only rows with `role='objetivo'` (the target
measurement). Control rows are legitimate measurements of ANOTHER image, and
including them would also change that other published cell, which nobody has
asked to touch; they stay in their own file, available but not integrated.
Warm-up repetitions (`is_warmup=1`) are kept: the integrator does not look at
them, but `load_benchmark()` later applies its own rule, and it is better for
it to see the whole block than one already trimmed twice by different criteria.

Usage: python3 benchmarks/adapt_acceptance_block.py <file> <profiler_label> <gpu_name>
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, arm, gpu = sys.argv[1], sys.argv[2], sys.argv[3]
    cab = [l for l in open(src) if l.startswith('#')]
    d = pd.read_csv(src, comment='#', low_memory=False)
    if 'role' not in d.columns:
        raise SystemExit(f'{src}: no trae columna role; no es un bloque de aceptación')
    d = d[d['role'] == 'objetivo'].drop(columns=['role'])
    if not len(d):
        raise SystemExit(f'{src}: no hay filas role="objetivo"')
    d['gpu_name'] = gpu
    d['profiler_label'] = arm
    d['catalog_backend'] = 'local'
    d['remeasure_round'] = 'postfix_coldblock'   # MUST start with 'postfix' or 'fix': that is
    # what build_postfix_latency_csv.py looks at to decide whether a row belongs to the post-fix campaign.
    base = os.path.basename(src).replace('.csv', '')
    out = os.path.join(DATA, f'measure_postfix_{base}.csv')
    with open(out, 'w') as fh:
        fh.writelines(cab)
        fh.write(f'# ADAPTED by adapt_acceptance_block.py: target role only ({len(d)} rows of '
                 f'{len(pd.read_csv(src, comment="#"))} original), gpu_name={gpu!r}, '
                 f'profiler_label={arm!r}. Control rows remain in source file.\n')
        d.to_csv(fh, index=False)
    print(f'escrito {out}  ({len(d)} filas, {int((d.is_warmup == 0).sum())} cronometradas)')


if __name__ == '__main__':
    main()
