#!/usr/bin/env python3
"""Parse L40S py3.8 local-catalog re-measurement timing logs → staging CSV.

Usage:
    python3 benchmarks/parse_l40s_py38_local_remed.py /tmp/l40s_py38_local_remed_<ts>/

Output: benchmarks/data/measure_l40s_py38_local_remed.csv
  - Includes ALL reps (rep0, rep1, ...) — filtering done in analysis
  - source_round='5E_l40s_py38_local_remed'
  - catalog_backend='local'
"""

import re
import sys
import os
import csv
from datetime import datetime

MP_MAP = {
    'QHY411-1_Lum_full':   151.2,
    'QHY411-3_Lum_131k':   151.2,
    'QHY411-3_Lum_full':   151.2,
    'QHY411-3_SDSSg_10k':  151.2,
    'QHY411-3_SDSSr_19k':  151.2,
    'QHY411-3_SDSSr_full': 151.2,
}

OUTPATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'measure_l40s_py38_local_remed.csv'
)

FIELDNAMES = [
    'machine', 'gpu_name', 'python_ver', 'profiler_label', 'environment',
    'image_label', 'mp', 'execution_time', 'n_sources_detected', 'timestamp',
    'naxis1', 'naxis2', 'filter', 'object', 'gpu_mem_total', 'gpu_mem_used',
    'gpu_temp', 'source_round', 'catalog_backend',
]


def parse_elapsed(s):
    if not s or s in ('ERROR', 'ABORT'):
        return None
    m = re.match(r'^(\d+):(\d{2}):(\d{2})\.(\d+)$', s)
    if m:
        h, mi, sec, frac = m.groups()
        return int(h)*3600 + int(mi)*60 + int(sec) + float('0.'+frac)
    return None


def read_timings(logdir):
    timing_file = os.path.join(logdir, 'timings.log')
    if not os.path.exists(timing_file):
        print(f"WARNING: no timings.log in {logdir}", file=sys.stderr)
        return []
    rows = []
    with open(timing_file) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('TIMING '):
                continue
            kv = dict(part.split('=', 1) for part in line[len('TIMING '):].split())
            rows.append(kv)
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    all_rows = []
    now_ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')

    for logdir in sys.argv[1:]:
        logdir = logdir.rstrip('/')
        timings = read_timings(logdir)
        print(f"  {logdir}: {len(timings)} TIMING lines")

        for kv in timings:
            rep = int(kv.get('rep', -1)) if kv.get('rep', 'ABORTED') not in ('ABORTED',) else -1
            elapsed_str = kv.get('elapsed', '')
            elapsed_s = parse_elapsed(elapsed_str)

            machine = kv.get('machine', 'hp3')
            image_label = kv.get('image_label', '')
            catalog = kv.get('catalog', 'local')

            all_rows.append({
                'machine':            machine,
                'gpu_name':           'NVIDIA L40S',
                'python_ver':         '3.8',
                'profiler_label':     'py38_baseline',
                'environment':        'profiler',
                'image_label':        image_label,
                'mp':                 MP_MAP.get(image_label, 151.2),
                'execution_time':     round(elapsed_s, 5) if elapsed_s is not None else '',
                'n_sources_detected': '',
                'timestamp':          now_ts,
                'naxis1':             '',
                'naxis2':             '',
                'filter':             '',
                'object':             '',
                'gpu_mem_total':      '',
                'gpu_mem_used':       '',
                'gpu_temp':           '',
                'source_round':       '5E_l40s_py38_local_remed',
                'catalog_backend':    catalog,
            })

    print(f"\nTotal rows (all reps including warmup): {len(all_rows)}")

    import collections
    counts_ok = collections.Counter()
    counts_err = collections.Counter()
    for r in all_rows:
        key = (r['image_label'],)
        if r['execution_time']:
            counts_ok[key] += 1
        else:
            counts_err[key] += 1

    print("\nReps per image (OK / ERR):")
    for img in sorted(set(list(counts_ok.keys()) + list(counts_err.keys()))):
        ok = counts_ok.get(img, 0)
        err = counts_err.get(img, 0)
        print(f"  {img[0]}: OK={ok}  ERR={err}")

    with open(OUTPATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nWritten: {OUTPATH}")


if __name__ == '__main__':
    main()
