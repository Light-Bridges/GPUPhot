#!/usr/bin/env python3
"""Cell-status ledger: why is a (profile, GPU, image) cell empty?

The unified latency CSV only carries runs that produced a result table, so a
missing cell there is ambiguous: it may have run out of memory, it may have
failed for another reason, it may have been withheld by the culling rules, or
it may never have been launched at all.  The heatmap used to label every empty
cell "OOM", which is false for the ones that were never launched.

Statuses
--------
measured  the cell survives into the unified CSV
oom       every attempt failed, at least one with a device out-of-memory signature
oom_host  every attempt failed for want of pinned host memory, before the device
          was ever loaded; declared, see DECLARED_STATUS
error     every attempt failed, none of them with an out-of-memory signature
withheld  attempts succeeded but the culling rules removed the cell
hang      the attempt took the host down; declared, see DECLARED_STATUS
not_run   the cell was never launched on that machine

Every status but `hang` is derived from what reached Elasticsearch.  A host that
hangs writes no event, so those cells arrive here indistinguishable from cells
nobody ever launched, and would be labelled "not run" -- which says the attempt
was never made when in fact the machine went down making it.  A host that dies
cannot file its own death certificate, so that one status is declared by hand
from the measurement agent's own record, with its source named in the table
below, and only for cells where it was actually observed.
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from build_unified_csv import RAW_CSV, MACHINE_MAP, GPU_NAME_MAP   # noqa: E402
from generate_manuscript_tables import GPU_LABEL_MAP                        # noqa: E402

UNIFIED = os.path.join(BASE, 'data', 'benchmark_latency.csv')
OUT_CSV = os.path.join(BASE, 'data', 'cell_status.csv')

# 'out of memory' covers both CuPy's own message ("Out of memory allocating N bytes")
# and the driver's ("cudaErrorMemoryAllocation: out of memory"); RMM raises
# std::bad_alloc with out_of_memory in the text.
OOM_MARKERS = ('out of memory', 'out_of_memory', 'bad_alloc')

# Cells whose verdict cannot come from Elasticsearch, keyed by
# (profiler_label, gpu_label, image_label) -> (status, evidence).
#
# The Orin NX runs JetPack 5 on an old kernel, and there memory exhaustion inside
# the multi-stream FFT section takes the whole host down instead of raising: the
# log stops dead after "CRITICAL MEMORY PRESSURE (ratio 0.984, Device 7.1/7.3
# GiB)", a free_all_blocks, and five concurrent FFT streams at 6.15 GB.  The Orin
# Super, same 8 GB of unified memory but JetPack 6, raises a textbook OOM on the
# same images.  Two independent occurrences, listed here and nothing beyond them:
# a third is suspected on QHY600-3_Lum (jo_exp2, 2026-08-29) but its log was never
# harvested, so it is not counted.
# The RTX 3090 host (ttt_server) has a second wall in front of the device one.  An
# instrumented shot of QHY411-3_Lum_131k on 2026-08-30 sampled VRAM and host MemFree
# once a second: the card peaked at 264 MB of 24,576 while host MemFree stayed
# between 750 and 920 MB, and the run died at 5.8 s.  The GPU was essentially empty.
# What failed is the pinned host allocation that stages a 151.2 MP frame, starved by
# a catalog postgres holding 32 GB of shm.  The exception is masked as
# cudaErrorInvalidValue because the containers predate 841988f.
#
# It is declared for the 151.2 MP cells whose every attempt carries that masked
# signature, and ONLY those: the guard below refuses to overwrite a cell that did
# record a device OOM, because both walls are real and which one a run hits depends
# on how much host memory happened to be free at the time.  The staging allocation is
# set by the frame, and every frame here is 151.2 MP: the same ~1.21 GB request
# appears byte-identical on two other hosts and three different 151.2 MP images.
# Evidence: data/raw/probe_rtx3090_host_memory.csv + tiro_3090_traceback.txt.
_TTT_3090_HOST_OOM = (
    'oom_host', 'pinned host memory starved (MemFree 750-920 MB, VRAM peak 264 MB '
                'of 24,576); instrumented shot 2026-08-30, data/raw/probe_rtx3090_host_memory.csv '
                '+ tiro_3090_traceback.txt'
)
_MP151 = ['QHY411-1_Lum_full', 'QHY411-3_Lum_131k', 'QHY411-3_Lum_full',
          'QHY411-3_SDSSg_10k', 'QHY411-3_SDSSr_19k', 'QHY411-3_SDSSr_full']

# Applied only where the derived status is 'error' (see the guard in build()).
DECLARED_IF_ERROR = {
    (prof, 'RTX 3090 (24 GB)', img): _TTT_3090_HOST_OOM
    for prof in ('py38_baseline', 'py312_cuml_adaptive', 'py312_cuml_always')
    for img in _MP151
}

DECLARED_STATUS = {
    ('py38_baseline', 'Orin NX (8 GB)', 'QHY600-3_Lum'):
        ('hang', 'host down mid-run, 2026-08-30 sonda91 (img3); '
                 'data/raw/probe_orin_host_down.csv + sonda91_tracebacks_20260830/'
                 'jetson_orin_py38_img3.txt'),
    ('py38_baseline', 'Orin NX (8 GB)', 'QHY411-1_Lum_bin2'):
        ('hang', 'host down mid-run on 2012QD8, campaign record; '
                 'the run this cell was retried for was never launched '
                 '(sonda91: img6 HOST_YA_CAIDO_NO_DISPARADO)'),
}



def _is_oom(text):
    t = str(text).lower()
    return any(m in t for m in OOM_MARKERS)


def build(raw_csv=RAW_CSV, unified_csv=UNIFIED, out_csv=OUT_CSV):
    raw = pd.read_csv(raw_csv)
    raw = raw[raw['campana'] != 'nsys'].copy()
    raw['machine'] = raw['machine'].replace(MACHINE_MAP)
    raw['gpu_name'] = raw['gpu_name'].replace(GPU_NAME_MAP)
    raw['gpu_label'] = (raw['gpu_name'].str.replace('NVIDIA ', '', regex=False)
                        .map(GPU_LABEL_MAP).fillna(raw['gpu_name']))

    exc = raw['excepcion'].fillna('').astype(str).str.strip()
    raw['failed'] = exc != ''
    raw['oom'] = exc.map(_is_oom)

    uni = pd.read_csv(unified_csv)
    uni['gpu_label'] = (uni['gpu_name'].str.replace('NVIDIA ', '', regex=False)
                        .map(GPU_LABEL_MAP).fillna(uni['gpu_name']))
    kept = set(map(tuple, uni[['profiler_label', 'gpu_label', 'image_label']]
                   .drop_duplicates().to_numpy()))

    rows = []
    grp = raw.groupby(['profiler_label', 'gpu_label', 'image_label'], sort=True)
    for (prof, gpu, img), g in grp:
        n_att = len(g)
        n_ok = int((~g['failed']).sum())
        n_oom = int(g['oom'].sum())
        if (prof, gpu, img) in kept:
            status = 'measured'
        elif n_ok > 0:
            status = 'withheld'
        elif n_oom > 0:
            status = 'oom'
        else:
            status = 'error'
        detail = ''
        if status in ('oom', 'error'):
            top = g.loc[g['failed'], 'excepcion'].astype(str).str.slice(0, 120)
            detail = top.value_counts().index[0] if len(top) else ''
        declared = DECLARED_STATUS.get((prof, gpu, img))
        if status == 'error':
            declared = declared or DECLARED_IF_ERROR.get((prof, gpu, img))
        if declared:
            status, detail = declared
        rows.append(dict(profiler_label=prof, gpu_label=gpu, image_label=img,
                         status=status, n_attempts=n_att, n_ok=n_ok,
                         n_oom=n_oom, detail=detail))

    # Declared cells that never reached Elasticsearch at all have no group above.
    seen = {(r['profiler_label'], r['gpu_label'], r['image_label']) for r in rows}
    for key, (status, detail) in DECLARED_STATUS.items():
        if key not in seen:
            rows.append(dict(profiler_label=key[0], gpu_label=key[1], image_label=key[2],
                             status=status, n_attempts=0, n_ok=0, n_oom=0, detail=detail))

    out = pd.DataFrame(rows).sort_values(['profiler_label', 'gpu_label', 'image_label'])
    out.to_csv(out_csv, index=False)

    print(f'cells with at least one attempt : {len(out)}')
    for st, n in out['status'].value_counts().items():
        print(f'  {st:<9}: {n}')
    print(f'\nwritten: {out_csv}')
    return out


if __name__ == '__main__':
    build()
