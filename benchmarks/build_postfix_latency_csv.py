#!/usr/bin/env python3
"""Build benchmark_latency_postfix.csv, the canonical file of the post-fix campaign.

WHY THIS EXISTS, and why it is not done by hand. The August campaign
(benchmark_latency.csv) is the evidence behind everything the paper has
published so far and IS NOT TOUCHED. The campaign that followed the
allocator fix (commit 1a23734, 2026-08-31 08:28) arrives in pieces, one
block per machine and arm, and each block comes in as its own
`data/measure_*.csv` file. This script merges them into a single file that
the generators read when USE_POSTFIX_CAMPAIGN is active.

A manual merge already destroyed the Jetson data once, so here:
  - blocks are NEVER edited: they are read and concatenated;
  - the minimal schema required by load_benchmark() and the session rule is enforced;
  - it checks for no rows predating the fix, which would belong to the other campaign;
  - it warns about exact duplicates and overlaps (same machine, arm, image and timestamp);
  - it prints coverage per card and arm, which decides whether a table can switch over.

Usage:
    python build_postfix_latency_csv.py            # build and summarize
    python build_postfix_latency_csv.py --dry-run  # report only, do not write
"""

import os
import sys
import glob
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(DATA, 'benchmark_latency_postfix.csv')

# Seed: the 31-Aug-to-3-Sep campaign already collected. It is one more block,
# not a special case.
SEED = os.path.join(DATA, 'benchmark_latency_allocator_fix.csv')
# New blocks. Named on purpose with the date INSIDE the file, not in the filename.
BLOCK_GLOB = os.path.join(DATA, 'measure_postfix_*.csv')

FIX_COMMIT_TS = pd.Timestamp('2026-08-31T08:28:00Z')

# What load_benchmark() and the session rule absolutely need.
REQUIRED = ['machine', 'gpu_name', 'profiler_label', 'image_label',
            'execution_time', 'timestamp']
# What the tables additionally use for filtering. If missing, filled with the
# campaign's value.
DEFAULTS = {'catalog_backend': 'local', 'remeasure_round': 'postfix'}


# Column synonyms that have shown up across blocks. They are normalized to the
# campaign's name because that is what the generators read; otherwise a useful
# column ends up split into two empty halves and nobody notices until they
# need it to diagnose something.
COLUMN_ALIASES = {'gpu_temp_c': 'gpu_temp'}


def _read(path):
    df = pd.read_csv(path, comment='#', low_memory=False)
    for alias, canon in COLUMN_ALIASES.items():
        if alias in df.columns:
            if canon in df.columns:
                df[canon] = df[canon].where(df[canon].notna(), df[alias])
            else:
                df[canon] = df[alias]
            df = df.drop(columns=[alias])
            print(f'  nota: {os.path.basename(path)} traia {alias!r}; normalizado a {canon!r}')
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f'{os.path.basename(path)}: faltan columnas obligatorias {missing}')
    for col, val in DEFAULTS.items():
        if col not in df.columns:
            print(f'  aviso: {os.path.basename(path)} no trae {col!r}; se rellena con {val!r}')
            df[col] = val
    df['source_file'] = os.path.basename(path)
    return df


def main():
    dry = '--dry-run' in sys.argv
    paths = ([SEED] if os.path.exists(SEED) else []) + sorted(glob.glob(BLOCK_GLOB))
    if not paths:
        raise SystemExit(f'no hay bloques que unir (ni {os.path.basename(SEED)} ni {BLOCK_GLOB})')

    print('Bloques:')
    parts = []
    for p in paths:
        d = _read(p)
        print(f'  {os.path.basename(p):<52} {len(d):>6} filas')
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    # What tells a post-fix row apart: the RUN LABEL, not the clock. The
    # commit timestamp (08:28) is when the fix landed in git; the containers
    # were patched earlier, and in fact 520 legitimate rows from the fix1
    # campaign predate that time. Filtering by time would have dropped them.
    # It is filtered by label, and time only serves as a warning check.
    ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce', format='mixed')
    is_post = (df['remeasure_round'].astype(str).str.startswith(('fix', 'postfix'))
               | df['environment'].astype(str).str.contains('fix', na=False))
    if (~is_post).any():
        print(f'\n  AVISO: {int((~is_post).sum())} filas sin etiqueta de campaña post-fix '
              '(remeasure_round fix*/postfix* ni environment con "fix"). Se DESCARTAN.')
        for f, n in df.loc[~is_post, 'source_file'].value_counts().items():
            print(f'         {n:>6} en {f}')
        df, ts, is_post = (df[is_post].reset_index(drop=True),
                           ts[is_post].reset_index(drop=True), None)
    early = ts < FIX_COMMIT_TS
    if early.any():
        print(f'\n  nota: {int(early.sum())} filas etiquetadas post-fix son anteriores al sello '
              f'del commit ({FIX_COMMIT_TS}); es normal, el contenedor se parchea antes de '
              'commitear. Se CONSERVAN, la etiqueta manda sobre el reloj.')

    key = ['machine', 'profiler_label', 'image_label', 'timestamp']
    dup = df.duplicated(subset=key, keep='first')
    if dup.any():
        print(f'\n  AVISO: {int(dup.sum())} filas duplicadas por {key}; se conserva la primera.')
        df = df[~dup].reset_index(drop=True)

    if dry:
        print(f'\n--dry-run: no se escribe {OUT}; la cobertura se calcula sobre el fichero '
              'existente si lo hay.')
    else:
        df.to_csv(OUT, index=False)
        print(f'\nEscrito {OUT}: {len(df)} filas de {df["source_file"].nunique()} bloques.')
        print('Recuerda: la regla de sesiones contaminadas vive en '
              'generate_manuscript_tables.py (CONTAMINATED_SESSIONS) y se aplica sola al leer.')
    if os.path.exists(OUT):
        _report_coverage(df)


def _report_coverage(df):
    """Cobertura POST-FILTROS y CONTRA LA CAMPAÑA, que es lo único que decide si algo conmuta.

    Dos cosas que la versión anterior hacía mal y confundieron a quien mide.
    Primera: contaba filas crudas, sin el descarte de calentamiento, el filtro MAD ni la
    exclusion of contaminated windows, which is what the generators apply; a cell with
    una sola repetición aparecía como medida y luego salía vacía en la tabla.
    Segunda: comparaba contra 19 imágenes fijas, que es el banco entero y NO el complemento de
    the consumer cards reach fewer cells than the datacenter ones, because the largest
    demás no les caben en memoria.  Contra 19, una columna completa parecía coja para siempre.
    El listón correcto es lo que esa tarjeta y ese brazo tienen EN LA CAMPAÑA.
    """
    import generate_manuscript_tables as T
    print('\nCoverage, already with the generator filters applied, against the earlier campaign:')
    for lab in ('py312_cuml_adaptive', 'py38_baseline', 'py312_cuml_always'):
        try:
            camp = T.load_benchmark([lab])
            post = T.load_benchmark([lab], csv_path=OUT)
        except Exception as e:
            print(f'  {lab}: no evaluable ({e})')
            continue
        c = camp.groupby('gpu_label')['image_label'].nunique()
        p = post.groupby('gpu_label')['image_label'].nunique()
        print(f'\n  {lab}')
        for gpu in sorted(set(c.index) | set(p.index)):
            a, b = int(c.get(gpu, 0)), int(p.get(gpu, 0))
            if a == 0 and b == 0:
                continue
            state = ('COMPLETA' if b >= a > 0 else
                     'sin dato post-fix' if b == 0 else f'faltan {a - b} de {a}')
            print(f'    {gpu:<22} campaña {a:>2}  post-fix {b:>2}   {state}')
    print('\nUna columna conmuta cuando dice COMPLETA.  "sin dato post-fix" no siempre bloquea:')
    print('the edge modules have no cuML (the import fails on that architecture), so there is no')
    print('asignador y el valor de campaña ya es el de la configuración publicada.')


if __name__ == '__main__':
    main()
