import os

import psycopg2
import pandas as pd

CATALOG_HOST = os.environ.get('GPUPHOT_CATALOG_HOST', '10.0.210.30')
CATALOG_PORT = int(os.environ.get('GPUPHOT_CATALOG_PORT', '5434'))
CATALOG_DB = 'catalogs_db'
CATALOG_USER = 'read_only'
CATALOG_PASSWORD = 'read_only'

_CATALOG_CONFIG = {
    'II/349/ps1': {
        'table': 'panstarrs_dr1',
        'mag_column': 'gmag',
        'ra_col': 'raj2000',
        'dec_col': 'dej2000',
        'column_translation': {
            'objid': 'objID', 'raj2000': 'RAJ2000', 'dej2000': 'DEJ2000',
            'gmag': 'gmag', 'rmag': 'rmag', 'imag': 'imag', 'zmag': 'zmag',
            'e_gmag': 'e_gmag', 'e_rmag': 'e_rmag', 'e_imag': 'e_imag', 'e_zmag': 'e_zmag',
        },
    },
    'I/355/gaiadr3': {
        'table': 'gaia_dr3',
        'mag_column': None,  # uses ref_filter directly
        'ra_col': 'raj2000',
        'dec_col': 'dej2000',
        'column_translation': {
            'source': 'Source', 'raj2000': 'RAJ2000', 'dej2000': 'DEJ2000',
            'bpmag': 'BPmag', 'bp-rp': 'BP-RP', 'fbp': 'FBP', 'e_fbp': 'e_FBP',
        },
    },
    'I/353/gsc242': {
        'table': 'gsc2',
        'mag_column': None,
        'ra_col': 'ra_icrs',
        'dec_col': 'de_icrs',
        'column_translation': {
            'gsc2': 'GSC2', 'ra_icrs': 'RA_ICRS', 'de_icrs': 'DE_ICRS', 'umag': 'umag',
        },
    },
    'II/379': {
        'table': 'skymapper',
        'mag_column': None,
        'ra_col': 'raicrs',
        'dec_col': 'deicrs',
        'column_translation': {
            'smss': 'SMSS', 'raicrs': 'RAICRS', 'deicrs': 'DEICRS',
            'upsf': 'uPSF', 'gpsf': 'gPSF', 'rpsf': 'rPSF', 'ipsf': 'iPSF', 'zpsf': 'zPSF',
        },
    },
}


def custom_vizier_catalog(coocenter, catalog, radius, mag_limit, ref_filter,
                           row_limit, expected_columns=None, timeout=450, **kwargs):
    if catalog not in _CATALOG_CONFIG:
        raise ValueError(f"Unsupported catalog for local query: {catalog}")

    config = _CATALOG_CONFIG[catalog]
    table = config['table']
    mag_column = config['mag_column'] or ref_filter.lower()
    ra_col = config['ra_col']
    dec_col = config['dec_col']
    translation = config['column_translation']

    ra = coocenter.ra.deg
    dec = coocenter.dec.deg
    mag_limit = round(float(mag_limit), 1)

    query = (
        f"SELECT * FROM {table} "
        f"WHERE q3c_radial_query({ra_col}, {dec_col}, {ra}, {dec}, {radius}) "
        f"AND {mag_column} < {mag_limit}"
    )
    if row_limit and row_limit > 0:
        query += f" LIMIT {row_limit}"

    conn = psycopg2.connect(
        database=CATALOG_DB, host=CATALOG_HOST, port=CATALOG_PORT,
        user=CATALOG_USER, password=CATALOG_PASSWORD,
        connect_timeout=min(timeout, 30),
    )
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=cols)
    df = df.rename(columns=translation)

    if expected_columns:
        present = [c for c in expected_columns if c in df.columns]
        df = df[present]

    return df
