# GPUPhot Database Reference

GPUPhot persists every processed image and its photometric results in a
**PostgreSQL** database extended with the **Q3C** plugin for fast astronomical
coordinate queries.  This document covers the full schema, how records are
created, and all available query helpers.

---

## Table of Contents

- [1. Connection & Environment Variables](#1-connection--environment-variables)
- [2. Schema Overview](#2-schema-overview)
  - [2.1 imastats — Image metadata](#21-imastats--image-metadata)
  - [2.2 imaphot — Photometry results](#22-imaphot--photometry-results)
  - [2.3 Relationship between tables](#23-relationship-between-tables)
- [3. Image Identifier (GPUPHOTI)](#3-image-identifier-gpuphoti)
- [4. Python Query API](#4-python-query-api)
  - [4.1 Cone search by coordinates](#41-cone-search-by-coordinates)
  - [4.2 Date range search](#42-date-range-search)
  - [4.3 Search by filename](#43-search-by-filename)
  - [4.4 Bulk search by filename list](#44-bulk-search-by-filename-list)
  - [4.5 Transient candidates](#45-transient-candidates)
  - [4.6 Inspect schema at runtime](#46-inspect-schema-at-runtime)
- [5. Direct SQL Examples](#5-direct-sql-examples)
- [6. Data Lifecycle](#6-data-lifecycle)

---

## 1. Connection & Environment Variables

All database functions read the following variables from the environment
(or `.env` when running inside Docker Compose):

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `postgres` | Hostname of the PostgreSQL service |
| `POSTGRES_PORT` | `5432` | Port used to connect. Inside `docker-compose.yml`, the worker/beat containers get this hardcoded to `5432` (the port `postgres` always listens on within the compose network); it only comes from `.env` when these functions run outside Docker Compose. See `POSTGRES_HOST_PORT` in `DOCKER.md` for the separate, host-side exposed port. |
| `POSTGRES_DB` | `GPUPhotDB` | Database name |
| `POSTGRES_USER` | `admin` | Write user (used by workers for inserts) |
| `POSTGRES_PASSWORD` | *(set in .env)* | Write user password |
| `POSTGRES_USER_READ` | `read_only` | Read-only user (used by search helpers) |
| `POSTGRES_PASSWORD_READ` | `read_only` | Read-only user password |

The read-only credentials (`POSTGRES_USER_READ` / `POSTGRES_PASSWORD_READ`)
are used by `database_search_utils` so that query scripts never need write
access to the database.

---

## 2. Schema Overview

### 2.1 `imastats` — Image metadata

One row per processed FITS image. Written by the worker immediately after
the pipeline completes for an image.

| Column | Type | FITS keyword | Description |
|---|---|---|---|
| `id` | text (PK) | `GPUPHOTI` | Unique image identifier (HMAC-SHA1 of file path — see §3) |
| `file_path` | text | — | Absolute path of the FITS file on the worker host |
| `naxis1` | integer | `NAXIS1` | Image width (pixels) |
| `naxis2` | integer | `NAXIS2` | Image height (pixels) |
| `telescop` | text | `TELESCOP` | Telescope name |
| `instrume` | text | `INSTRUME` | Instrument name |
| `camera` | text | `CAMERA` | Camera model |
| `filter` | text | `FILTER` | Filter used for the observation |
| `date_obs` | timestamp | `DATE-OBS` | Observation date/time (UTC) |
| `exptime` | float | `EXPTIME` | Exposure time (seconds) |
| `object` | text | `OBJECT` | Target object name |
| `ra` | float | `RA` | Field centre Right Ascension (degrees) |
| `dec` | float | `DEC` | Field centre Declination (degrees) |
| `fwhm` | float | `FWHM` | Measured PSF FWHM (arcsec) |
| `maglim` | float | `MAGLIM` | 5σ limiting magnitude of the image |
| `header` | hstore | — | Full FITS header stored as a PostgreSQL key-value map |

The `ra`/`dec` columns are indexed with **Q3C** for sub-second cone searches
over millions of rows.

The `header` hstore column gives access to every keyword in the original FITS
header without needing to reload the file:

```sql
-- Read a specific keyword from the stored header
SELECT id, header -> 'AIRMASS' AS airmass FROM imastats LIMIT 10;
```

Inserts use `ON CONFLICT (id) DO UPDATE`, so reprocessing an image updates
the existing row rather than creating a duplicate.

---

### 2.2 `imaphot` — Photometry results

One row per detected source per image. Written in bulk by the worker after
all sources in the image have been measured and calibrated.

| Column | Type | Description |
|---|---|---|
| `id` | text (FK → `imastats.id`) | Image identifier linking this source to its image |
| `ra` | float | Source Right Ascension (degrees) |
| `dec` | float | Source Declination (degrees) |
| `flux` | float | Calibrated flux |
| `dflux` | float | Flux uncertainty (1σ) |
| `trans` | boolean | `true` if the source was flagged as a transient candidate |

Inserts first delete any existing rows with the same `id` to ensure
idempotency — reprocessing an image replaces all its photometry rows.

---

### 2.3 Relationship between tables

```
imastats (1) ──< imaphot (N)
    id  ──────────  id
```

A single image (`imastats` row) produces N photometry rows in `imaphot`, one
per detected source.  Join them on `id` to combine image-level metadata with
source-level measurements.

---

## 3. Image Identifier (GPUPHOTI)

Each image is assigned a unique identifier derived from its file path using
HMAC-SHA1:

```python
import hashlib, hmac

def generate_gpuphotid(file_path: str) -> str:
    h = hmac.new(b'GPUPHOT_KEY', file_path.encode(), hashlib.sha1)
    return h.hexdigest()
```

The same file path always produces the same ID, so reprocessing the same
file updates existing records rather than creating duplicates.  The ID is
also written back to the FITS header as the `GPUPHOTI` keyword.

---

## 4. Python Query API

All helpers live in `gpuphot_worker.database_search_utils` and return a
`pandas.DataFrame`.  They use the read-only database user.

```python
from gpuphot_worker.database_search_utils import (
    search_by_radec,
    search_by_date_range,
    search_by_filename,
    search_by_filenames,
    search_transients,
    search_transient_images,
    get_tables_and_columns,
)
```

---

### 4.1 Cone search by coordinates

```python
# All images whose field centre is within 1° of RA=83.82°, Dec=−5.39°
df = search_by_radec(ra=83.82, dec=-5.39, radius=1.0, table='imastats')

# All individual sources within 0.1° of the same position
df = search_by_radec(ra=83.82, dec=-5.39, radius=0.1, table='imaphot')
```

Both tables support Q3C cone searches via `q3c_radial_query`.
`radius` is in degrees.

---

### 4.2 Date range search

Returns a joined result with all `imastats` columns plus `flux` and `dflux`
from `imaphot` for images observed within the date window.

```python
df = search_by_date_range('2025-01-01', '2025-06-30')
print(df[['file_path', 'date_obs', 'filter', 'flux']].head())
```

---

### 4.3 Search by filename

Partial match — finds all images whose `file_path` contains the given string.

```python
df = search_by_filename('M42')          # any file with 'M42' in its path
df = search_by_filename('2025-03-15')   # all images from a given date
```

---

### 4.4 Bulk search by filename list

Retrieves `imaphot` + `imastats` rows for a list of specific file paths.
Useful when reprocessing a known set of files.

```python
files = ['/data/night1/img001.fits', '/data/night1/img002.fits']
df = search_by_filenames(files)
```

---

### 4.5 Transient candidates

```python
# All individual transient sources detected after a given date
df = search_transients(date_after='2025-06-01')
# Columns: imaphot.* + file_path, date_obs, filter (from imastats)

# Summary view: one row per transient source with image context
df = search_transient_images(date_after='2025-06-01')
# Columns: id, flux, dflux, file_path, date_obs, filter
```

A source has `trans = True` when the pipeline detects it as a candidate
transient (not present in the reference catalog within the matching radius).

---

### 4.6 Inspect schema at runtime

```python
schema = get_tables_and_columns()
for table, columns in schema.items():
    print(f"\n{table}")
    for col_name, col_type in columns:
        print(f"  {col_name}: {col_type}")
```

---

## 5. Direct SQL Examples

Connect to the database directly (e.g. from JupyterLab inside the Docker
environment) using the `psycopg2` or `sqlalchemy` drivers:

```python
import os
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER_READ', 'read_only')}"
    f":{os.getenv('POSTGRES_PASSWORD_READ', 'read_only')}"
    f"@{os.getenv('POSTGRES_HOST', 'postgres')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'GPUPhotDB')}"
)
```

**Count processed images per filter:**
```sql
SELECT filter, COUNT(*) AS n_images
FROM imastats
GROUP BY filter
ORDER BY n_images DESC;
```

**Average limiting magnitude per night:**
```sql
SELECT DATE(date_obs) AS night, AVG(maglim) AS avg_maglim
FROM imastats
GROUP BY night
ORDER BY night;
```

**All sources brighter than magnitude 16 (flux > threshold) with coordinates:**
```sql
SELECT p.ra, p.dec, p.flux, p.dflux, i.date_obs, i.filter
FROM imaphot p
JOIN imastats i ON p.id = i.id
WHERE p.flux > 10000
ORDER BY p.flux DESC
LIMIT 100;
```

**Cone search in raw SQL (Q3C):**
```sql
SELECT p.ra, p.dec, p.flux, i.date_obs
FROM imaphot p
JOIN imastats i ON p.id = i.id
WHERE q3c_radial_query(p.ra, p.dec, 83.82, -5.39, 0.5);
-- Arguments: ra_col, dec_col, centre_ra, centre_dec, radius_degrees
```

**Retrieve any FITS keyword stored in the header hstore:**
```sql
SELECT id, header -> 'AIRMASS' AS airmass, header -> 'GAIN' AS gain
FROM imastats
WHERE date_obs > '2025-01-01';
```

---

## 6. Data Lifecycle

```
FITS image
    │
    ▼
Worker pipeline
    ├─ generate_gpuphotid(file_path)  →  unique id (HMAC-SHA1)
    ├─ populate_ima_stats()           →  upsert row in imastats
    └─ insert_dataframe_to_postgres() →  delete + insert rows in imaphot
```

- **Reprocessing** the same file path produces the same `id`, so both tables
  are updated in place — no duplicate rows are created.
- **Deletes** in `imaphot` before re-insert are transactional; a failed
  re-insert leaves the table empty for that image rather than corrupt.
- The `header` hstore is always overwritten on reprocessing, so it reflects
  the most recent version of the FITS header.
