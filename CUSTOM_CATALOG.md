# Guide for Custom Catalog Integration in GPUPhot

**Version:** 1.0
**Date:** 2024-08-01

This document provides a detailed technical guide for developers who want to integrate a local or custom catalog system with `GPUPhot` using the `custom_vizier_search_func` interface.

## 1. Executive Summary

`GPUPhot` offers an extension mechanism to replace its built-in Vizier client with a custom search function. This allows users to query private data sources, such as a local PostgreSQL database, an internal REST API, or any other astronomical data source, as long as the interface contract is respected.

The integration is controlled by two key parameters that can be passed to the `process_image` method:

- `custom_vizier_search_func` (callable): A Python function that implements the search logic.
- `custom_vizier_timeout` (int): A timeout in seconds for the execution of this function.

## 2. Custom Function Contract

For the integration to work correctly, the custom function **must** adhere to the following signature and behavior:

```python
from astropy.coordinates import SkyCoord
import pandas as pd
from typing import Optional, List, Dict

def my_search_function(
    coocenter: SkyCoord,
    catalog: str,
    radius: float,
    mag_limit: float,
    ref_filter: str,
    row_limit: int,
    expected_columns: Optional[List[str]] = None,
    timeout: int = 400,
) -> Optional[pd.DataFrame]:
    """
    Performs a conical search on a custom data source.

    Args:
        coocenter (SkyCoord): Coordinates of the search center.
        catalog (str): Catalog identifier (e.g., 'II/349/ps1').
        radius (float): Search radius in degrees.
        mag_limit (float): Upper magnitude limit for the objects.
        ref_filter (str): Magnitude filter/band to use.
        row_limit (int): Maximum number of rows to return.
        expected_columns (Optional[List[str]]): A list of columns that GPUPhot expects.
                                                 The function can use this to optimize
                                                 the query and return only what is necessary.
        timeout (int): Query timeout in seconds.

    Returns:
        A pandas.DataFrame with the results or None in case of an error.
    """
    # ... implementation here ...
```

### Expected Behavior:

1.  **Input:** The function will receive the described parameters, which define a standard conical search.
2.  **Successful Output:** It must return a `pandas.DataFrame` containing the catalog data. **It is crucial that the columns of this DataFrame use the canonical names that `GPUPhot` expects.** (See Section 4).
3.  **Failure or No Results:** In case of an error (e.g., timeout, connection failure, catalog not found) or if the search yields no results, the function should return `None` or an empty DataFrame. `GPUPhot` is designed to handle this case and, if possible, will fall back to its built-in Vizier client.
4.  **Columns:** If the returned DataFrame does not contain the `expected_columns`, `GPUPhot` will also attempt a fallback.

## 3. Implementation Example: PostgreSQL Connection with Q3C

Below is a robust and secure example of how to connect `GPUPhot` to a PostgreSQL database that uses the `q3c` extension for spatial indexing.

### 3.1. Configuration File (`CATALOG_CONFIG`)

It is good practice to externalize the specific configuration for each catalog. This facilitates maintenance and extensibility.

```python
# catalog_config.py

CATALOG_CONFIG: Dict[str, Dict] = {
    "II/349/ps1": {
        "table": "panstarrs_dr1",
        "ra_col": "raj2000",
        "dec_col": "dej2000",
        "column_translation": {
            "objid": "objID",
            "raj2000": "RAJ2000",
            "dej2000": "DEJ2000",
            "gmag": "gmag",
            "rmag": "rmag",
            "imag": "imag",
            "zmag": "zmag",
            "e_gmag": "e_gmag",
            "e_rmag": "e_rmag",
            "e_imag": "e_imag",
            "e_zmag": "e_zmag",
        },
    },
    "I/355/gaiadr3": {
        "table": "gaia_dr3",
        "ra_col": "raj2000",
        "dec_col": "dej2000",
        "column_translation": {
            "source": "Source",
            "raj2000": "RAJ2000",
            "dej2000": "DEJ2000",
            "gmag": "Gmag",
            "bpmag": "BPmag",
            "rpmag": "RPmag",
        },
    },
    # Add more catalogs here...
}
```

**Configuration Keys:**

-   `table`: Name of the table in the database.
-   `ra_col`, `dec_col`: Names of the coordinate columns in the table.
-   `column_translation`: A dictionary that maps database column names (keys) to `GPUPhot`'s canonical column names (values).

### 3.2. Connection Pool and Search Logic

To avoid the overhead of creating a new database connection for each query, using a **connection pool** is highly recommended.

```python
# custom_search_logic.py

import os
import time
import logging
import pandas as pd
import psycopg2
import psycopg2.pool
from psycopg2.extensions import AsIs
from astropy.coordinates import SkyCoord
from typing import Optional, List

from .catalog_config import CATALOG_CONFIG

logger = logging.getLogger(__name__)

# --- Connection Pool Initialization ---
try:
    db_params = {
        "database": os.getenv("DB_CATALOG_NAME", "catalogs_db"),
        "host": os.getenv("DB_CATALOG_HOST", "localhost"),
        "user": os.getenv("DB_CATALOG_USER", "readonly_user"),
        "password": os.getenv("DB_CATALOG_PASS", "password"),
        "port": os.getenv("DB_CATALOG_PORT", 5432),
    }
    connection_pool = psycopg2.pool.SimpleConnectionPool(1, 10, **db_params)
    logger.info("Catalog database connection pool created successfully.")
except psycopg2.Error as e:
    logger.critical(f"Could not create the connection pool: {e}")
    connection_pool = None

# --- Search Function Implementation ---

def custom_vizier_catalog(
    coocenter: SkyCoord,
    catalog: str,
    radius: float,
    mag_limit: float,
    ref_filter: str,
    row_limit: int,
    expected_columns: Optional[List[str]] = None,
    timeout: int = 400,
) -> Optional[pd.DataFrame]:
    """
    PostgreSQL search implementation with q3c, connection pooling, and timeouts.
    """
    if not connection_pool:
        logger.error("The connection pool is not available.")
        return None

    start_time = time.perf_counter()

    # 1. Validate configuration
    cfg = CATALOG_CONFIG.get(catalog)
    if not cfg:
        logger.warning(f"Catalog '{catalog}' is not supported in the local configuration.")
        return None

    # 2. Build the query safely
    db_cols = list(cfg["column_translation"].keys())
    select_clause = ", ".join(f'"{col}"' for col in db_cols)

    # Determine the magnitude column from the canonical ref_filter
    inverted_translation = {v: k for k, v in cfg["column_translation"].items()}
    if ref_filter not in inverted_translation:
        logger.warning(f"Filter '{ref_filter}' cannot be translated for catalog '{catalog}'.")
        return None
    mag_column_db = inverted_translation[ref_filter]

    query_template = """
        SELECT {select_clause}
        FROM {table_name}
        WHERE q3c_radial_query({ra_col}, {dec_col}, %(ra)s, %(dec)s, %(radius)s)
          AND {mag_col} < %(mag_limit)s
        LIMIT %(row_limit)s;
    """

    params = {
        "ra": coocenter.ra.deg,
        "dec": coocenter.dec.deg,
        "radius": radius,
        "mag_limit": mag_limit,
        "row_limit": row_limit if row_limit > 0 else None,
    }

    conn = None
    try:
        # 3. Get a connection from the pool and execute
        conn = connection_pool.getconn()
        with conn.cursor() as cur:
            # Apply per-transaction timeout
            cur.execute(f"SET statement_timeout = {int(timeout * 1000)}")

            final_query = cur.mogrify(
                query_template.format(
                    select_clause=AsIs(select_clause),
                    table_name=AsIs(f'"{cfg["table"]}"'),
                    ra_col=AsIs(f'"{cfg["ra_col"]}"'),
                    dec_col=AsIs(f'"{cfg["dec_col"]}"'),
                    mag_col=AsIs(f'"{mag_column_db}"'),
                ),
                params,
            )
            cur.execute(final_query)
            results = cur.fetchall()
            df = pd.DataFrame(results, columns=db_cols)

    except (psycopg2.Error, ConnectionError) as e:
        logger.error(f"Database error for catalog '{catalog}': {e}")
        return None  # Return None on error
    finally:
        if conn:
            connection_pool.putconn(conn)  # Always return the connection to the pool!

    # 4. Post-processing: translate columns and clean up
    df = df.rename(columns=cfg["column_translation"])

    # Remove duplicates based on all columns except ID if it exists
    if 'ID' in df.columns:
        cols_for_dedup = [c for c in df.columns if c != 'ID']
    else:
        cols_for_dedup = list(df.columns)
    df.drop_duplicates(subset=cols_for_dedup, keep='first', inplace=True)

    duration = time.perf_counter() - start_time
    logger.info(f"Search on '{catalog}' returned {len(df)} rows in {duration:.2f}s.")

    return df
```

## 4. Canonical Column Names and Supported Catalogs

To ensure compatibility, the `DataFrame` returned by the custom function must use the following column names (where applicable).

| Catalog        | Supported `ref_filter`s | Expected Canonical Columns (`expected_columns`)                                                                                             |
| :-------------- | :---------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **`II/379`**    | `g`, `r`, `i`, `z`, `v` | `['ID', 'RA', 'DEC', 'gPSF', 'rPSF', 'iPSF', 'zPSF', 'vPSF', 'e_gPSF', 'e_rPSF', 'e_iPSF', 'e_zPSF', 'e_vPSF']`                               |
| **`I/355/gaiadr3`** | `G`, `BP`, `RP`         | `['Source', 'RA', 'DEC', 'Gmag', 'BPmag', 'RPmag', 'e_Gmag', 'e_BPmag', 'e_RPmag']`                                                            |
| **`II/349/ps1`**  | `g`, `r`, `i`, `z`, `y` | `['objID', 'RA', 'DEC', 'gmag', 'rmag', 'imag', 'zmag', 'ymag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_ymag']`                               |
| **`I/353/gsc242`**  | `B`, `V`, `R`, `I`, `N` | `['GSCID', 'RA', 'DEC', 'Bmag', 'Vmag', 'Rmag', 'Imag', 'Nmag']`                                                                                |

**Important Note:** The `custom_vizier_catalog` function must be able to handle the `ref_filter` to select the correct magnitude column in the `WHERE` clause of the SQL query.

## 5. Integration with `process_image`

Finally, to use your custom function, it's good practice to group the advanced parameters into a dictionary and pass them to `process_image`. This makes the call cleaner and easier to manage.

```python
from gpuphot import get_processor
from gpuphot_worker.utils import open_image_file
from .custom_search_logic import custom_vizier_catalog  # Import your function

# 1. Create the processor
processor = get_processor(instrument_name='my_instrument')

# 2. Load the image
imdata, imheader = open_image_file('path/to/image.fits')

# 3. Define the custom catalog parameters in a dictionary
custom_catalog_kwargs = {
    'custom_vizier_search_func': custom_vizier_catalog,
    'custom_vizier_timeout': 120  # 2-minute timeout
}

# 4. Process the image, unpacking the dictionary as keyword arguments
phot_df, hwcs = processor.process_image(
    imdata,
    imheader,
    **custom_catalog_kwargs
)

# 5. Use the results
if phot_df is not None:
    print(f"Photometry completed. Found {len(phot_df)} stars.")
else:
    print("Image processing failed.")
```

---
End of Document.
