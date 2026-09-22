# SPDX-License-Identifier: MIT
"""
Utilities for inserting processed image results into PostgreSQL.

This module contains helper functions used by worker tasks to persist
photometry results and image statistics into PostgreSQL using SQLAlchemy.

Notes
-----
- Environment variables POSTGRES_* are used to build the connection string.
- The functions intentionally use simple SQL snippets for compatibility with
  existing database schemas. They focus on clarity and traceable error logging.
"""

import hashlib
import hmac
import os

from astropy.io import fits
from sqlalchemy import create_engine, text, exc

from gpuphot_worker.utils import logger




def __generate_connection_string():
    """
    Generate a PostgreSQL connection string using environment variables.

    Returns
    -------
    str
        PostgreSQL connection string.
    """

    # Get database connection parameters from environment variables
    db_name = os.getenv('POSTGRES_DB', 'GPUPhotDB')
    user = os.getenv('POSTGRES_USER', 'admin')
    password = os.getenv('POSTGRES_PASSWORD', 'gpuphot')
    host = os.getenv('POSTGRES_HOST', 'postgres')
    port = os.getenv('POSTGRES_PORT', '5432')

    # Create the connection string
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db_name}'

    return connection_string


def insert_dataframe_to_postgres(df, unique_col='id'):
    """
    Insert a pandas DataFrame into the `imaphot` PostgreSQL table.

    The function deletes any existing rows with matching values in `unique_col`
    and then inserts the DataFrame in chunks. Uses SQLAlchemy engine for
    transactional safety.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to insert.
    unique_col : str, optional
        Name of the column with unique values (default is 'id').

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Get database connection parameters from environment variables
        connection_string = __generate_connection_string()

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Proceed only if DataFrame is not empty
        if not df.empty:
            unique_values = df[unique_col].unique()

            with engine.begin() as connection:  # Use transaction context manager
                # Delete existing records in one go
                delete_query = text(f"DELETE FROM {'imaphot'} WHERE {unique_col} IN :values")
                connection.execute(delete_query, {"values": tuple(unique_values)})

                # Insert the DataFrame into the PostgreSQL table in chunks
                chunk_size = 5000  # Adjust based on your needs
                for start in range(0, len(df), chunk_size):
                    end = start + chunk_size
                    df.iloc[start:end].to_sql('imaphot', con=connection, if_exists='append', index=False)

        return True

    except exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error: {str(e)}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

    return False


def queryStrAdd(query: str, toAdd: str) -> str:
    """
    Add a quoted string value to an SQL query fragment.

    Parameters
    ----------
    query : str
        Existing SQL query fragment.
    toAdd : str
        String value to add (will be single-quoted).

    Returns
    -------
    str
        Updated SQL query fragment.
    """
    return query + "'" + toAdd + "', "


def queryAdd(query: str, toAdd) -> str:
    """
    Add a non-string value to an SQL query fragment.

    Parameters
    ----------
    query : str
        Existing SQL query fragment.
    toAdd : Any
        Value to add; converted to string.

    Returns
    -------
    str
        Updated SQL query fragment.
    """
    return query + str(toAdd) + ", "

# Takes a FITS header and formats it into a PSQL-compatible HStore fragment.
def headerToHstore(header):
    """
    Convert a FITS header to a PostgreSQL HStore-compatible string.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header to convert.

    Returns
    -------
    str
        HStore-compatible string fragment (without enclosing braces).
    """
    fragment = ""
    for key, value in header.items():
        if key != "COMMENT" and not isinstance(value, fits.header._HeaderCommentaryCards):
            # Convert everything to string and escape double quotes
            key_str = str(key).replace('"', '\\"')
            value_str = str(value).replace('"', '\\"')
            fragment += f'"{key_str}" => "{value_str}", '
    return fragment[:-2]


def insert_header(header, file_path):
    """
    Generate an SQL query to insert or update a FITS header row in the `imastats` table.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header.
    file_path : str
        Path of the FITS file (stored in the table).

    Returns
    -------
    str
        SQL query string that performs an upsert (ON CONFLICT DO UPDATE).
    """
    query_parts = ["INSERT INTO imastats (id, file_path, "]

    columns = [
        "naxis1", "naxis2", "telescop", "instrume", "camera", "filter",
        "date_obs", "exptime", "object", "ra", "dec", "fwhm", "maglim", "header"
    ]
    query_parts.append(", ".join(columns))
    query_parts.append(") VALUES (")

    query_parts.append(f"'{header.get('GPUPHOTI', '')}', '{file_path}', ")

    keys = [
        "NAXIS1", "NAXIS2", "TELESCOP", "INSTRUME", "CAMERA", "FILTER",
        "DATE-OBS", "EXPTIME", "OBJECT", "RA", "DEC", "FWHM", "MAGLIM"
    ]

    for key in keys:
        value = header.get(key, 'NULL')
        if value == 'NULL' or value is None:
            query_parts.append("NULL, ")
        elif isinstance(value, (int, float)):
            query_parts.append(f"{value}, ")
        else:
            query_parts.append(f"'{str(value)}', ")

    hstore_fragment = headerToHstore(header)
    query_parts.append(f"'{hstore_fragment}'")

    query = ''.join(query_parts) + ") ON CONFLICT (id) DO UPDATE SET "

    update_parts = [f"{col} = EXCLUDED.{col}" for col in columns]
    query += ", ".join(update_parts)

    return query


def populate_ima_stats(gpuphotid, file_path, header, delete_prev=True):
    """
    Insert or update image statistics in the `imastats` table.

    Parameters
    ----------
    gpuphotid : str
        Unique identifier for the image.
    file_path : str
        Path of the image file to store in the table.
    header : astropy.io.fits.Header
        FITS header of the image.
    delete_prev : bool, optional
        Whether to delete previous entries for this image (default True).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """

    try:
        connection_string = __generate_connection_string()
        engine = create_engine(connection_string)

        with engine.begin() as connection:
            if delete_prev:
                delete_query = text(f"DELETE FROM imastats WHERE id = '{gpuphotid}'")
                connection.execute(delete_query)

            q = insert_header(header, file_path)
            connection.execute(text(q))
        return True
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    return False


def generate_gpuphotid(process_file):
    """
    Generate a unique identifier for a processed file using HMAC-SHA1.

    Parameters
    ----------
    process_file : str
        Path or string identifying the processed file.

    Returns
    -------
    str
        Hexadecimal HMAC-SHA1 digest.
    """
    h = hmac.new('GPUPHOT_KEY'.encode(), process_file.encode(), hashlib.sha1)
    return str(h.hexdigest())
