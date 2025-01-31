import hashlib
import hmac
import os

from astropy.io import fits
from sqlalchemy import create_engine, text, exc

from gpuphot_worker.utils import logger




def __generate_connection_string():
    """
    Generate a PostgreSQL connection string using environment variables.

    :return: PostgreSQL connection string.
    :rtype: str
    """

    # Get database connection parameters from environment variables
    db_name = os.getenv('POSTGRES_DB', 'GPUPhotDB')
    user = os.getenv('POSTGRES_USER', 'admin')
    password = os.getenv('POSTGRES_PASSWORD', 'gpuphot')
    host = 'postgres'
    port = '5432'

    # Create the connection string
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db_name}'

    return connection_string


def insert_dataframe_to_postgres(df, tbl_name, unique_col='id'):
    """
    Insert a DataFrame into a PostgreSQL table, replacing existing records.

    :param df: DataFrame to insert.
    :type df: pandas.DataFrame
    :param tbl_name: Name of the target table.
    :type tbl_name: str
    :param unique_col: Name of the column with unique values.
    :type unique_col: str
    :return: True if successful, False otherwise.
    :rtype: bool
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
                delete_query = text(f"DELETE FROM {tbl_name} WHERE {unique_col} IN :values")
                connection.execute(delete_query, {"values": tuple(unique_values)})

                # Insert the DataFrame into the PostgreSQL table in chunks
                chunk_size = 5000  # Adjust based on your needs
                for start in range(0, len(df), chunk_size):
                    end = start + chunk_size
                    df.iloc[start:end].to_sql(tbl_name, con=connection, if_exists='append', index=False)

        return True

    except exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error: {str(e)}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

    return False


def queryStrAdd(query: str, toAdd: str) -> str:
    """
    Add a string value to an SQL query.

    :param query: Existing SQL query.
    :type query: str
    :param toAdd: String to add to the query.
    :type toAdd: str
    :return: Updated SQL query.
    :rtype: str
    """
    return query + "'" + toAdd + "', "


def queryAdd(query: str, toAdd) -> str:
    """
    Add a non-string value to an SQL query.

    :param query: Existing SQL query.
    :type query: str
    :param toAdd: Value to add to the query.
    :type toAdd: Any
    :return: Updated SQL query.
    :rtype: str
    """
    return query + str(toAdd) + ", "

# Takes a FITS header and formats it into a PSQL-compatible HStore fragment.
def headerToHstore(header):
    """
    Convert a FITS header to a PostgreSQL HStore-compatible string.

    :param header: FITS header.
    :type header: astropy.io.fits.Header
    :return: HStore-compatible string.
    :rtype: str
    """
    fragment = ""
    for key, value in header.items():
        if key != "COMMENT" and not isinstance(value, fits.header._HeaderCommentaryCards):
            # Convertir todo a string y escapar las comillas dobles
            key_str = str(key).replace('"', '\\"')
            value_str = str(value).replace('"', '\\"')
            fragment += f'"{key_str}" => "{value_str}", '
    return fragment[:-2]


def insert_header(header, file_path):
    """
    Generate an SQL query to insert or update a FITS header in the database.

    :param header: FITS header.
    :type header: astropy.io.fits.Header
    :param file_path: Path of the FITS file.
    :type file_path: str
    :return: SQL query string.
    :rtype: str
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
    Insert or update image statistics in the database.

    :param gpuphotid: Unique identifier for the image.
    :type gpuphotid: str
    :param file_path: Path of the image file.
    :type file_path: str
    :param header: FITS header of the image.
    :type header: astropy.io.fits.Header
    :param delete_prev: Whether to delete previous entries for this image.
    :type delete_prev: bool
    :return: True if successful, False otherwise.
    :rtype: bool
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
    Generate a unique identifier for a processed file.

    :param process_file: Path of the processed file.
    :type process_file: str
    :return: Unique identifier.
    :rtype: str
    """
    h = hmac.new('GPUPHOT_KEY'.encode(), process_file.encode(), hashlib.sha1)
    return str(h.hexdigest())
