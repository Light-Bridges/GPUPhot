import os

import pandas as pd
import psycopg2
from psycopg2 import sql

from gpuphot.logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


def connect_to_db():
    """
    Establish a connection to the PostgreSQL database.

    :return: A database connection object or None if the connection fails.
    :rtype: psycopg2.extensions.connection or None
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB', 'GPUPhotDB'),
            user=os.getenv('POSTGRES_USER_READ', 'read_only'),
            password=os.getenv('POSTGRES_PASSWORD_READ', 'read_only'),
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        return conn
    except (Exception, psycopg2.Error) as error:
        print("Error connecting to PostgreSQL:", error)
        return None


def get_tables_and_columns():
    """
    Retrieve all tables and their columns from the database, including column data types.

    :return: A dictionary where keys are table names and values are lists of tuples (column_name, data_type).
    :rtype: dict
    """
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()

        # Query to get all tables and their columns with data types
        query = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """

        cur.execute(query)
        results = cur.fetchall()

        # Organize results into a dictionary
        tables_dict = {}
        for table_name, column_name, data_type in results:
            if table_name not in tables_dict:
                tables_dict[table_name] = []
            tables_dict[table_name].append((column_name, data_type))

        return tables_dict
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
        return {}
    finally:
        if conn:
            cur.close()
            conn.close()


def results_to_dataframe(results, column_names):
    """
    Convert query results into a pandas DataFrame.

    :param results: List of tuples containing query results.
    :type results: list
    :param column_names: List of column names.
    :type column_names: list
    :return: A pandas DataFrame.
    :rtype: pd.DataFrame
    """
    if results is None or len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results, columns=column_names)
    return df


def search_by_radec(ra, dec, radius, table='imastats'):
    """
    Search for records within a specified radius around given coordinates (RA, Dec).

    :param ra: Right Ascension (in degrees).
    :type ra: float
    :param dec: Declination (in degrees).
    :type dec: float
    :param radius: Search radius (in degrees).
    :type radius: float
    :param table: Table to search in ('imastats' or 'imaphot').
    :type table: str
    :return: A DataFrame containing the search results.
    :rtype: pd.DataFrame
    """
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()

        # Validate the selected table
        if table not in ['imastats', 'imaphot']:
            raise ValueError("Table must be 'imastats' or 'imaphot'.")

        query = sql.SQL("""
            SELECT * FROM {}
            WHERE q3c_radial_query(ra, dec, {}, {}, {})
        """).format(sql.Identifier(table), sql.Literal(ra), sql.Literal(dec), sql.Literal(radius))

        cur.execute(query)
        results = cur.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cur.description]

        # Convert results to DataFrame
        df = results_to_dataframe(results, column_names)

        return df
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
        return pd.DataFrame()
    finally:
        if conn:
            cur.close()
            conn.close()


def search_by_date_range(start_date, end_date):
    """
    Search for records within a specified date range.

    :param start_date: Start date (inclusive).
    :type start_date: str
    :param end_date: End date (inclusive).
    :type end_date: str
    :return: A DataFrame containing the search results.
    :rtype: pd.DataFrame
    """
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()
        query = sql.SQL("""
            SELECT i.*, p.flux, p.dflux
            FROM imastats i
            JOIN imaphot p ON i.id = p.id
            WHERE i.date_obs BETWEEN {} AND {}
        """).format(sql.Literal(start_date), sql.Literal(end_date))

        cur.execute(query)
        results = cur.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cur.description]

        # Convert results to DataFrame
        df = results_to_dataframe(results, column_names)

        return df
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
        return pd.DataFrame()
    finally:
        if conn:
            cur.close()
            conn.close()


def search_by_filename(filename_part):
    """
    Search for records by filename or partial filename.

    :param filename_part: Part of the filename to search for.
    :type filename_part: str
    :return: A DataFrame containing the search results.
    :rtype: pd.DataFrame
    """
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()
        query = sql.SQL("""
            SELECT * FROM imastats
            WHERE file_path LIKE {}
        """).format(sql.Literal(f'%{filename_part}%'))

        cur.execute(query)
        results = cur.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cur.description]

        # Convert results to DataFrame
        df = results_to_dataframe(results, column_names)

        return df
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
    finally:
        if conn:
            cur.close()
            conn.close()


def search_transients(date_after):
    """
    Search for transient objects observed after a specified date.

    :param date_after: Date to search for transients after (inclusive).
    :type date_after: str
    :return: A DataFrame containing the search results.
    :rtype: pd.DataFrame
    """
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()
        query = sql.SQL("""
            SELECT p.*, i.file_path, i.date_obs, i.filter
            FROM imaphot p
            JOIN imastats i ON p.id = i.id
            WHERE p.trans = TRUE
              AND i.date_obs > {}
        """).format(sql.Literal(date_after))

        cur.execute(query)
        results = cur.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cur.description]

        # Convert results to DataFrame
        df = results_to_dataframe(results, column_names)

        return df
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
    finally:
        if conn:
            cur.close()
            conn.close()


def search_transient_images(date_after):
    """
    Search for transient images observed after a specified date.

    :param date_after: Date to search for transient images after (inclusive).
    :type date_after: str
    :return: A DataFrame containing the search results.
    :rtype: pd.DataFrame
    """
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()
        query = sql.SQL("""
            SELECT p.id, p.flux, p.dflux, i.file_path, i.date_obs, i.filter
            FROM imaphot p
            JOIN imastats i ON p.id = i.id
            WHERE p.trans = TRUE
              AND i.date_obs > {}
        """).format(sql.Literal(date_after))

        cur.execute(query)
        results = cur.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cur.description]

        # Convert results to DataFrame
        df = results_to_dataframe(results, column_names)

        return df
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        if conn:
            cur.close()
            conn.close()


def search_by_filenames(filenames):
    """
    Search for records by a list of filenames.

    :param filenames: List of filenames to search for.
    :type filenames: list
    :return: A DataFrame containing the search results.
    :rtype: pd.DataFrame
    """
    conn = connect_to_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()

        # Create a string with filenames for the query
        formatted_filenames = ', '.join(sql.Literal(f).as_string(conn) for f in filenames)

        query = sql.SQL("""
            SELECT p.*, i.*
            FROM imaphot p
            JOIN imastats i ON p.id = i.id
            WHERE i.file_path IN ({})
        """).format(sql.SQL(formatted_filenames))

        cur.execute(query)
        results = cur.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cur.description]

        # Convert results to DataFrame
        df = results_to_dataframe(results, column_names)

        return df
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        if conn:
            cur.close()
            conn.close()
