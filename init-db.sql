-- Install the Q3C extension
CREATE EXTENSION IF NOT EXISTS q3c;

-- Install the HSTORE extension
CREATE EXTENSION IF NOT EXISTS hstore;

-- Create a read-only user
CREATE USER read_only WITH PASSWORD 'read_only';
GRANT CONNECT ON DATABASE "GPUPhotDB" TO read_only;
GRANT USAGE ON SCHEMA public TO read_only;

-- Create the imaphot table
CREATE TABLE IF NOT EXISTS imaphot (
    id CHAR(40),
    ra REAL,
    dec REAL,
    flux REAL,
    dflux REAL,
    trans BOOLEAN
);

-- Create the imastats table
CREATE TABLE IF NOT EXISTS imastats (
    id CHAR(40) PRIMARY KEY UNIQUE,
    file_path TEXT UNIQUE,
    naxis1 INTEGER,
    naxis2 INTEGER,
    telescop TEXT,
    instrume TEXT,
    camera TEXT,
    filter TEXT,
    date_obs TIMESTAMP,
    exptime REAL,
    object TEXT,
    ra REAL,
    dec REAL,
    fwhm REAL,
    maglim REAL,
    header HSTORE
);

-- Grant read permissions to the read_only user
GRANT SELECT ON ALL TABLES IN SCHEMA public TO read_only;

-- Create a Q3C index on the ra and dec columns
CREATE INDEX ON imaphot (q3c_ang2ipix(ra, dec));
CREATE INDEX ON imastats (q3c_ang2ipix(ra, dec));

-- Create an index on the file_path and id column
CREATE INDEX ON imaphot (id);

CREATE INDEX ON imastats (id);
CREATE INDEX ON imastats (file_path);
CREATE INDEX ON imastats (date_obs);
CREATE INDEX ON imastats (filter);

