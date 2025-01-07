-- Install the Q3C extension
CREATE EXTENSION IF NOT EXISTS q3c;

-- Create a read-only user
CREATE USER read_only WITH PASSWORD 'read_only';
GRANT CONNECT ON DATABASE "GPUPhotDB" TO read_only;
GRANT USAGE ON SCHEMA public TO read_only;

-- Create the default table
CREATE TABLE IF NOT EXISTS imaphot (
    imageid TEXT UNIQUE,
    ra REAL,
    dec REAL,
    flux REAL,
    dflux REAL,
    trans BOOLEAN
);

-- Grant read permissions to the read_only user
GRANT SELECT ON ALL TABLES IN SCHEMA public TO read_only;

-- Create a Q3C index on the ra and dec columns
CREATE INDEX ON imaphot (q3c_ang2ipix(ra, dec));

-- Create an index on the imageid column
CREATE INDEX ON imaphot (imageid);

-- Insert or update records in the imaphot table
-- Use ON CONFLICT to handle duplicate imageid entries
