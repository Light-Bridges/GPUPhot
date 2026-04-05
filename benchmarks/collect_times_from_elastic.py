#!/usr/bin/env python3
"""
Collect Processing Times from Elasticsearch and Correlate with PostgreSQL Photometry Data.

This script is designed to benchmark the performance of the GPUPhot pipeline by:
1.  Querying **Elasticsearch** to retrieve execution logs of the `process_image` function.
2.  Extracting key metadata (execution time, timestamp, OBLINEID, GPU info) from the logs.
3.  Querying a **PostgreSQL** database (IMA_STATS/IMA_PHOT) to retrieve the ground-truth number of sources detected for each image.
4.  Merging these datasets into a CSV file for further analysis.

Key Features:
- **Robust Elastic Querying**: Uses scroll API to handle large datasets and retries on failure.
- **Two-Phase DB Lookup**: Optimizes database load by first resolving OBLINEIDs to ImageIDs and then aggregating counts, avoiding massive joins.
- **System Info Extraction**: Captures hardware details (GPU model, driver, OS) to allow hardware-specific performance analysis.

Usage Examples:
  # Basic usage (defaults to last 24h):
  python benchmarks/collect_times_from_elastic.py

  # Specify a time range and output file:
  python benchmarks/collect_times_from_elastic.py \
    --time-from "2025-12-01T00:00:00" \
    --time-to "2025-12-31T23:59:59" \
    --out benchmarks/es_times_dec2025.csv

  # Custom Elastic and DB connection (if env vars are not set):
  export ES_URL=http://es-host:9200
  export DATABASE_URL=postgresql://user:pass@db-host:5432/db_name
  python benchmarks/collect_times_from_elastic.py --es-index "gpuphot-events-*"

Environment Variables:
  - ELASTICSEARCH_HOST, ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD
  - IMA_STATS_HOST, IMA_STATS_PORT, IMA_STATS_USER, IMA_STATS_PASSWORD, IMA_STATS_DB (for DB connection)

Output CSV Columns:
  - group_key: The unique identifier (OBLINEID) linking logs to DB records.
  - execution_time: Time taken by `process_image` (seconds).
  - timestamp: Log timestamp.
  - n_sources_detected: Number of sources reported in the log (parsed from return value).
  - db_total: Total sources found in PostgreSQL for this image.
  - db_transients: Number of transient sources found in PostgreSQL.
  - [Metadata]: instrume, camera, filter, naxis1, naxis2, etc.
  - [System Info]: gpu_name, gpu_driver, os, python_ver, etc.
"""

from __future__ import annotations

import os
import sys
import argparse
import csv
import json
import logging
from typing import Dict, Any, Optional, Tuple

# Networking and DB
import requests
import re
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None


LOG = logging.getLogger("collect_times")


# Helper: safe_post with retries (used by many functions)
def safe_post(url: str, headers: dict, payload: Any, auth: Optional[Tuple[str, str]] = None, timeout: int = 60, retries: int = 2):
    """Simple safe POST wrapper around requests.post with json payload and basic retries.

    Returns the requests.Response object or raises the underlying exception after retries.
    """
    session = requests.Session()
    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            resp = session.post(url, json=payload, headers=headers, auth=auth, timeout=timeout)
            return resp
        except Exception as e:
            last_exc = e
            LOG.debug("HTTP POST attempt %d failed: %s", attempt, e)
            if attempt <= retries:
                import time

                time.sleep(0.5 * attempt)
                continue
            raise


# Helper: parse extra.return_value / extra.arguments text for row count and FITS-like header
def _parse_header_and_nrows_from_text(text: str) -> Tuple[Optional[int], Dict[str, Any]]:
    """Parse a blob and extract DataFrame row count and header key/value pairs.

    Returns (n_rows_or_None, header_dict_normalized).
    """
    n_rows = None
    header = {}
    if not text or not isinstance(text, str):
        return None, {}

    m = re.search(r"\[(\d+)\s+rows\s*[x×]\s*(\d+)\s+columns\]", text, flags=re.IGNORECASE)
    if m:
        try:
            n_rows = int(m.group(1))
        except Exception:
            n_rows = None

    dict_like = re.findall(r"['\"]?([A-Za-z0-9_\-]+)['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", text)
    for k, v in dict_like:
        kn = re.sub(r"\W+", "", k).lower()
        val = v.strip()
        if re.fullmatch(r"-?\d+", val):
            try:
                header[kn] = int(val)
                continue
            except Exception:
                pass
        if re.fullmatch(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", val):
            try:
                header[kn] = float(val)
                continue
            except Exception:
                pass
        header[kn] = val

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"\[\s*\d+\s+rows", line):
            continue
        m2 = re.match(r"([A-Z0-9\-]+)\s*=\s*'([^']*)'", line)
        if m2:
            k = m2.group(1)
            v = m2.group(2)
            kn = re.sub(r"\W+", "", k).lower()
            if kn not in header:
                if re.fullmatch(r"-?\d+", v):
                    header[kn] = int(v)
                elif re.fullmatch(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", v):
                    try:
                        header[kn] = float(v)
                    except Exception:
                        header[kn] = v
                else:
                    header[kn] = v
            continue
        m3 = re.match(r"([A-Z0-9\-]+)\s*=\s*([^/]+)", line)
        if m3:
            k = m3.group(1)
            v = m3.group(2).strip()
            v = re.split(r"\s*/\s*", v)[0].strip()
            v = v.strip("\"' ")
            kn = re.sub(r"\W+", "", k).lower()
            if kn not in header:
                if re.fullmatch(r"-?\d+", v):
                    header[kn] = int(v)
                elif re.fullmatch(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", v):
                    try:
                        header[kn] = float(v)
                    except Exception:
                        header[kn] = v
                else:
                    header[kn] = v

    return n_rows, header


def parse_args():
    p = argparse.ArgumentParser(description="Collect processing times from ES and correlate with photometry DB")
    p.add_argument("--es-url", default=os.environ.get("ELASTICSEARCH_HOST", "http://10.0.210.30:9200"), help="Elasticsearch base URL (ENV: ELASTICSEARCH_HOST)")
    p.add_argument("--es-index", default="logstash-*", help="Elasticsearch index or index pattern to query (e.g. logstash-*)")
    p.add_argument("--es-user", default=os.environ.get("ELASTICSEARCH_USER", 'elastic'), help="ES basic auth user (ENV: ELASTICSEARCH_USER)")
    p.add_argument("--es-pass", default=os.environ.get("ELASTICSEARCH_PASSWORD", 'admin'), help="ES basic auth pass (ENV: ELASTICSEARCH_PASSWORD)")
    p.add_argument("--time-from", help="RFC3339 or ES date math string for range start, optional")
    p.add_argument("--time-to", help="RFC3339 or ES date math string for range end, optional")
    # Do NOT auto-consume DATABASE_URL environment variable here to avoid
    # accidentally using a read-only or CI-specific URL during local tests.
    # If the user wants to use a custom database, they can pass --db-url explicitly.
    p.add_argument("--db-url", default=None, help="Postgres DSN (sqlalchemy style or libpq). If omitted the script will assemble a default from IMA_STATS_* env vars")
    p.add_argument("--out", dest="out_csv", default="benchmarks/es_times_by_image.csv", help="Output CSV file")
    p.add_argument("--es-size", type=int, default=10000, help="Max number of image buckets to request from ES (terms size)")
    p.add_argument("--es-query-file", help="Path to a JSON file with the exact ES request body to POST (if provided, used as-is)")
    # By default group by the full extra.return_value blob and extract OBID via regex
    p.add_argument("--es-group-field", default="extra.return_value", help="Field used to group events (can be in fields[] or _source). Default: extra.return_value")
    p.add_argument("--es-duration-field", default="extra.execution_time", help="Field that stores execution time/duration (seconds). Default: extra.execution_time")
    p.add_argument("--es-timestamp-field", default="@timestamp", help="Timestamp field name to use (default @timestamp)")
    p.add_argument("--es-group-regex", default=r"OBLINEID\s*=\s*(\d+)|OBID\s*=\s*(\d+)", help="Regex to extract OBLINEID or OBID from the group field content (prefers OBLINEID)")
    # enable scroll by default; provide --no-scroll to disable
    p.add_argument("--no-scroll", action="store_false", dest="use_scroll", help="Disable Elasticsearch scroll API (enabled by default)")
    p.set_defaults(use_scroll=True)
    p.add_argument("--scroll-ttl", default="2m", help="Scroll context TTL (e.g. 2m)")
    p.add_argument("--scroll-size", type=int, default=500, help="Number of hits per scroll page")
    p.add_argument("--max-hits", type=int, default=0, help="Optional max number of hits to fetch (0 = unlimited)")
    # per-hit export enabled by default; provide --no-per-hit to disable
    p.add_argument("--no-per-hit", action="store_false", dest="per_hit", help="Disable per-hit export (enabled by default)")
    p.set_defaults(per_hit=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dump-samples", type=int, default=0, help="If >0, write up to N raw extra.return_value samples to a JSONL file for inspection")
    p.add_argument("--samples-out", default=None, help="Path to write samples JSONL (default: <out>.samples.jsonl)")
    #    p.add_argument("--db-date-from", default=None, help="Optional lower bound on image date (dateobs) to filter DB queries (ISO format or ES date math).")
    #    p.add_argument("--db-date-to", default=None, help="Optional upper bound on image date (dateobs) to filter DB queries (ISO format or ES date math).")
    # Note: DB date filters are not accepted. The DB filter range is derived from DATE-OBS values extracted from ES hits.
    return p.parse_args()


def build_es_agg_query(time_from: Optional[str], time_to: Optional[str], size: int = 10000, group_field: str = "extra.path", duration_field: str = "extra.execution_time") -> Dict[str, Any]:
    # Build a lightweight aggregation query to compute per-image stats
    # Try to adapt the provided field names to ES-appropriate names for aggregation
    # If the user provided a dotted field like extra.path, we attempt to use the keyword subfield for grouping
    aggr_field = group_field
    if not aggr_field.endswith(".keyword"):
        aggr_field_keyword = f"{aggr_field}.keyword"
    else:
        aggr_field_keyword = aggr_field

    query: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"must": []}},
        "aggs": {
            "by_group": {
                "terms": {"field": aggr_field_keyword, "size": size},
                "aggs": {
                    "avg_duration": {"avg": {"field": duration_field}},
                    "min_ts": {"min": {"field": "@timestamp"}},
                    "max_ts": {"max": {"field": "@timestamp"}}
                }
            }
        }
    }

    if time_from or time_to:
        rng = {k: v for k, v in (("gte", time_from), ("lte", time_to)) if v}
        query["query"]["bool"]["must"].append({"range": {"timestamp": rng}})

    return query


def fetch_es_buckets(es_url: str, es_index: str, query: Dict[str, Any], auth: Optional[Tuple[str, str]] = None, group_field: str = "extra.path", duration_field: str = "extra.execution_time", timestamp_field: str = "@timestamp", use_scroll: bool = False, scroll_ttl: str = "2m", scroll_size: int = 500, max_hits: int = 0) -> Dict[str, Dict[str, Any]]:
    """Execute ES search (aggregation or hits) and return a mapping group_key->stats.
    Supports both aggregation responses (aggregations.by_group.buckets) and hits[].fields format as produced by Kibana queries.
    """
    search_url = f"{es_url.rstrip('/')}/{es_index}/_search"
    headers = {"Content-Type": "application/json"}
    # make a copy of the query and strip internal keys not valid for ES
    if isinstance(query, dict):
        payload = dict(query)
        payload.pop('_es_group_regex', None)
    else:
        payload = query
    LOG.debug("Querying ES %s with payload: %s", search_url, json.dumps(payload)[:1000])
    resp = safe_post(search_url, headers, payload, auth=auth, timeout=60)
    # resp = requests.post(search_url, headers=headers, data=json.dumps(payload), auth=auth, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Case A: Aggregations present (our own built query)
    aggs = data.get("aggregations")
    if aggs and "by_group" in aggs:
        buckets = aggs.get("by_group", {}).get("buckets", [])
        result: Dict[str, Dict[str, Any]] = {}
        for b in buckets:
            key = b.get("key")
            result[key] = {
                "avg_processing_time_s": b.get("avg_duration", {}).get("value") if isinstance(b.get("avg_duration"), dict) else None,
                "n_events": b.get("doc_count"),
                "first_ts": b.get("min_ts", {}).get("value_as_string") if b.get("min_ts") else None,
                "last_ts": b.get("max_ts", {}).get("value_as_string") if b.get("max_ts") else None,
            }
        return result

    # Case B: No aggregations - expect hits[] (Kibana-style query that returns fields)
    # If use_scroll is requested, paginate using the scroll API and collect all hits
    hits = []
    if use_scroll:
        # Prepare initial payload: ensure size is set
        payload = dict(query)
        # If user limited max_hits, request at most that many in the first page to avoid over-fetching
        initial_size = scroll_size if not max_hits else min(scroll_size, max_hits)
        payload["size"] = initial_size
        # ensure no internal keys are sent to ES
        payload.pop('_es_group_regex', None)
        # track_total_hits cannot be disabled in a scroll context
        payload.pop('track_total_hits', None)
        search_url = f"{es_url.rstrip('/')}/{es_index}/_search?scroll={scroll_ttl}"
        LOG.info("Starting scroll search (size=%d) to %s", scroll_size, search_url)
        resp = safe_post(search_url, {"Content-Type": "application/json"}, payload, auth=auth, timeout=120)
        # resp = requests.post(search_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), auth=auth, timeout=120)
        # resp.raise_for_status()
        data_all = resp.json()
        sid = data_all.get("_scroll_id")
        page_hits = data_all.get("hits", {}).get("hits", [])
        hits.extend(page_hits)
        total_fetched = len(page_hits)
        LOG.debug("Fetched page hits=%d, total=%d", len(page_hits), total_fetched)
        # Loop until no more hits or max_hits reached
        while sid and page_hits:
            if max_hits and total_fetched >= max_hits:
                LOG.info("Reached max_hits=%d, stopping scroll", max_hits)
                break
            scroll_url = f"{es_url.rstrip('/')}/_search/scroll"
            body = {"scroll": scroll_ttl, "scroll_id": sid}
            resp = safe_post(scroll_url, {"Content-Type": "application/json"}, body, auth=auth, timeout=120)
            # resp = requests.post(scroll_url, headers={"Content-Type": "application/json"}, data=json.dumps(body), auth=auth, timeout=120)
            # resp.raise_for_status()
            data_all = resp.json()
            sid = data_all.get("_scroll_id")
            page_hits = data_all.get("hits", {}).get("hits", [])
            hits.extend(page_hits)
            total_fetched += len(page_hits)
            LOG.debug("Fetched page hits=%d, total=%d", len(page_hits), total_fetched)
        LOG.info("Scroll completed, total hits fetched=%d", len(hits))
    else:
        # Non-scroll search: avoid sending internal key
        if isinstance(query, dict):
            payload = dict(query)
            payload.pop('_es_group_regex', None)
            payload.pop('track_total_hits', None)
        else:
            payload = query
        resp = safe_post(search_url, headers, payload, auth=auth, timeout=60)
        # resp = requests.post(search_url, headers=headers, data=json.dumps(payload), auth=auth, timeout=60)
        hits = resp.json().get("hits", {}).get("hits", [])

    # We'll aggregate on the client side and also extract metadata from extra.return_value
    stats: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        # Try fields first (Kibana returns 'fields'), then _source
        rec_fields = h.get("fields") or {}
        src = h.get("_source") or {}

        def extract(field_name: str):
            # prefer fields[], which often stores arrays
            if field_name in rec_fields:
                val = rec_fields.get(field_name)
                if isinstance(val, list):
                    return val[0]
                return val
            # try keyword variant
            kw = f"{field_name}.keyword"
            if kw in rec_fields:
                v = rec_fields.get(kw)
                if isinstance(v, list):
                    return v[0]
                return v
            # fallback to _source
            if field_name in src:
                return src.get(field_name)
            # try nested extraction for dotted names
            parts = field_name.split('.')
            cur = src
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    cur = None
                    break
            return cur

        group_key = extract(group_field)
        if group_key is None:
            # try fallback keys that might identify images
            group_key = extract('extra.path') or extract('extra.observation_night') or 'UNDEFINED'

        dur = extract(duration_field)
        # extract timestamp
        ts = extract(timestamp_field) or extract('@timestamp')

        # Normalize possible list values
        if isinstance(group_key, list):
            group_key = group_key[0]
        # If user provided a regex to extract a sub-key, apply it
        # (e.g. extract OBID from a header-like string stored in extra.return_value)
        # We'll try to apply the regex on the string representation of the group_key.
        # The regex should contain one capture group.
        es_group_regex = query.get('_es_group_regex') if isinstance(query, dict) else None
        # Note: prefer regex passed as parameter, but in this function we cannot access CLI args directly.
        # We will allow the caller to pass regex via the query payload under key '_es_group_regex',
        # or you can supply via CLI; the main() passes args.es_group_regex via the query payload below.
        if es_group_regex and isinstance(group_key, (str, bytes)):
            try:
                m = re.search(es_group_regex, str(group_key))
                if m:
                    # pick the first non-empty capture group (supports alternatives)
                    g = next((gg for gg in m.groups() if gg), None)
                    if g:
                        group_key = g
            except re.error:
                pass

        try:
            dur_val = float(dur) if dur is not None else None
        except Exception:
            dur_val = None

        # Parse extra.return_value for number of rows and header fields if present
        n_sources = None
        header_vals: Dict[str, Any] = {}
        # Prefer to extract from rec_fields if present
        rv = None
        if 'extra.return_value' in rec_fields:
            v = rec_fields.get('extra.return_value')
            if isinstance(v, list):
                rv = v[0]
            else:
                rv = v
        elif 'extra.return_value' in src:
            rv = src.get('extra.return_value')
        # rv is expected to be a long string containing DataFrame summary and FITS header
        # If extra.return_value isn't present, try extra.arguments as fallback
        if not rv:
            # try extra.arguments as fallback
            if 'extra.arguments' in rec_fields:
                av = rec_fields.get('extra.arguments')
                rv = av[0] if isinstance(av, list) else av
            elif 'extra.arguments' in src:
                rv = src.get('extra.arguments')
        if rv and isinstance(rv, str):
            parsed_n, parsed_header = _parse_header_and_nrows_from_text(rv)
            if parsed_n is not None:
                n_sources = parsed_n
            if parsed_header:
                header_vals.update(parsed_header)

        # Prefer OBLINEID over OBID when choosing group_key; add logging and robust handling to fetch_db_counts, returning mapping keyed by OBLINEID if present, else OBID. Log DB connection attempts and number of rows returned, and handle exceptions.
        # Prefer OBLINEID if present (unique per image), otherwise OBID; use as group_key to link to Postgres
        if header_vals:
            prefer_key = None
            for candidate in ('oblineid', 'obid'):
                if header_vals.get(candidate) is not None:
                    prefer_key = header_vals.get(candidate)
                    break
            if prefer_key is not None:
                try:
                    group_key = str(int(prefer_key))
                except Exception:
                    group_key = str(prefer_key)

        st = stats.setdefault(group_key, {"sum_dur": 0.0, "count": 0, "first_ts": None, "last_ts": None, "n_sources": None, "header": {}})
        if dur_val is not None:
            st["sum_dur"] += dur_val
            st["count"] += 1
        # store parsed values (if any) - keep the maximum n_sources observed
        if n_sources is not None:
            try:
                if st.get('n_sources') is None or (isinstance(n_sources, int) and n_sources > st.get('n_sources', 0)):
                    st['n_sources'] = n_sources
            except Exception:
                st['n_sources'] = n_sources
        # merge header_vals
        if header_vals:
            st_header = st.setdefault('header', {})
            for kkk,vvv in header_vals.items():
                if kkk not in st_header:
                    st_header[kkk] = vvv

    # Convert to output format
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in stats.items():
        avg = (v["sum_dur"] / v["count"]) if v["count"] > 0 else None
        out[k] = {
            "avg_processing_time_s": avg,
            "n_events": v["count"],
            "first_ts": v["first_ts"],
            "last_ts": v["last_ts"],
            "n_sources_detected": v.get("n_sources"),
            "header": v.get("header", {}),
        }
    return out


def fetch_db_counts(db_url: str, keys: Optional[list] = None, date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, int]:
    """Query Postgres for counts per OBID by joining imaphot_v31 -> imastats_v31.

    Returns a mapping obid_str -> num_sources (int). If DATABASE_URL is not provided or
    psycopg2 is unavailable, returns an empty dict.
    """
    if not db_url:
        LOG.warning("No DATABASE_URL provided; returning empty DB counts")
        return {}
    if psycopg2 is None:
        LOG.error("psycopg2 not available in this environment. Install it to query Postgres.")
        return {}

    # Mask DB URL for logging
    try:
        masked_db = re.sub(r"(^.*://[^:]+):[^@]+@", r"\1:***@", db_url)
    except Exception:
        masked_db = '***'
    LOG.info("fetch_db_counts: connecting to DB: %s", masked_db)

    # WARNING: this function may be expensive on huge tables. We'll implement
    # a safer version that first obtains imageid lists per key and then
    # aggregates counts in imaphot_v31 per-imageid arrays to avoid a large join.
    sql = """
    WITH key_images AS (
        SELECT COALESCE(m.header->'OBLINEID', m.header->'OBID') AS key,
               array_agg(m.imageid) AS imageids
        FROM imastats_v31 m
        WHERE (m.header ? 'OBLINEID') OR (m.header ? 'OBID')
        GROUP BY key
    )
    SELECT ki.key,
           COALESCE(p.cnt,0) AS num_sources,
           COALESCE(p.trans,0) AS num_transients
    FROM key_images ki
    LEFT JOIN LATERAL (
        SELECT COUNT(*) AS cnt, SUM(CASE WHEN trans IS TRUE THEN 1 ELSE 0 END) AS trans
        FROM imaphot_v31 p
        WHERE p.imageid = ANY(ki.imageids)
    ) p ON true
    """
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        LOG.info("fetch_db_counts: DB query returned %d rows", len(rows))
        result = {}
        for key, num, num_transients in rows:
            if key is None:
                continue
            # key may be returned as text; normalize to integer-like string when possible
            ks = str(key)
            try:
                ks = str(int(ks))
            except Exception:
                ks = ks
            result[ks] = {"total": int(num), "transients": int(num_transients) if num_transients is not None else 0}
        return result
    except Exception as e:
        LOG.exception("fetch_db_counts: DB query failed: %s", e)
        return {}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def fetch_db_counts_for_keys(db_url: str, keys: list) -> Dict[str, Dict[str, int]]:
    """Fetch counts for the provided list of OBLINEID/OBID keys in bulk.
    Uses a 2-phase approach:
      1. Resolve keys to imageids using imastats_v31.
      2. Query imaphot_v31 using the resolved imageids to get counts.

    Returns mapping key_str -> { 'total': int, 'transients': int }
    """
    if not db_url:
        LOG.warning("No DATABASE_URL provided; returning empty DB counts for keys")
        return {}
    if psycopg2 is None:
        LOG.error("psycopg2 not available; cannot query DB for keys")
        return {}
    if not keys:
        return {}

    # Normalize keys to strings and filter (we only consider OBLINEID keys; skip 'UNDEFINED')
    key_list = [str(k) for k in set(keys) if k and str(k).upper() != 'UNDEFINED']
    if not key_list:
        return {}

    # Mask DB URL for logging
    try:
        masked_db = re.sub(r"(^.*://[^:]+):[^@]+@", r"\1:***@", db_url)
    except Exception:
        masked_db = '***'
    LOG.info("fetch_db_counts_for_keys: connecting to DB: %s (keys=%d)", masked_db, len(key_list))

    # We'll process keys in batches to avoid huge parameter lists and heavy memory usage.
    batch_size = 200
    conn = None
    result = {}
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        for i in range(0, len(key_list), batch_size):
            batch = key_list[i:i+batch_size]
            LOG.debug('Querying DB for batch %d..%d (keys=%d)', i, i+len(batch)-1, len(batch))
            
            # Step 1: Resolve imageids from imastats_v31
            # We use a VALUES clause to pass keys.
            vals = []
            params = []
            for k in batch:
                vals.append('(%s::text)')
                params.append(k)
            values_clause = ','.join(vals)
            
            # We filter imastats_v31 by matching OBLINEID to the key.
            # Using hstore operator -> to get value as text.
            sql_step1 = f"""
            WITH input_keys(key) AS (VALUES {values_clause})
            SELECT m.imageid, k.key
            FROM imastats_v31 m
            JOIN input_keys k ON (
                (m.header->'OBLINEID' = k.key)
            )
            """
            cur.execute(sql_step1, tuple(params))
            rows_step1 = cur.fetchall()
            
            if not rows_step1:
                continue

            # Collect imageids and map them back to keys
            imageids = []
            imageid_to_keys = {} # imageid -> list of keys (usually one)
            for img_id, k in rows_step1:
                imageids.append(img_id)
                if img_id not in imageid_to_keys:
                    imageid_to_keys[img_id] = []
                imageid_to_keys[img_id].append(k)
            
            # Step 2: Get counts from imaphot_v31 using the resolved imageids
            # This avoids joining the huge tables directly.
            if not imageids:
                continue
                
            sql_step2 = """
            SELECT imageid, count(*), sum(case when trans then 1 else 0 end)
            FROM imaphot_v31
            WHERE imageid = ANY(%s)
            GROUP BY imageid
            """
            cur.execute(sql_step2, (list(set(imageids)),))
            rows_step2 = cur.fetchall()
            
            # Step 3: Aggregate results back to keys
            counts_map = {r[0]: {'total': r[1], 'trans': r[2]} for r in rows_step2}
            
            # Iterate over the resolved mappings from Step 1
            for img_id, k in rows_step1:
                c = counts_map.get(img_id, {'total': 0, 'trans': 0})
                ks = str(k)
                try:
                    ks = str(int(ks))
                except Exception:
                    ks = ks
                
                if ks not in result:
                    result[ks] = {'total': 0, 'transients': 0}
                
                # Sum up counts (in case multiple images map to the same key, though unlikely for OBLINEID)
                result[ks]['total'] += c['total']
                result[ks]['transients'] += (c['trans'] if c['trans'] else 0)
                
        LOG.info('fetch_db_counts_for_keys: Finished processing all batches. Found results for %d keys.', len(result))
        return result
    except Exception as e:
        LOG.exception("fetch_db_counts_for_keys: DB query failed: %s", e)
        return {}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def write_per_hit_csv(rows: list, db_map: Dict[str, Dict[str, int]], out_csv: str):
    """Write a CSV with per-hit rows enriched with DB counts and parsed header fields."""
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    header_fields = ['instrume','inmodel','camera','naxis1','naxis2','obid','orid','dateproc','inserial','telescop','filter','dateobs','exptime','object']
    # System info fields to extract
    sys_fields = [
        'gpu_name', 'gpu_driver', 'gpu_id', 'gpu_load', 'gpu_mem_free', 'gpu_mem_total', 'gpu_mem_used', 'gpu_temp',
        'arch', 'cupy_ver', 'kernel', 'os', 'os_ver', 'processor', 'python_ver'
    ]
    columns = [
        'group_key', 'execution_time', 'timestamp', 'n_sources_detected', 'db_total', 'db_transients'
    ] + header_fields + sys_fields
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for r in rows:
            key = str(r.get('group_key')) if r.get('group_key') is not None else ''
            dbc = db_map.get(key) if db_map else None
            db_total = dbc.get('total') if isinstance(dbc, dict) else dbc
            db_trans = dbc.get('transients') if isinstance(dbc, dict) else None
            header = r.get('header') or {}
            sys_info = r.get('sys_info') or {}
            row = [
                key,
                r.get('execution_time'),
                r.get('timestamp'),
                r.get('n_sources_detected'),
                db_total,
                db_trans,
            ]
            for hf in header_fields:
                row.append(header.get(hf))
            for sf in sys_fields:
                row.append(sys_info.get(sf))
            writer.writerow(row)
    LOG.info('Wrote per-hit results to %s (rows=%d)', out_csv, len(rows))


def merge_and_write(es_map: Dict[str, Dict[str, Any]], db_map: Dict[str, int], out_csv: str):
    # Build rows covering union of keys
    keys = set(es_map.keys()) | set(db_map.keys())
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    # Header fields we parsed from extra.return_value (normalized names)
    header_fields = ['instrume','inmodel','camera','naxis1','naxis2','obid','orid','dateproc','inserial','telescop','filter','dateobs','exptime','object']
    columns = [
        "group_key",
        "avg_processing_time_s",
        "n_events",
        "n_sources_detected",
        "num_sources_db_total",
        "num_sources_db_transients",
        "first_ts",
        "last_ts",
    ] + header_fields
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for k in sorted(keys):
            esv = es_map.get(k, {}) or {}
            header = esv.get('header', {}) if esv else {}
            db_counts = db_map.get(k) if db_map else None
            db_total = db_counts.get('total') if isinstance(db_counts, dict) else db_counts
            db_trans = db_counts.get('transients') if isinstance(db_counts, dict) else None
            row = [
                k,
                esv.get("avg_processing_time_s"),
                esv.get("n_events", 0),
                esv.get("n_sources_detected"),
                db_total,
                db_trans,
                esv.get("first_ts"),
                esv.get("last_ts"),
            ]
            for hf in header_fields:
                row.append(header.get(hf))
            writer.writerow(row)
    LOG.info("Wrote results to %s", out_csv)


def verify_index_exists(es_url: str, es_index: str, auth: Optional[Tuple[str, str]] = None) -> list:
    """Verifica si el índice (o patrón) existe en ES.
    - Si el índice exacto existe hace return [index]
    - Si no, intenta consultar _cat/indices/{pattern} para devolver coincidencias
    - Si no hay coincidencias, devuelve una lista de índices disponibles limitada a 50
    """
    base = es_url.rstrip('/')
    # Try HEAD on the exact index
    try:
        head_url = f"{base}/{es_index}"
        r = requests.head(head_url, auth=auth, timeout=10)
        if r.status_code == 200:
            return [es_index]
    except Exception:
        # ignore and try _cat
        pass

    # Try _cat/indices with the pattern
    try:
        cat_url = f"{base}/_cat/indices/{es_index}?format=json&h=index"
        r = requests.get(cat_url, auth=auth, timeout=15)
        if r.status_code == 200:
            arr = r.json()
            matches = [item.get('index') for item in arr if 'index' in item]
            if matches:
                return matches
    except Exception:
        pass

    # Fallback: list some indices to help the user
    try:
        cat_all = f"{base}/_cat/indices?format=json&h=index,store.size&s=store.size:desc"
        r = requests.get(cat_all, auth=auth, timeout=15)
        if r.status_code == 200:
            arr = r.json()
            sample = [item.get('index') for item in arr[:50] if 'index' in item]
            return sample
    except Exception:
        pass

    return []


def estimate_total_hits(es_url: str, es_index: str, query: Dict[str, Any], auth: Optional[Tuple[str, str]] = None) -> Optional[int]:
    """Run a lightweight search (size=0) to estimate total matching hits for the provided query.
    Returns the total count or None on error.
    """
    try:
        base = es_url.rstrip('/')
        url = f"{base}/{es_index}/_search"
        # copy payload and request no hits
        if isinstance(query, dict):
            payload = dict(query)
            payload.pop('_es_group_regex', None)
            payload['size'] = 0
            # Ensure track_total_hits is not disabled when we want totals
            if payload.get('track_total_hits') is False:
                payload.pop('track_total_hits', None)
        else:
            payload = {'query': query, 'size': 0}
        resp = safe_post(url, {'Content-Type': 'application/json'}, payload, auth=auth, timeout=30)
        data = resp.json()
        hits = data.get('hits', {}).get('total')
        # hits can be either a number or an object {'value': N, 'relation': 'eq'}
        if isinstance(hits, dict):
            return int(hits.get('value', 0))
        if isinstance(hits, (int, float)):
            return int(hits)
    except Exception:
        LOG.debug('Could not estimate total hits')
    return None


def main():
    logging.basicConfig(level=logging.INFO)
    # Debug marker to ensure main started (visible when running interactively)
    print('collect_times: main() start', flush=True)
    args = parse_args()
    # Log effective arguments (mask sensitive values)
    try:
        cfg = dict(vars(args))
        # mask ES password and DB URL password if present
        if cfg.get('es_pass'):
            cfg['es_pass'] = '***'
        if cfg.get('db_url'):
            try:
                cfg['db_url'] = re.sub(r"(^.*://[^:]+):[^@]+@", r"\1:***@", cfg['db_url'])
            except Exception:
                cfg['db_url'] = '***'
        LOG.info("Effective CLI args: %s", json.dumps(cfg, indent=2))
    except Exception:
        LOG.debug('Could not serialize args for logging')

    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    if args.es_user and args.es_pass:
        auth = (args.es_user, args.es_pass)
    else:
        auth = None

    # If no explicit DB URL provided, do NOT use DATABASE_URL env implicitly (it may point to read_only CI URL).
    # Instead, prefer building a default DB URL from IMA_STATS_* env vars. If the user wants to use DATABASE_URL,
    # they must pass it explicitly via --db-url.
    if not args.db_url:
        if os.environ.get('DATABASE_URL') is not None:
            LOG.info("Environment variable DATABASE_URL detected but will be ignored (use --db-url to override). Using IMA_STATS_* defaults instead.")

        # Provide defaults so the user doesn't need to export env vars each time.
        ima_host = os.environ.get('IMA_STATS_HOST', '10.0.210.30')
        ima_port = os.environ.get('IMA_STATS_PORT', '5433')
        ima_user = os.environ.get('IMA_STATS_USER', 'postgres')
        ima_pass = os.environ.get('IMA_STATS_PASSWORD', 'admin')
        ima_db = os.environ.get('IMA_STATS_DB', 'astrodb')
        # Build DSN unconditionally (we have defaults). If the real environment overrides, it will be used.
        args.db_url = f"postgresql://{ima_user}:{ima_pass}@{ima_host}:{ima_port}/{ima_db}"
        # Mask password for logging
        try:
            masked = f"postgresql://{ima_user}:***@{ima_host}:{ima_port}/{ima_db}"
        except Exception:
            masked = args.db_url
        LOG.info("No DATABASE_URL provided; using IMA_STATS defaults to build DB URL: %s", masked)

    # If user provided a raw ES query file, use it as-is
    if args.es_query_file:
        with open(args.es_query_file, 'r') as qf:
            query_payload = json.load(qf)
    else:
        # Default: build a raw Kibana-style query that returns hits (per-hit) for process_image
        # This avoids using aggregations by default and returns the fields we need.
        query_payload = {
            "track_total_hits": False,
            "size": args.scroll_size if args.use_scroll else args.es_size,
            "version": True,
            # request both _source and fields to maximize chance of retrieving large return_value blobs
            "_source": [
                "@timestamp",
                "extra.execution_time",
                "extra.function_name",
                "extra.path",
                "extra.system_info.gpu.name",
                "extra.system_info.gpu.driver_version",
                "extra.system_info.gpu.id",
                "extra.system_info.gpu.load",
                "extra.system_info.gpu.memory_free",
                "extra.system_info.gpu.memory_total",
                "extra.system_info.gpu.memory_used",
                "extra.system_info.gpu.temperature",
                "extra.system_info.architecture",
                "extra.system_info.cupy_version",
                "extra.system_info.kernel_version",
                "extra.system_info.os",
                "extra.system_info.os_version",
                "extra.system_info.processor",
                "extra.system_info.python_version"
            ],
            "fields": [
                "extra.return_value",
                "extra.arguments",
                "@timestamp",
                "extra.system_info.gpu.name",
                "extra.system_info.gpu.driver_version",
                "extra.system_info.gpu.id",
                "extra.system_info.gpu.load",
                "extra.system_info.gpu.memory_free",
                "extra.system_info.gpu.memory_total",
                "extra.system_info.gpu.memory_used",
                "extra.system_info.gpu.temperature",
                "extra.system_info.architecture",
                "extra.system_info.cupy_version",
                "extra.system_info.kernel_version",
                "extra.system_info.os",
                "extra.system_info.os_version",
                "extra.system_info.processor",
                "extra.system_info.python_version"
            ],
            "query": {
                "bool": {
                    "filter": [
                        # {"match_phrase": {"extra.environment": "profiler"}},
                        {"wildcard": {"extra.environment": "*profiler*"}},
                        # {"match_phrase": {"extra.environment": "nvtx"}},
                        {"match_phrase": {"extra.application": "gpuphot"}},
                        {"match_phrase": {"extra.function_name": "process_image"}},
                        {"exists": {"field": "extra.execution_time"}},
                        {"exists": {"field": "extra.return_value"}}
                    ]
                }
            }
        }

    # If user didn't specify a time window, apply a conservative default to avoid scanning the entire index.
    # Default: last 30 days (Elasticsearch date math). If user passed --time-from/--time-to, respect them.
    if not args.es_query_file:
        if not args.time_from and not args.time_to:
            default_from = "now-1d"
            default_to = "now"
            LOG.info("No time range provided; applying default time window: %s - %s", default_from, default_to)
            # Attach a range filter on @timestamp in the query payload
            try:
                rng = {"range": {"@timestamp": {"gte": default_from, "lte": default_to}}}
                query_payload["query"]["bool"]["filter"].append(rng)
            except Exception:
                LOG.debug("Could not attach default time range to query payload")
        else:
            # User provided at least one boundary; attach it explicitly
            rng = {k: v for k, v in (("gte", args.time_from), ("lte", args.time_to)) if v}
            if rng:
                LOG.info("Applying user-specified time window: %s", rng)
                try:
                    query_payload["query"]["bool"]["filter"].append({"range": {"@timestamp": rng}})
                except Exception:
                    LOG.debug("Could not attach user time range to query payload")

    # If user provided a regex to extract group key (e.g., OBID from header string), attach it to the query payload
    if args.es_group_regex:
        try:
            # store it under a reserved key that won't be sent to ES if using aggregations, but will be present if sending a raw query
            query_payload['_es_group_regex'] = args.es_group_regex
        except Exception:
            LOG.debug('Could not attach es_group_regex to query payload')

    # Log ES query payload (truncated) for visibility
    try:
        qp = json.dumps(query_payload, ensure_ascii=False)
        if len(qp) > 2000:
            LOG.info("ES query payload (truncated): %s... [total_len=%d]", qp[:2000], len(qp))
        else:
            LOG.info("ES query payload: %s", qp)
    except Exception:
        LOG.debug('Could not serialize query payload for logging')

    # Log final effective runtime configuration (mask sensitive values)
    try:
        final_cfg = dict(vars(args))
        # mask DB password inside db_url if present
        if final_cfg.get('db_url'):
            try:
                final_cfg['db_url'] = re.sub(r"(^.*://[^:]+):[^@]+@", r"\1:***@", final_cfg['db_url'])
            except Exception:
                final_cfg['db_url'] = '***'
        if final_cfg.get('es_pass'):
            final_cfg['es_pass'] = '***'
        LOG.info("Effective runtime configuration: %s", json.dumps(final_cfg, indent=2, ensure_ascii=False))
    except Exception:
        LOG.debug('Could not serialize final cfg for logging')

    LOG.info("Verifying index existence in Elasticsearch %s (index pattern=%s)", args.es_url, args.es_index)
    index_matches = verify_index_exists(args.es_url, args.es_index, auth=auth)
    if not index_matches:
        LOG.warning("No matching indices found for pattern '%s'. Available indices (sample): %s", args.es_index, ", ".join(index_matches))
        LOG.warning("Exiting due to missing index.")
        sys.exit(1)

    # Decide which index string to use for the search API
    if len(index_matches) == 1 and index_matches[0] == args.es_index:
        es_index_to_use = args.es_index
    elif len(index_matches) == 1:
        es_index_to_use = index_matches[0]
        LOG.info("Using matched index: %s", es_index_to_use)
    else:
        # Multiple matches: pass them as comma-separated list (Elasticsearch supports this)
        es_index_to_use = ",".join(index_matches)
        LOG.info("Pattern matched multiple indices; using indices: %s", es_index_to_use)

    LOG.info("Querying Elasticsearch %s (index=%s)", args.es_url, es_index_to_use)
    # Elasticsearch disallows disabling track_total_hits in a scroll context
    if args.use_scroll and isinstance(query_payload, dict):
        if query_payload.get('track_total_hits') is False:
            LOG.debug("Removing track_total_hits from query payload because scroll is enabled")
            query_payload.pop('track_total_hits', None)
    # Estimate total hits to help avoid accidental full-index scans
    try:
        total_est = estimate_total_hits(args.es_url, es_index_to_use, query_payload, auth=auth)
        if total_est is not None:
            LOG.info("Estimated total matching documents in ES: %d", total_est)
            if total_est > 10000 and not args.max_hits:
                LOG.warning("Large result set estimated (%d). Consider using --max-hits or --time-from/--time-to to limit the query.", total_est)
    except Exception:
        LOG.debug("Could not estimate total hits")
    hits_rows = fetch_hits_raw(args.es_url, es_index_to_use, query_payload, auth=auth, group_field=args.es_group_field, duration_field=args.es_duration_field, timestamp_field=args.es_timestamp_field, use_scroll=args.use_scroll, scroll_ttl=args.scroll_ttl, scroll_size=args.scroll_size, max_hits=args.max_hits)
    # Optionally dump raw extra.return_value samples for inspection
    if args.dump_samples and args.dump_samples > 0:
        samples_out = args.samples_out or f"{args.out_csv}.samples.jsonl"
        try:
            os.makedirs(os.path.dirname(samples_out) or '.', exist_ok=True)
            written = 0
            with open(samples_out, 'w', encoding='utf8') as sf:
                for r in hits_rows:
                    rv = r.get('raw_return')
                    if not rv:
                        continue
                    # Keep a compact sample record
                    rec = {
                        'group_key': r.get('group_key'),
                        'timestamp': r.get('timestamp'),
                        'execution_time': r.get('execution_time'),
                        'n_sources_detected': r.get('n_sources_detected'),
                        'raw_return': rv,
                    }
                    sf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    written += 1
                    if written >= args.dump_samples:
                        break
            LOG.info('Wrote %d raw samples to %s', written, samples_out)
        except Exception as e:
            LOG.warning('Could not write samples file %s: %s', samples_out, e)

    # Now enrich per-hit rows by querying Postgres for the extracted keys and write a per-hit CSV
    try:
        # Extract unique keys from hits_rows for DB lookup (prefer header-derived group_key)
        all_keys = set()
        for r in hits_rows:
            k = r.get('group_key')
            if k and str(k).upper() != 'UNDEFINED':
                all_keys.add(str(k))

            # also try to pick oblineid/obid from parsed header if available
            hdr = r.get('header') or {}
            for cand in ('oblineid', 'obid'):
                if hdr.get(cand) is not None:
                    try:
                        cand_key = str(int(hdr.get(cand)))
                    except Exception:
                        cand_key = str(hdr.get(cand))
                    all_keys.add(cand_key)

        all_keys = [k for k in all_keys if k]
        LOG.info('Fetched %d unique keys for DB count lookup', len(all_keys))

        db_counts_enriched = fetch_db_counts_for_keys(args.db_url, all_keys) if all_keys else {}
        LOG.info('Enriched results with DB counts (found counts for %d keys)', len(db_counts_enriched))
        if db_counts_enriched:
            # log a small sample of results for visibility
            sample_items = list(db_counts_enriched.items())[:10]
            LOG.info('Sample DB counts: %s', json.dumps(sample_items, ensure_ascii=False))

        out_per_hit_csv = args.out_csv.replace('.csv', '_per_hit.csv')
        write_per_hit_csv(hits_rows, db_counts_enriched, out_per_hit_csv)
    except Exception as e:
        LOG.warning('Error during per-hit DB count enrichment or CSV writing: %s', e)
def fetch_hits_raw(es_url: str, es_index: str, query: Dict[str, Any], auth: Optional[Tuple[str, str]] = None, group_field: str = "extra.path", duration_field: str = "extra.execution_time", timestamp_field: str = "@timestamp", use_scroll: bool = False, scroll_ttl: str = "2m", scroll_size: int = 500, max_hits: int = 0):
    """Fetch raw hits (paginated with scroll if requested) and return list of rows (dicts).
    Each row contains: group_key, execution_time, timestamp, n_sources, header dict
    """
    rows = []
    # reuse logic from fetch_es_buckets for scrolling
    search_url = f"{es_url.rstrip('/')}/{es_index}/_search"
    headers = {"Content-Type": "application/json"}
    if use_scroll:
        payload = dict(query)
        # If user asked for a max_hits, request at most that many in the first page
        initial_size = scroll_size if not max_hits else min(scroll_size, max_hits)
        payload['size'] = initial_size
        payload.pop('_es_group_regex', None)
        # track_total_hits cannot be disabled for scroll
        payload.pop('track_total_hits', None)
        url = f"{es_url.rstrip('/')}/{es_index}/_search?scroll={scroll_ttl}"
        resp = safe_post(url, headers, payload, auth=auth, timeout=120)
        # resp = requests.post(url, headers=headers, data=json.dumps(payload), auth=auth, timeout=120)
        resp.raise_for_status()
        data_all = resp.json()
        sid = data_all.get('_scroll_id')
        page_hits = data_all.get('hits', {}).get('hits', [])
        hits = page_hits[:]
        total = len(hits)
        while sid and page_hits:
            if max_hits and total >= max_hits:
                break
            scroll_url = f"{es_url.rstrip('/')}/_search/scroll"
            body = {'scroll': scroll_ttl, 'scroll_id': sid}
            resp = safe_post(scroll_url, headers, body, auth=auth, timeout=120)
            # resp = requests.post(scroll_url, headers=headers, data=json.dumps(body), auth=auth, timeout=120)
            # resp.raise_for_status()
            data_all = resp.json()
            sid = data_all.get('_scroll_id')
            page_hits = data_all.get('hits', {}).get('hits', [])
            hits.extend(page_hits)
            total = len(hits)
    else:
        # non-scroll: avoid sending internal key
        if isinstance(query, dict):
            payload = dict(query)
            payload.pop('_es_group_regex', None)
            payload.pop('track_total_hits', None)
        else:
            payload = query
        resp = safe_post(search_url, headers, payload, auth=auth, timeout=60)
        # resp = requests.post(search_url, headers=headers, data=json.dumps(payload), auth=auth, timeout=60)
        hits = resp.json().get('hits', {}).get('hits', [])

    # Process hits into rows
    for h in hits:
        rec_fields = h.get('fields') or {}
        src = h.get('_source') or {}

        def extract(field_name: str):
            if field_name in rec_fields:
                val = rec_fields.get(field_name)
                if isinstance(val, list):
                    return val[0]
                return val
            kw = f"{field_name}.keyword"
            if kw in rec_fields:
                v = rec_fields.get(kw)
                if isinstance(v, list):
                    return v[0]
                return v
            if field_name in src:
                return src.get(field_name)
            parts = field_name.split('.')
            cur = src
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    cur = None
                    break
            return cur

        group_key = extract(group_field) or 'UNDEFINED'
        if isinstance(group_key, list):
            group_key = group_key[0]
        # apply regex if in query payload
        es_group_regex = query.get('_es_group_regex') if isinstance(query, dict) else None
        if es_group_regex and isinstance(group_key, (str, bytes)):
            try:
                m = re.search(es_group_regex, str(group_key))
                if m:
                    g = next((gg for gg in m.groups() if gg), None)
                    if g:
                        group_key = g
            except re.error:
                pass

        dur = extract(duration_field)
        ts = extract(timestamp_field) or extract('@timestamp')

        # Extract system info fields
        sys_info = {
            'gpu_name': extract('extra.system_info.gpu.name'),
            'gpu_driver': extract('extra.system_info.gpu.driver_version'),
            'gpu_id': extract('extra.system_info.gpu.id'),
            'gpu_load': extract('extra.system_info.gpu.load'),
            'gpu_mem_free': extract('extra.system_info.gpu.memory_free'),
            'gpu_mem_total': extract('extra.system_info.gpu.memory_total'),
            'gpu_mem_used': extract('extra.system_info.gpu.memory_used'),
            'gpu_temp': extract('extra.system_info.gpu.temperature'),
            'arch': extract('extra.system_info.architecture'),
            'cupy_ver': extract('extra.system_info.cupy_version'),
            'kernel': extract('extra.system_info.kernel_version'),
            'os': extract('extra.system_info.os'),
            'os_ver': extract('extra.system_info.os_version'),
            'processor': extract('extra.system_info.processor'),
            'python_ver': extract('extra.system_info.python_version')
        }

        # Parse extra.return_value or extra.arguments to extract header fields and n_rows
        rv = None
        if 'extra.return_value' in rec_fields:
            v = rec_fields.get('extra.return_value')
            if isinstance(v, list):
                rv = v[0]
            else:
                rv = v
        elif 'extra.return_value' in src:
            rv = src.get('extra.return_value')

        if not rv:
            if 'extra.arguments' in rec_fields:
                av = rec_fields.get('extra.arguments')
                rv = av[0] if isinstance(av, list) else av
            elif 'extra.arguments' in src:
                rv = src.get('extra.arguments')

        n_sources = None
        header_vals: Dict[str, Any] = {}
        if rv and isinstance(rv, str):
            parsed_n, parsed_header = _parse_header_and_nrows_from_text(rv)
            if parsed_n is not None:
                n_sources = parsed_n
            if parsed_header:
                header_vals.update(parsed_header)

        # Prefer OBLINEID over OBID when choosing group_key
        if header_vals:
            prefer_key = None
            for candidate in ('oblineid', 'obid'):
                if header_vals.get(candidate) is not None:
                    prefer_key = header_vals.get(candidate)
                    break
            if prefer_key is not None:
                try:
                    group_key = str(int(prefer_key))
                except Exception:
                    group_key = str(prefer_key)

        # Build row with normalized header fields
        row = {
            'group_key': group_key,
            'execution_time': dur,
            'timestamp': ts,
            'n_sources_detected': n_sources,
            'header': header_vals,
            'raw_return': rv,
            'sys_info': sys_info
        }
        # As fallback, if no normalized header fields found, include raw rec_fields keys
        if not header_vals:
            for k, v in rec_fields.items():
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                row['header'][k] = v

        rows.append(row)

    return rows

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        LOG.exception('Fatal error: %s', e)
        sys.exit(1)
