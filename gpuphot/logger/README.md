# gpuphot.logger

This package provides hierarchical and structured logging utilities used by
GPUPhot workers and tools. It focuses on producing readable nested logs on
console and, when enabled, structured telemetry to external systems such as
Logstash.

Purpose
-------
- Provide a singleton logger factory with a readable, indented console
  formatter for nested or recursive operations.
- Offer a lightweight handler (`NotifyingHandler`) to forward structured
  log entries to callbacks (e.g., web UIs, monitoring dashboards).
- Collect relevant system metadata (GPU, Python, OS and Git commit) to
  include in structured log payloads for traceability.
- Supply a decorator (`hierarchical_debug`) to instrument functions with
  entry/exit logs, argument summaries and execution time, including
  automatic error logging.

Key modules and symbols
-----------------------
- `hierarchical_logging.py` — main implementation file. Exposes:
  - `IndentFormatter` — logging.Formatter that supports per-thread indentation.
  - `NotifyingHandler(callback)` — handler that forwards structured events to a callback.
  - `SystemInfo` — singleton that collects and caches system metadata.
  - `hierarchical_debug(logger_name)` — decorator factory to wrap functions and
    automatically emit structured start/end/exception logs.
  - `setup_logstash_handler(logger)` — helper to configure an asynchronous
    Logstash handler when `LOGSTASH_LOGGING` is enabled via environment.
  - `setup_logger(name)` — convenience function returning the package's
    singleton logger instance.

Environment variables
---------------------
The module reads configuration from environment variables (and `.env` if
present). Important keys:

- `GPUPHOT_LOG_LEVEL` — logging level override. Typical values: `DEBUG`,
  `INFO`, `ERROR`.
- `LOGSTASH_LOGGING` — if set to `true` (case-insensitive), the module will
  attempt to configure a Logstash handler (`logstash_async`).
- `LOGSTASH_HOST` — host for Logstash connection (default: `localhost`).
- `LOGSTASH_PORT` — port for Logstash connection (default: `5000`).
- `GPUPHOT_ENVIRONMENT` — added to structured Logstash payloads as `environment`.

Usage examples
--------------
Basic logger retrieval and usage:

```python
from gpuphot.logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)
logger.info('Starting worker')
```

Instrumenting a function with automatic structured logs:

```python
from gpuphot.logger.hierarchical_logging import hierarchical_debug

@hierarchical_debug('gpuphot')
def process_image(path):
    # function body
    pass
```

Forwarding logs to a callback (web UI example):

```python
from gpuphot.logger.hierarchical_logging import NotifyingHandler, setup_logger

def ui_callback(entry):
    # entry is a dict with keys: message, level, timestamp, logger_name, etc.
    send_to_websocket(entry)

logger = setup_logger('gpuphot')
handler = NotifyingHandler(ui_callback)
logger.addHandler(handler)
```

Operational notes
-----------------
- The module attempts to import `logstash_async` and will silently skip Logstash
  configuration if the dependency is not available. This design makes the
  module safe to import on developer machines without external telemetry.
- `SystemInfo` tries to read the local Git `HEAD` to include commit information
  in logs; this will gracefully return `None` when a `.git` directory is not
  present (for example in some packaging or deployment scenarios).
- The indentation feature in `IndentFormatter` is thread-local to avoid
  cross-thread interference when many tasks are processed concurrently.
- Avoid changing runtime logic in this package during documentation-only
  passes.

Developer notes and extension points
------------------------------------
- To enable Logstash structured logging in development or production, install
  `logstash_async` and set `LOGSTASH_LOGGING=true` and the connection
  variables in the environment.
- The `NotifyingHandler` is intentionally minimal: the callback is responsible
  for delivery semantics (backpressure, batching, error handling).
- If you want to enrich log payloads with additional context, prefer adding
  keys to the `extra` parameter in logger calls (e.g. `logger.debug("msg",
  extra={'user': user_id})`) to keep handlers generic.

Testing and validation
----------------------
- Unit tests for logging behavior should mock external dependencies (GPUtil,
  logstash_async) and assert that `hierarchical_debug` logs start/finish and
  exceptions with expected extras.

- Quick manual test: run the module as a script and verify `example_function`
  output (it exercises the decorator and handler setup path).

Contact and context
-------------------
If you need clarification about logging payload fields or want a sample
Logstash dashboard mapping, ask and I will prepare an example mapping and a
minimal integration test.
