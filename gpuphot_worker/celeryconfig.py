# SPDX-License-Identifier: MIT
"""
Celery configuration for the gpuphot_worker service.

This module defines sensible defaults for Celery and supports overriding any
configuration using environment variables prefixed with "CELERY_". Environment
variables are converted to appropriate Python types (bool, int, float, list)
when possible.

Examples (in docker-compose.yml):
  - CELERY_BROKER_URL=redis://redis:6379/0
  - CELERY_RESULT_BACKEND=redis://redis:6379/0
  - CELERY_TASK_SERIALIZER=json
  - CELERY_RESULT_SERIALIZER=json
  - CELERY_ACCEPT_CONTENT=json
  - CELERY_TIMEZONE=UTC
"""

import os

# Celery Configuration defaults for gpuphot
result_extended = True
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
result_expires = 24 * 3600
worker_max_tasks_per_child = 1
beat_scheduler = 'redbeat.RedBeatScheduler'
redbeat_redis_url = 'redis://redis:6379/1'
# Uncomment to limit memory per child worker (example value in KB)
# worker_max_memory_per_child = 500000


# Override settings from environment variables with prefix CELERY_
for key, value in os.environ.items():
    if key.startswith('CELERY_'):
        key = key[7:].lower()
        # Convert string values to appropriate Python types
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit() and value.count('.') == 1:
            value = float(value)
        elif ',' in value:
            # Comma-separated lists become Python lists of stripped strings
            value = [v.strip() for v in value.split(',')]
        globals()[key] = value

# This setup allows flexible configuration via environment variables for
# containerized deployments. Values are placed in module globals so Celery
# can import them via app.config_from_object('gpuphot_worker.celeryconfig').
