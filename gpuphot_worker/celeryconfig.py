# SPDX-License-Identifier: MIT
"""
Celery configuration for the gpuphot_worker service.

This module defines defaults for Celery and supports overriding any
configuration using environment variables prefixed with "CELERY_".

Environment variables are automatically converted to appropriate Python types:
- "true"/"false" (case-insensitive) -> bool
- digits -> int
- digits with one dot -> float
- comma-separated values -> list of strings

Common configuration keys expected in environment:
- CELERY_BROKER_URL: URL of the message broker (e.g., amqp://user:pass@host:port).
- CELERY_RESULT_BACKEND: URL of the result backend (e.g., redis://host:port/db).
- CELERY_Worker_CONCURRENCY: Number of concurrent worker processes.

This dynamic loading allows flexible deployment in Docker/Kubernetes environments
without modifying the code.
"""

import os

# --- Default Configuration ---

# Enable extended task result attributes (name, args, kwargs, etc.)
result_extended = True

# Serialization format for tasks and results (JSON is standard and secure)
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']

# Result expiration time in seconds (24 hours)
result_expires = 24 * 3600

# Restart worker process after executing this many tasks.
# This helps mitigate memory leaks in long-running worker processes,
# which is crucial for memory-intensive image processing tasks.
worker_max_tasks_per_child = 1

# Scheduler class for periodic tasks (RedBeat allows dynamic scheduling via Redis)
beat_scheduler = 'redbeat.RedBeatScheduler'
redbeat_redis_url = 'redis://redis:6379/1'

# Optional: Limit memory per child worker (in KB) to force restart if exceeded.
# worker_max_memory_per_child = 500000


# --- Dynamic Configuration Loading ---

# Iterate over all environment variables and override defaults if they start with CELERY_
for key, value in os.environ.items():
    if key.startswith('CELERY_'):
        # Remove prefix and convert to lowercase to match Celery config keys
        config_key = key[7:].lower()
        
        # Type conversion logic
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit() and value.count('.') == 1:
            value = float(value)
        elif ',' in value:
            # Convert comma-separated string to list
            value = [v.strip() for v in value.split(',')]
            
        # Set the configuration in the module's global scope
        globals()[config_key] = value
