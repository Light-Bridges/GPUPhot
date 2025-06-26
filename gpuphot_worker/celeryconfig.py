import os

# Celery Configuration for gpuphot
# You can set these values in your .env file and reference them in docker-compose.yml
# for the gpuphot_worker service using environment variables

result_extended = True
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
result_expires = 24 * 3600
worker_max_tasks_per_child = 1
beat_scheduler = 'redbeat.RedBeatScheduler'
redbeat_redis_url = 'redis://redis:6379/1'
# worker_max_memory_per_child = 500000

# Override settings with environment variables
for key, value in os.environ.items():
    if key.startswith('CELERY_'):
        key = key[7:].lower()
        # Convert string values to appropriate types
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit() and value.count('.') == 1:
            value = float(value)
        elif ',' in value:
            value = [v.strip() for v in value.split(',')]
        globals()[key] = value

# This setup allows for flexible configuration management, enabling easy
# adjustments to Celery settings through environment variables.
#
# Example usage in docker-compose.yml:
#   environment:
#     - CELERY_BROKER_URL=redis://redis:6379/0
#     - CELERY_RESULT_BACKEND=redis://redis:6379/0
#     - CELERY_TASK_SERIALIZER=json
#     - CELERY_RESULT_SERIALIZER=json
#     - CELERY_ACCEPT_CONTENT=json
#     - CELERY_TIMEZONE=UTC
#     - CELERY_TASK_TIME_LIMIT=3600
#     - CELERY_TASK_SOFT_TIME_LIMIT=1800
#
