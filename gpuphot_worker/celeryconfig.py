# Celery Configuration for gpuphot
# You can set these values in your .env file and reference them in docker-compose.yml
# for the gpuphot_worker service using environment variables

# Broker settings
# CELERY_BROKER_URL = 'redis://redis:6379/0'
# CELERY_RESULT_BACKEND = 'redis://redis:6379/0'

# Task serialization format
# CELERY_TASK_SERIALIZER = 'json'

# Result serialization format
# CELERY_RESULT_SERIALIZER = 'json'

# List of accepted content types
# CELERY_ACCEPT_CONTENT = ['json']

# Time zone settings
# CELERY_TIMEZONE = 'UTC'

# Task routing
# CELERY_TASK_ROUTES = {
#     'tasks.process_image': {'queue': 'image_processing'},
#     'tasks.calibrate_image': {'queue': 'calibration'}
# }

# Task execution settings
# CELERY_TASK_TIME_LIMIT = 3600  # 1 hour
# CELERY_TASK_SOFT_TIME_LIMIT = 1800  # 30 minutes

# Worker settings
# CELERYD_MAX_TASKS_PER_CHILD = 10
# CELERYD_PREFETCH_MULTIPLIER = 1

# Beat settings (for periodic tasks)
# CELERYBEAT_SCHEDULE = {
#     'daily-dark-frame-calibration': {
#         'task': 'tasks.calibrate_dark_frames',
#         'schedule': crontab(hour=2, minute=0),  # Run at 2:00 AM every day
#     },
#     'hourly-image-processing': {
#         'task': 'tasks.process_new_images',
#         'schedule': crontab(minute=0),  # Run every hour
#     }
# }

# Logging
# CELERYD_LOG_FORMAT = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'

# To use these settings:
#
# 1. Add the desired variables to your .env file:
#    CELERY_BROKER_URL=redis://redis:6379/0
#    CELERY_TIMEZONE=UTC
#
# 2. Reference them in your docker-compose.yml for the gpuphot_worker service:
#    gpuphot_worker:
#      environment:
#        - CELERY_BROKER_URL=${CELERY_BROKER_URL}
#        - CELERY_TIMEZONE=${CELERY_TIMEZONE}
#
# This setup allows for flexible configuration management, enabling easy
# adjustments to Celery settings through environment variables.
#
# For more advanced configurations, you can set any top-level Celery key
# using an environment variable prefixed with NEW_CELERY_ followed by
# the config key name in lowercase. The value must be valid YAML or JSON.
#
# Example:
# export NEW_CELERY_task_routes='{"tasks.process_large_image": {"queue": "high_memory"}}'
#
# Remember to set CELERY_CONFIG_MODULE=celeryconfig in your
# environment to enable Celery to read these configurations.