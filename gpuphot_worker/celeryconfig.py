# Celery Configuration
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
# CELERY_TASK_ROUTES = {'tasks.email': {'queue': 'email'}}

# Task execution settings
# CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes
# CELERY_TASK_SOFT_TIME_LIMIT = 15 * 60  # 15 minutes

# Worker settings
# CELERYD_MAX_TASKS_PER_CHILD = 100
# CELERYD_PREFETCH_MULTIPLIER = 4

# Beat settings (for periodic tasks)
# CELERYBEAT_SCHEDULE = {
#     'add-every-30-seconds': {
#         'task': 'tasks.add',
#         'schedule': 30.0,
#         'args': (16, 16)
#     },
# }

# Logging
# CELERYD_LOG_FORMAT = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'

# Celery Configuration
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
# CELERY_TASK_ROUTES = {'tasks.email': {'queue': 'email'}}

# Task execution settings
# CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes
# CELERY_TASK_SOFT_TIME_LIMIT = 15 * 60  # 15 minutes

# Worker settings
# CELERYD_MAX_TASKS_PER_CHILD = 100
# CELERYD_PREFETCH_MULTIPLIER = 4

# Beat settings (for periodic tasks)
# CELERYBEAT_SCHEDULE = {
#     'add-every-30-seconds': {
#         'task': 'tasks.add',
#         'schedule': 30.0,
#         'args': (16, 16)
#     },
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
# export NEW_CELERY_broker_transport_options='{"visibility_timeout": 36000}'
#
# Remember to set CELERY_CONFIG_MODULE=new_celery_config.as_module in your
# environment to enable Celery to read these custom environment variables.