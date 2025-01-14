from celery import Celery
from celery.schedules import crontab

app = Celery('gpuphot_worker')
app.config_from_object('gpuphot_worker.celeryconfig')
app.autodiscover_tasks(['gpuphot_worker.tasks'])

# Define the periodic task schedule for Celery Beat
app.conf.beat_schedule = {
    'process_prered_images_using_dummy_instrument': {
        'task': 'gpuphot_worker.tasks.process_directory_task',
        'schedule': crontab(hour='10', minute='0'),  # Executes daily at 10 AM

        # This task is an example of how to schedule automatic tasks in Celery.
        # In this example, it processes images in the 'prered' directory
        # that contain "dummy" in their filename, using the configuration
        # for the instrument 'dummy_instrument'. It is set to not reprocess
        # images that have already been processed.
        # Parameters passed to process_directory_task:
        # - path: 'prered' (the directory to search for images)
        # - filename: 'dummy' (specific filename filter)
        # - instrument_name: 'dummy_instrument' (the name of the instrument used for processing)
        # - exclude_pattern: None (no exclusion pattern applied)
        # - reprocess: False (do not reprocess images that have already been processed)
        'args': ('prered', 'dummy', 'dummy_instrument', None, False),  # Arguments passed to the task
    },
}

if __name__ == '__main__':
    app.start()
