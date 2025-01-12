from celery import Celery
from celery.schedules import crontab

app = Celery('gpuphot_worker')
app.config_from_object('gpuphot_worker.celeryconfig')
app.autodiscover_tasks(['gpuphot_worker.tasks'])

# app.conf.beat_schedule['dummy_task'] = {
#     'task': 'gpuphot_worker.tasks.dummy_task',
#     'schedule': crontab(minute='*/5'),
#     'args': ('dummy_instrument',),
# }

app.conf.beat_schedule = {
    'dummy_task': {
        'task': 'gpuphot_worker.tasks.dummy_task',
        'schedule': crontab(minute='*/5'),
        'args': ('dummy_instrument',),
    },
    'dummy_task_2': {
        'task': 'gpuphot_worker.tasks.dummy_task',
        'schedule': crontab(hour='12'),
        'args': ('dummy_instrument',),
    },
}

if __name__ == '__main__':
    app.start()
