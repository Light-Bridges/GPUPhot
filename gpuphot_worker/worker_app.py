from celery import Celery

app = Celery('gpuphot_worker')
app.config_from_object('gpuphot_worker.celeryconfig')
app.autodiscover_tasks(['gpuphot_worker.tasks'])

if __name__ == '__main__':
    app.start()
