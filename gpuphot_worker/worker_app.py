from celery import Celery

app = Celery('gpuphot_tasks')
app.config_from_object('gpuphot_worker.celeryconfig')

if __name__ == '__main__':
    app.start()