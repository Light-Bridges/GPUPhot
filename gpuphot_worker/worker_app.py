from celery import Celery
from celery.schedules import crontab

app = Celery('gpuphot_worker')
app.config_from_object('gpuphot_worker.celeryconfig')
app.autodiscover_tasks(['gpuphot_worker.tasks'])
# # Configura la tarea update_tasks para que se ejecute cada minuto
# app.conf.beat_schedule['update_tasks'] = {
#     'task': 'gpuphot_worker.tasks.update_tasks',  # Reemplaza con el path correcto de tu tarea update_tasks
#     'schedule': crontab(minute='*'),  # Cada minuto
# }
if __name__ == '__main__':
    app.start()
