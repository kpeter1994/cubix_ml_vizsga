from celery import Celery
from celery.schedules import crontab


app = Celery(
    'tasks',
    broker="redis://localhost:6390",
    backend="redis://localhost:6390",
    include=['app.tasks.tasks']
)

app.conf.beat_schedule = {
    'run-every-5-minutes': {
        'task': 'tasks.process_feed',
        'schedule': crontab(minute='*/5'),
    },
    'run-every-week': {
        'task': 'tasks.process_train',
        'schedule': crontab(day_of_week='monday', hour=0, minute=0),
    },
}
app.conf.timezone = 'Europe/Budapest'