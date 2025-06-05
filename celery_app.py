from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv
import os

load_dotenv()

app = Celery(
    'tasks',
    broker=os.getenv("REDIS_ROUTER"),
    backend=os.getenv("REDIS_ROUTER"),
    include=['app.tasks.tasks']
)

app.conf.beat_schedule = {
    'run-every-5-minutes': {
        'task': 'tasks.process_feed',
        'schedule': crontab(minute='*/1'),
    },
    'run-every-week': {
        'task': 'tasks.process_train',
        'schedule': crontab(minute='*/1'),
    },
}
app.conf.timezone = 'Europe/Budapest'