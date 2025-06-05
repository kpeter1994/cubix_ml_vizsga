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
        'task': 'tasks.process_feed',       # a feladat neve, amit a tasks.py-ban definiálsz
        'schedule': crontab(minute='*/10'),   # 5 percenként
    },
    'run-every-week': {
        'task': 'tasks.process_train',       # a feladat neve, amit a tasks.py-ban definiálsz
        'schedule': crontab(minute='*/1'),   # 5 percenként
    },
}
app.conf.timezone = 'Europe/Budapest'           # időzóna beállítás (opcionális, de jó gyakorlat)