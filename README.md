    
    python -m venv venv

    venv\Scripts\activate

    pip install fastapi

Függőségek felsorolása

    pip freeze > requirements.txt

SQL Alchemy

    pip install sqlalchemy 

Setup

    python -m pip install setuptools
    pip install -r requirements.txt

Futtatás

    uvicorn books:app --reload --port 8000

## Celery

    celery -A celery_app worker --loglevel=info --pool=solo

    celery -A celery_app beat --loglevel=info

A projekt célja egy hírgyűjtő alkalmazáshoz készíteni egy algoritmust, ami képes a hír címéből és a leírásból megállapítani a hír kategóriáját.

A hírgyűjtő alkalmazás RSS feedek segítségével gyűjti a híreket, de a hírek kategorizálása nem egységes. Azt a problémát hivatott megoldani a kategorizáló program.

A modell tanításához egy kb 26 000 sorból álló összegyűjtött adatbázist használok, de a cél, hogy mlops segítségével folyamatosan frissüljön az algoritmus, így több adattal egyre jobb minőségű prediktort tudunk készíteni.
