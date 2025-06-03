    
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

Futattás

    uvicorn books:app --reload --port 8000

## Celery

    celery -A celery_app worker --loglevel=info --pool=solo

    celery -A celery_app beat --loglevel=info

A projekt célja egy hírgyüjtő lakalmazáshoz készíteni egy algoritmust, amki képes a hír címéből és a leírásből megállapítani a hír kategoriáját.

A hírgyüjtő alakalamzás RRS feedek segítségével gyüjti a híreket, de a hírek kategorizálása nem egységes. Azt a probémát hívaott megoldani a kategrizáló program.

A model tanításához egy kb 26 000 sorból álló összegyűjtött adatbázist használok, de a cél, hogy mlops segítségével folyamatosan firssüljön az algoritmus, így több adattal egyre jobb minőságű prediktert tudunk készíteni.