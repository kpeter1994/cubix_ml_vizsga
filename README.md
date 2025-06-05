
Futtatás

    uvicorn books:app --reload --port 8000


A projekt célja egy hírgyűjtő alkalmazáshoz készíteni egy algoritmust, ami képes a hír címéből és a leírásból megállapítani a hír kategóriáját.

A hírgyűjtő alkalmazás RSS feedek segítségével gyűjti a híreket, de a hírek kategorizálása nem egységes. Azt a problémát hivatott megoldani a kategorizáló program.

A modell tanításához egy kb 26 000 sorból álló összegyűjtött adatbázist használok, de a cél, hogy mlops segítségével folyamatosan frissüljön az algoritmus, így több adattal egyre jobb minőségű prediktort tudunk készíteni.

## main.py

Ez a fájl egy FastAPI alkalmazást definiál: inicializálja a PredictorService-t és ArticleService-t, majd két végpontot hoz létre. A GET / lekérdezi az adatbázisból a legutóbbi 100 cikket, rendezve megjelenési idő szerint, a POST / pedig a bejövő Pydantic-modell alapján létrehoz egy új cikket az adatbázisban (internálisan meghívva az ArticleService.create_article-t, ahol opcionálisan prediktorral számolja a kategóriát).

## ArticleService 

Az ArticleService beolvassa egy RSS/Atom feedet, kinyeri belőle a cikkek adatait (cím, link, összefoglaló, kategória, megjelenési idő), majd ezeket elmenti az adatbázisba. Opcionálisan egy prediktor szolgáltatással is kiszámolja a cikkek előrejelzett kategóriáját.

## PredictorService 

A PredictorService a cikkek szövegét előkészíti (szólemmaszintre bontja és eltávolítja az ékezeteket), majd ebből TF-IDF vektort képez, és kétféle modellt képes tanítani (XGBoost és neurális háló). A tanítás után a legjobb hálózati modellt elmenti, valamint betölti a vektorizálót és a címkekódolót, hogy új szövegek esetén előrejelzést tudjon adni: a predict metódus visszaadja a cikk kategóriáját.


## tasks.py

process_feed feladata, hogy felsorolt RSS-feed URL-ekről sorban letöltse a cikkeket: minden URL-re létrehoz egy ArticleService példányt a PredictorService-szel, majd meghívja create_articles metódust, így az új bejegyzéseket az adatbázisba menti és kategorizálja.

process_train pedig végigmegy az adatbázisban még nem lemmatizált cikkeken, létrehozza a lemmatizált szöveget a lemmatize_data_to_db metódussal, majd elindítja a neurális hálózat tanítását (train_neural_network), és menti a legjobb modellt és a vektorizálót.


## celery_app.py

Ez a fájl definiálja a Celery alkalmazást és annak beállításait: megadja a Redis broker és backend címét, betölti az app.tasks.tasks modulban lévő feladatokat, beállítja a beat_schedule-et két időzített feladatra (process_feed ötpercenként és process_train minden hétfőn éjfélkor), valamint beállítja az időzónát Europe/Budapest-re.

### Indítás

    celery -A celery_app worker --loglevel=info --pool=solo

    celery -A celery_app beat --loglevel=info

































