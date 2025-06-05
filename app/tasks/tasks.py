from app.services.PredictorService import PredictorService
from celery_app import app
from app.services.ArticleService import ArticleService


@app.task(name='tasks.process_feed')
def process_feed():
    feed_urls = [
        "https://www.portfolio.hu/rss/all.xml",
        "https://index.hu/24ora/rss?&rovatkeres=osszes",
        "https://mandiner.hu/rss",
        "https://24.hu/feed/",
        "https://telex.hu/rss",
        "https://hvg.hu/rss",
        "https://www.vg.hu/feed",
        "https://hirtv.hu/rss",
        "https://www.szeretlekmagyarorszag.hu/feed/",
        "https://blikkruzs.blikk.hu/rss",
        "https://nepszava.hu/rss",
        "https://magyarnemzet.hu/feed",
        "https://www.blikk.hu/rss",
        "https://www.borsonline.hu/rss"
    ]

    predictor_service = PredictorService()

    for url in feed_urls:
        article_service = ArticleService(url, predictor_service)
        article_service.create_articles()


@app.task(name='tasks.process_train')
def process_train():
    predictor_service = PredictorService()
    predictor_service.lemmatize_data_to_db()
    predictor_service.train_neural_network()