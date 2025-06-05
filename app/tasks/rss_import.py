from app.services.ArticleService import ArticleService
from app.services.PredictorService import PredictorService


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
        article_service = ArticleService(url,predictor_service)
        article_service.create_articles()


process_feed()