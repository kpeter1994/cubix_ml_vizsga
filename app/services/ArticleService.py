import feedparser
from datetime import datetime, timezone
from urllib.parse import urlparse
from app.db.database import SessionLocal
import app.models as models
from bs4 import BeautifulSoup
from app.services.PredictorService import PredictorService


class ArticleService:
    def __init__(self, url, predictor_service: PredictorService = None):
        self.url = url
        self.predictor_service = predictor_service
        if self.url:
            self.feed = feedparser.parse(self.url)
        else:
            self.feed = None

    def get_category(self, entry):
        category = None
        parsed_url = urlparse(entry.link)
        path_parts = parsed_url.path.strip("/").split("/")
        if path_parts:
            category = path_parts[0]

        if "tags" in entry and entry.tags:
            category = entry.tags[0].term
        return category

    def parse(self):
        articles = []

        for entry in self.feed.entries:

            articles.append({
                "title": entry.title,
                "link": entry.link,
                "summary": BeautifulSoup(str(entry.get("summary", "")), 'html.parser').get_text(),
                "category": self.get_category(entry),
                "published": datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                if entry.get("published_parsed") else datetime.now(timezone.utc),
            })

        return articles
    def create_article(self, article, commit=True):
        with SessionLocal() as db:
            article_model = models.Articles()
            article_model.title = article["title"]
            article_model.link = article["link"]
            article_model.summary = article["summary"]
            article_model.category = article["category"]
            if self.predictor_service:
                article_model.predicted_category = self.predictor_service.predict(f'{article["title"]} {article["summary"]}')
            article_model.published = article["published"]
            db.add(article_model)
            if commit:
                db.commit()
                return {
                    "id": article_model.id,
                    "title": article_model.title,
                    "summary": article_model.summary,
                    "link": article_model.link,
                    "category": article_model.category,
                    "predicted_category": article_model.predicted_category,
                }

    def create_articles(self):
        with SessionLocal() as db:
            for article in self.parse():
                exists = db.query(models.Articles).filter_by(link=article["link"]).first()
                if exists:
                    continue
                self.create_article(article, commit=False)
            db.commit()
        return "Hozzáadás sikeres"




