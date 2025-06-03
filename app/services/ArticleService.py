import feedparser
from datetime import datetime, timezone
from urllib.parse import urlparse
from app.db.database import SessionLocal
import app.models as models


class ArticleService:
    def __init__(self, url):
        self.url = url
        self.feed = feedparser.parse(self.url)

    def get_category(self, entry):
        category = None
        if "tags" in entry and entry.tags:
            category = entry.tags[0].term

        parsed_url = urlparse(entry.link)
        path_parts = parsed_url.path.strip("/").split("/")
        if path_parts:
            category = path_parts[0]
        return category

    def parse(self):

        articles = []

        for entry in self.feed.entries:

            articles.append({
                "title": entry.title,
                "link": entry.link,
                "summary": entry.get("summary", ""),
                "category": self.get_category(entry),
                "published": datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                if entry.get("published_parsed") else datetime.now(timezone.utc),
            })

        return articles

    def create_articles(self):
        with SessionLocal() as db:
            for article in self.parse():
                exists = db.query(models.Articles).filter_by(link=article["link"]).first()
                if exists:
                    continue
                article_model = models.Articles()
                article_model.title = article["title"]
                article_model.link = article["link"]
                article_model.summary = article["summary"]
                article_model.category = article["category"]
                article_model.published = article["published"]
                db.add(article_model)
            db.commit()
        return "Hozzáadás sikeres"

