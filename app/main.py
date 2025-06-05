from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from app.db.database import SessionLocal
from sqlalchemy import desc, DateTime
from app import models
from datetime import datetime


from app.services.ArticleService import ArticleService
from app.services.PredictorService import PredictorService

app = FastAPI()
predictor_service = PredictorService()
article_service = ArticleService(None, predictor_service)

class Article(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=2048)
    link: str = Field(min_length=1, max_length=515)
    category: str = Field(min_length=1, max_length=515)
    published: datetime = Field(default_factory=datetime.now)


@app.get("/")
def get_articles():
    with SessionLocal() as db:
        articles = db.query(models.Articles) \
            .order_by(desc(models.Articles.published)) \
            .limit(100) \
            .all()
        return articles

@app.post("/")
def create_article(article: Article):
    try:
        data = article_service.create_article(article.dict(), commit=True)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
