from sqlalchemy import Column, Integer, String, DateTime
from app.db.database import Base

class Articles(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    summary = Column(String)
    lemmatized_text = Column(String, nullable=True)
    link = Column(String)
    category = Column(String)
    predicted_category = Column(String, nullable=True)
    published = Column(DateTime)
