from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from app.db.database import Base

class Articles(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    summary = Column(String(2048))
    lemmatized_text = Column(String(2048), nullable=True)
    link = Column(String(515))
    category = Column(String(515))
    predicted_category = Column(String(100), nullable=True)
    published = Column(DateTime)

class Training(Base):
    __tablename__ = "training"
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    training_date = Column(DateTime)
    accuracy = Column(Float)
    active = Column(Boolean, default=0)
