from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

app = FastAPI()


@app.get("/")
def predict():
    return {"message": "Welcome to the FastAPI application!"}