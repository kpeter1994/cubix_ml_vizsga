FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN sed -i -E '/^pywin32==|^pywinpty==/d' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .


#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
