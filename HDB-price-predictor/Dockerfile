FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY model_artifacts/ ./model_artifacts/

EXPOSE 8000

# Render injects $PORT at runtime; shell form expands env vars
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
