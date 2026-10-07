# Shared base image for every Python service. The entrypoint command is
# overridden per service in docker-compose.yml.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY services/ ./services/
COPY training/ ./training/
COPY console/ ./console/
COPY data/thresholds.json data/split_summary.json data/features_valid.parquet ./data/

# mlflow.db / mlruns are NOT baked into the image: the `mlflow` compose
# service owns that state in its own named volumes (mlflow_db, mlflow_data),
# so every container resolves artifact paths against the SAME server,
# regardless of which host or CI runner built the image.

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

EXPOSE 8000 8001 8002 8003
