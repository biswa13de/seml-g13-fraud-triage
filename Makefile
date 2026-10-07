.PHONY: data train mlflow-up up down test load logs run-local

VENV := .venv/bin

data:
	$(VENV)/python data/download_paysim.py
	$(VENV)/python -m training.build_features

# Trains against the MLflow TRACKING SERVER (not a bare sqlite file), so the
# resulting runs/artifacts are resolvable by every container later. Start it
# first: `docker compose up -d mlflow` (also brought up by `make up`).
train:
	MLFLOW_TRACKING_URI=http://localhost:5000 $(VENV)/python -m training.train
	MLFLOW_TRACKING_URI=http://localhost:5000 $(VENV)/python -m training.select_thresholds
	MLFLOW_TRACKING_URI=http://localhost:5000 $(VENV)/python -m training.register

mlflow-up:
	docker compose up -d mlflow
	@echo "MLflow UI: http://localhost:5000  (wait a few seconds for healthy)"

up:
	docker compose up --build

down:
	docker compose down -v

# Model-quality/explainability/performance tests need the registered
# champion model, served by the `mlflow` container (`make mlflow-up` first).
# Data/contract/event/fallback tests need no infra at all.
test:
	MLFLOW_TRACKING_URI=http://localhost:5000 $(VENV)/python -m pytest tests/ -v

load:
	$(VENV)/python scripts/load_test.py

# No-Docker path: needs `brew install redis && redis-server &`, and the
# MLflow tracking server running locally first:
#   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db \
#     --artifacts-destination ./mlruns --serve-artifacts
run-local:
	REDIS_URL=redis://localhost:6379/0 MLFLOW_TRACKING_URI=http://localhost:5000 \
		$(VENV)/uvicorn services.scoring.main:app --port 8001 &
	sleep 2
	REDIS_URL=redis://localhost:6379/0 SCORING_SERVICE_URL=http://localhost:8001 \
		$(VENV)/uvicorn services.triage_api.main:app --port 8000 &
	REDIS_URL=redis://localhost:6379/0 $(VENV)/uvicorn services.case_service.main:app --port 8002 &
	REDIS_URL=redis://localhost:6379/0 $(VENV)/python -m services.feature_updater.main &
	REDIS_URL=redis://localhost:6379/0 MLFLOW_TRACKING_URI=http://localhost:5000 \
		$(VENV)/uvicorn services.monitor.main:app --port 8003 &
	wait
