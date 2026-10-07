"""
services/scoring/main.py
----------------------------------------------------------------
scoring-service :8001 — ML component.

Loads the champion model from the MLflow Model Registry at startup
(FR10), serves /score over the pipe-and-filter pipeline (pipeline.py),
and exposes /health for the triage-api gateway's heartbeat tactic (FR9).
----------------------------------------------------------------
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow.lightgbm
import redis
from fastapi import FastAPI, HTTPException

from common.config import settings
from common.feature_store import RedisVelocityStore
from common.features import FEATURE_NAMES
from common.logger import Timer, get_logger
from common.schemas import PaymentRequest, ScoreResult
from services.scoring.explain import ReasonCodeExplainer
from services.scoring.pipeline import score_payment

logger = get_logger(__name__)


class ModelStore:
    model = None
    explainer: ReasonCodeExplainer | None = None
    model_version: str = "unknown"
    loaded_at: datetime | None = None


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.model_name}@{settings.model_alias}"
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
        store.model = mlflow.lightgbm.load_model(model_uri)
        store.explainer = ReasonCodeExplainer(store.model)
        store.model_version = mv.version
        store.loaded_at = datetime.now(timezone.utc)
        logger.info("Model loaded", extra={"model_version": store.model_version})
    except Exception as exc:
        logger.error("Failed to load model from registry", extra={"error": str(exc)})
    yield
    logger.info("scoring-service shutting down")


app = FastAPI(title="Fraud Scoring Service", version="1.0.0", lifespan=lifespan)
redis_client = redis.from_url(settings.redis_url)
velocity_store = RedisVelocityStore(redis_client)


@app.get("/health", tags=["ops"])
def health():
    """Heartbeat endpoint — triage-api polls this to decide whether to fall
    back to rules-only mode (FR9)."""
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_name": settings.model_name,
        "model_version": store.model_version,
        "loaded_at": store.loaded_at,
    }


@app.post("/score", response_model=ScoreResult, tags=["inference"])
def score(payment: PaymentRequest):
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    with Timer() as t:
        try:
            result = score_payment(
                payment, velocity_store, store.model, store.explainer,
                FEATURE_NAMES, settings.model_name, store.model_version,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Scoring failed", extra={"txn_id": payment.txn_id, "error": str(exc)})
            raise HTTPException(status_code=500, detail="Scoring failed")
    logger.info(
        "Payment scored",
        extra={"txn_id": payment.txn_id, "risk_score": result.risk_score, "latency_ms": t.elapsed_ms},
    )
    return result
