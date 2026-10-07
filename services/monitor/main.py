"""
services/monitor/main.py
----------------------------------------------------------------
monitor-service :8003 — non-ML (ops) component.

Consumes payment.decided in a background thread to track the decision mix,
score distribution (for PSI drift, FR8) and a rolling latency sample, then
serves it all at GET /metrics for the console's Monitoring tab.
----------------------------------------------------------------
"""
import threading
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI

from common.config import settings
from common.events import consume_forever, redis_client_for_consumer
from common.logger import get_logger
from services.monitor.drift import psi

logger = get_logger(__name__)

WINDOW = 5000


class State:
    decision_counts = Counter()
    scores: deque = deque(maxlen=WINDOW)
    total = 0
    baseline_scores: np.ndarray | None = None


state = State()


def _load_baseline() -> np.ndarray | None:
    """Score the validation split once at startup with the champion model so
    PSI compares like with like: risk-score distribution vs risk-score
    distribution, not raw transaction amounts."""
    path = Path("data/features_valid.parquet")
    if not path.exists():
        return None
    import mlflow.lightgbm
    import pandas as pd

    from common.features import FEATURE_NAMES

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        model = mlflow.lightgbm.load_model(f"models:/{settings.model_name}@{settings.model_alias}")
    except Exception as exc:
        logger.warning("Could not load baseline model for PSI", extra={"error": str(exc)})
        return None
    valid = pd.read_parquet(path)
    return model.predict_proba(valid[FEATURE_NAMES])[:, 1]


def run_consumer() -> None:
    client = redis_client_for_consumer(settings.redis_url)

    def handle(msg_id, event: dict) -> None:
        state.total += 1
        state.decision_counts[event["decision"]] += 1
        state.scores.append(event["risk_score"])

    consume_forever(client, settings.stream_payment_decided, "monitor", "monitor-consumer-1", handle)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.baseline_scores = _load_baseline()
    thread = threading.Thread(target=run_consumer, daemon=True)
    thread.start()
    yield


app = FastAPI(title="Monitor Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "healthy"}


@app.get("/metrics", tags=["ops"])
def metrics():
    scores = np.array(state.scores) if state.scores else np.array([])
    drift = (
        psi(state.baseline_scores, scores)
        if state.baseline_scores is not None and len(scores) > 50
        else None
    )
    alert = drift is not None and drift > settings.psi_alert_threshold
    return {
        "total_decisions": state.total,
        "decision_mix": dict(state.decision_counts),
        "score_window_size": len(scores),
        "score_mean": float(scores.mean()) if len(scores) else None,
        "psi_drift": drift,
        "drift_alert": alert,
    }
