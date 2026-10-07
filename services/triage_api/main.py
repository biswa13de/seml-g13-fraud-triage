"""
services/triage_api/main.py
----------------------------------------------------------------
triage-api :8000 — Gateway + Decision Orchestrator (non-ML component).

Request flow (FR1, FR2, FR3, FR4, FR9):
  1. API-key auth
  2. Pydantic validation (schemas.PaymentRequest)
  3. Hard rules (rules.py)                         -- deterministic
  4. Heartbeat check -> call scoring-service, or
     fall back to rules-only if unhealthy/timeout   -- FR9 fallback tactic
  5. Policy: risk score -> decision (policy.py)
  6. Publish payment.decided to Redis Streams       -- Event-Driven pattern
  7. Respond
----------------------------------------------------------------
"""
from contextlib import asynccontextmanager

import httpx
import redis
from fastapi import Depends, FastAPI, Header, HTTPException

from common.config import settings
from common.events import publish
from common.logger import Timer, get_logger
from common.schemas import Decision, PaymentDecidedEvent, PaymentRequest, TriageDecision
from services.triage_api.heartbeat import Heartbeat
from services.triage_api.policy import TriagePolicy
from services.triage_api.rules import check_hard_rules

logger = get_logger(__name__)

API_KEY = "demo-key-g13"  # for the assignment demo only; a real deployment uses a secret store

redis_client = redis.from_url(settings.redis_url)
policy = TriagePolicy(settings.thresholds_path)
heartbeat = Heartbeat(
    url=f"{settings.scoring_service_url}/health",
    interval_s=settings.heartbeat_interval_seconds,
    failure_threshold=settings.heartbeat_failure_threshold,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    heartbeat.start()
    logger.info("triage-api started")
    yield
    heartbeat.stop()


app = FastAPI(title="Fraud Triage Gateway", version="1.0.0", lifespan=lifespan)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "healthy", "scoring_service_healthy": heartbeat.is_healthy}


def _call_scorer(payment: PaymentRequest) -> dict | None:
    try:
        resp = httpx.post(
            f"{settings.scoring_service_url}/score",
            json=payment.model_dump(by_alias=True, mode="json"),
            timeout=settings.scoring_timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Scorer call failed, falling back", extra={"error": str(exc)})
        return None


@app.post("/v1/payments/triage", response_model=TriageDecision, tags=["triage"])
def triage(payment: PaymentRequest, _: None = Depends(require_api_key)):
    with Timer() as t:
        rule_hit = check_hard_rules(payment)
        if rule_hit is not None:
            decision, rule_name = rule_hit
            score = None
        else:
            decision = None
            rule_name = None
            score = _call_scorer(payment) if heartbeat.is_healthy else None

    # Built AFTER the `with` block closes, so t.elapsed_ms (set in
    # Timer.__exit__) reflects the full decision latency, not a partial read.
    if rule_hit is not None:
        result = TriageDecision(
            txn_id=payment.txn_id, decision=decision, risk_score=1.0,
            reason_codes=[f"Hard rule triggered: {rule_name}"],
            model_version="n/a (rule)", degraded=False, rule_triggered=rule_name,
            latency_ms=t.elapsed_ms,
        )
    elif score is not None:
        decision = policy.decide(score["risk_score"])
        result = TriageDecision(
            txn_id=payment.txn_id, decision=decision, risk_score=score["risk_score"],
            reason_codes=score["reason_codes"], model_version=score["model_version"],
            degraded=False, latency_ms=t.elapsed_ms,
        )
    else:
        # Fallback (FR9): scorer down or unhealthy -> rules-only, never hang
        result = TriageDecision(
            txn_id=payment.txn_id, decision=Decision.STEP_UP, risk_score=0.5,
            reason_codes=["Scoring service unavailable; defaulting to step-up authentication"],
            model_version="n/a (degraded)", degraded=True, latency_ms=t.elapsed_ms,
        )

    _publish_decision(payment, result)
    logger.info("Payment triaged", extra={
        "txn_id": payment.txn_id, "decision": result.decision, "degraded": result.degraded,
        "latency_ms": result.latency_ms,
    })
    return result


def _publish_decision(payment: PaymentRequest, result: TriageDecision) -> None:
    event = PaymentDecidedEvent(
        txn_id=payment.txn_id, decision=result.decision, risk_score=result.risk_score,
        reason_codes=result.reason_codes, amount=payment.amount, name_orig=payment.name_orig,
        name_dest=payment.name_dest, step=payment.step, model_version=result.model_version,
        degraded=result.degraded,
    )
    try:
        publish(redis_client, settings.stream_payment_decided, event.model_dump(mode="json"))
    except Exception as exc:
        # Publishing is best-effort: a Redis blip must not fail the payment decision itself.
        logger.error("Failed to publish payment.decided", extra={"error": str(exc)})
