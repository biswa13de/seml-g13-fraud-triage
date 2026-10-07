"""
services/scoring/pipeline.py
----------------------------------------------------------------
The Pipe-and-Filter Pattern in code (Session 4 slide 31):

    validate_payment -> load_velocity -> extract_features -> predict -> explain

Each stage is a small, independently testable function. Side effects
(reading Redis, calling SHAP) are isolated to one stage each, so the
pipeline itself stays a pure composition that is easy to unit test with
fakes.
----------------------------------------------------------------
"""
from common.feature_store import VelocityStore
from common.features import compute_features
from common.schemas import PaymentRequest, ScoreResult
from services.scoring.explain import ReasonCodeExplainer


def validate_payment(payment: PaymentRequest) -> PaymentRequest:
    """Pydantic already enforces types and ranges; this stage adds the one
    business rule too dynamic for a schema: TRANSFER/CASH_OUT only, since
    the model was trained exclusively on those (see training/build_features.py)."""
    if payment.type not in ("TRANSFER", "CASH_OUT"):
        raise ValueError(f"Scoring not supported for payment type {payment.type}")
    return payment


def load_velocity(payment: PaymentRequest, store: VelocityStore):
    return store.get(payment.name_dest, payment.step)


def extract_features(payment: PaymentRequest, velocity) -> dict:
    txn = {
        "amount": payment.amount,
        "oldbalanceOrg": payment.oldbalance_orig,
        "oldbalanceDest": payment.oldbalance_dest,
        "type": payment.type.value,
        "step": payment.step,
    }
    return compute_features(txn, velocity)


def predict(feature_row: dict, model, feature_names: list[str]) -> float:
    x = [[feature_row[name] for name in feature_names]]
    return float(model.predict_proba(x)[0][1])


def explain(feature_row: dict, explainer: ReasonCodeExplainer) -> list[str]:
    return explainer.explain(feature_row)


def score_payment(
    payment: PaymentRequest, store: VelocityStore, model, explainer: ReasonCodeExplainer,
    feature_names: list[str], model_name: str, model_version: str,
) -> ScoreResult:
    """Composes the full pipe-and-filter chain for one payment."""
    payment = validate_payment(payment)
    velocity = load_velocity(payment, store)
    features = extract_features(payment, velocity)
    risk_score = predict(features, model, feature_names)
    reason_codes = explain(features, explainer)
    return ScoreResult(
        txn_id=payment.txn_id,
        risk_score=risk_score,
        reason_codes=reason_codes,
        model_name=model_name,
        model_version=model_version,
    )
