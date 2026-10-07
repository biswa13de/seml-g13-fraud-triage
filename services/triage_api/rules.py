"""Deterministic hard-rule engine, applied BEFORE the ML model (FR2).

Kept as plain Python functions, not learned, so it is auditable and doubles
as the fallback decision-maker when scoring-service is unhealthy (FR9)."""
from common.config import settings
from common.schemas import Decision, PaymentRequest


def check_hard_rules(payment: PaymentRequest) -> tuple[Decision, str] | None:
    """Returns (decision, rule_name) if a hard rule fires, else None (fall
    through to the ML score)."""
    if payment.name_dest in settings.blocklisted_accounts:
        return Decision.BLOCK, "blocklisted_payee"
    if payment.amount > settings.hard_block_amount:
        return Decision.BLOCK, "amount_exceeds_hard_limit"
    return None
