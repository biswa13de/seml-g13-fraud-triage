"""FR9: when the scorer is unhealthy or unreachable, triage-api falls back
to a safe rules-only decision instead of hanging or erroring, and always
marks the response degraded=True so downstream systems and the analyst
console can tell the difference."""
from services.triage_api.heartbeat import Heartbeat
from services.triage_api.policy import TriagePolicy
from services.triage_api.rules import check_hard_rules
from common.schemas import Decision, PaymentRequest


def test_heartbeat_reports_unhealthy_when_target_unreachable():
    hb = Heartbeat(url="http://localhost:1/health", interval_s=60, failure_threshold=1)
    hb._check_once()
    assert hb.is_healthy is False


def test_hard_rule_blocks_amount_over_limit(monkeypatch):
    from common.config import settings
    monkeypatch.setattr(settings, "hard_block_amount", 1000.0)
    payment = PaymentRequest(
        txn_id="T1", step=1, type="TRANSFER", amount=5000.0,
        nameOrig="C1", oldbalanceOrg=10000.0, nameDest="C2", oldbalanceDest=0.0,
    )
    result = check_hard_rules(payment)
    assert result is not None
    decision, rule = result
    assert decision == Decision.BLOCK
    assert rule == "amount_exceeds_hard_limit"


def test_hard_rule_blocks_blocklisted_payee(monkeypatch):
    from common.config import settings
    monkeypatch.setattr(settings, "blocklisted_accounts", {"C-BAD"})
    payment = PaymentRequest(
        txn_id="T1", step=1, type="TRANSFER", amount=100.0,
        nameOrig="C1", oldbalanceOrg=1000.0, nameDest="C-BAD", oldbalanceDest=0.0,
    )
    decision, rule = check_hard_rules(payment)
    assert decision == Decision.BLOCK
    assert rule == "blocklisted_payee"


def test_policy_decides_allow_below_both_thresholds(tmp_path):
    thresholds_file = tmp_path / "thresholds.json"
    thresholds_file.write_text('{"t_stepup": 0.3, "t_block": 0.7}')
    policy = TriagePolicy(str(thresholds_file))
    assert policy.decide(0.1) == Decision.ALLOW
    assert policy.decide(0.5) == Decision.STEP_UP
    assert policy.decide(0.9) == Decision.BLOCK
