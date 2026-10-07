"""QA3 (PLAN.md Sec 4): Explainability. Every scored payment gets 1-3 reason
codes as plain text, computed within a tight latency budget."""
import time

import numpy as np



def _sample_feature_row() -> dict:
    return {
        "amount": 9500.0, "log_amount": np.log1p(9500.0), "is_transfer": 1.0,
        "oldbalance_orig": 9500.0, "amount_to_balance_ratio": 1.0, "drains_account": 1.0,
        "orig_zero_balance": 0.0, "oldbalance_dest": 0.0, "dest_zero_balance": 1.0,
        "hour_of_day": 3.0, "is_night": 1.0, "dest_txn_count_24h": 8.0, "dest_amount_sum_24h": 60000.0,
    }


def test_reason_codes_are_non_empty_and_bounded(explainer):
    reasons = explainer.explain(_sample_feature_row())
    assert 1 <= len(reasons) <= 3
    assert all(isinstance(r, str) and len(r) > 0 for r in reasons)


def test_explanation_overhead_within_budget(explainer):
    row = _sample_feature_row()
    # warm up (first SHAP call pays a one-off JIT-ish cost)
    explainer.explain(row)
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        explainer.explain(row)
        samples.append((time.perf_counter() - t0) * 1000)
    p95 = sorted(samples)[int(0.95 * len(samples)) - 1]
    assert p95 <= 10.0, f"explain() p95={p95:.2f}ms exceeds the 10ms QA3 budget"


def test_reasons_reference_known_feature_semantics(explainer):
    """A clearly draining, fresh-receiver payment should mention balance or
    draining in its top reason, not an unrelated weak signal."""
    reasons = explainer.explain(_sample_feature_row())
    joined = " ".join(reasons).lower()
    assert any(kw in joined for kw in ("balance", "empties", "drain"))
