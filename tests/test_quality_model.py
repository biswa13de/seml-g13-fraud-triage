"""QA2 (PLAN.md Sec 4): Model accuracy on imbalanced fraud data.
Gate thresholds match training/register.py's quality gate — if these fail,
the registry script would have refused to promote the model too."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from common.features import FEATURE_NAMES
from training.train import recall_at_fpr

MIN_PR_AUC = 0.80
MIN_RECALL_AT_1PCT_FPR = 0.85


@pytest.fixture(scope="module")
def test_split():
    try:
        return pd.read_parquet("data/features_test.parquet")
    except FileNotFoundError:
        pytest.skip("features_test.parquet not built")


def test_pr_auc_meets_gate(champion_model, test_split):
    scores = champion_model.predict_proba(test_split[FEATURE_NAMES])[:, 1]
    pr_auc = average_precision_score(test_split["isFraud"], scores)
    assert pr_auc >= MIN_PR_AUC


def test_recall_at_1pct_fpr_meets_gate(champion_model, test_split):
    scores = champion_model.predict_proba(test_split[FEATURE_NAMES])[:, 1]
    recall = recall_at_fpr(test_split["isFraud"].to_numpy(), scores, target_fpr=0.01)
    assert recall >= MIN_RECALL_AT_1PCT_FPR


def test_directional_draining_account_raises_risk(champion_model):
    """A transfer that drains the sender's full balance to a zero-balance
    receiver must score at least as risky as an otherwise identical payment
    that leaves balances untouched (monotonicity sanity check)."""
    base = {
        "amount": 500.0, "log_amount": np.log1p(500.0), "is_transfer": 1.0,
        "oldbalance_orig": 5000.0, "amount_to_balance_ratio": 500.0 / 5001.0,
        "drains_account": 0.0, "orig_zero_balance": 0.0, "oldbalance_dest": 3000.0,
        "dest_zero_balance": 0.0, "hour_of_day": 14.0, "is_night": 0.0,
        "dest_txn_count_24h": 1.0, "dest_amount_sum_24h": 500.0,
    }
    risky = base | {
        "amount": 5000.0, "log_amount": np.log1p(5000.0), "amount_to_balance_ratio": 5000.0 / 5001.0,
        "drains_account": 1.0, "oldbalance_dest": 0.0, "dest_zero_balance": 1.0,
    }
    x_base = np.array([[base[f] for f in FEATURE_NAMES]])
    x_risky = np.array([[risky[f] for f in FEATURE_NAMES]])
    score_base = champion_model.predict_proba(x_base)[0][1]
    score_risky = champion_model.predict_proba(x_risky)[0][1]
    assert score_risky >= score_base


def test_invariant_to_sender_account_id():
    """The model never sees nameOrig/nameDest as a feature, so scoring must
    not depend on them (fairness-adjacent robustness check)."""
    assert "nameOrig" not in FEATURE_NAMES and "nameDest" not in FEATURE_NAMES
