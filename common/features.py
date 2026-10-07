"""Single source of truth for model features.

`compute_features` is used by BOTH the offline training pipeline (called with
pandas Series) and the online scoring service (called with scalars), so the
two can never drift apart (no training-serving skew).

Only information available at authorisation time is used: post-transaction
balances (newbalanceOrig / newbalanceDest) are deliberately excluded because
they leak the outcome of the payment.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

VELOCITY_WINDOW_STEPS = 24  # PaySim step = 1 hour, so this is a 24 h window
NIGHT_HOURS = (0, 1, 2, 3, 4, 5)
SCORED_TYPES = ("TRANSFER", "CASH_OUT")  # the only types where fraud occurs

FEATURE_NAMES = [
    "amount",
    "log_amount",
    "is_transfer",
    "oldbalance_orig",
    "amount_to_balance_ratio",
    "drains_account",
    "orig_zero_balance",
    "oldbalance_dest",
    "dest_zero_balance",
    "hour_of_day",
    "is_night",
    "dest_txn_count_24h",
    "dest_amount_sum_24h",
]


@dataclass(frozen=True)
class Velocity:
    """Behavioural aggregates for the receiving account over the last 24 h,
    counting only payments that happened BEFORE the current one."""

    dest_txn_count_24h: float = 0.0
    dest_amount_sum_24h: float = 0.0


def compute_features(txn: Mapping, velocity: Velocity | Mapping) -> dict:
    """Map one payment (or a column-wise batch of payments) to model features.

    Works element-wise for scalars and pandas Series alike.
    """
    if isinstance(velocity, Velocity):
        velocity = velocity.__dict__

    amount = txn["amount"]
    bal_orig = txn["oldbalanceOrg"]
    bal_dest = txn["oldbalanceDest"]
    hour = txn["step"] % 24

    return {
        "amount": amount * 1.0,
        "log_amount": np.log1p(amount),
        "is_transfer": (txn["type"] == "TRANSFER") * 1.0,
        "oldbalance_orig": bal_orig * 1.0,
        "amount_to_balance_ratio": amount / (bal_orig + 1.0),
        "drains_account": ((bal_orig > 0) & (amount >= 0.99 * bal_orig)) * 1.0,
        "orig_zero_balance": (bal_orig == 0) * 1.0,
        "oldbalance_dest": bal_dest * 1.0,
        "dest_zero_balance": (bal_dest == 0) * 1.0,
        "hour_of_day": hour * 1.0,
        "is_night": np.isin(hour, NIGHT_HOURS) * 1.0,
        "dest_txn_count_24h": velocity["dest_txn_count_24h"] * 1.0,
        "dest_amount_sum_24h": velocity["dest_amount_sum_24h"] * 1.0,
    }


def to_vector(features: Mapping) -> np.ndarray:
    """Order a single payment's features exactly as the model was trained."""
    return np.array([[float(features[name]) for name in FEATURE_NAMES]])
