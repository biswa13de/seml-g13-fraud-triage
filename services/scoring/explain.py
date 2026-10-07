"""Reason-code generation (QA3: Explainability, PLAN.md Sec 4).

Uses a SHAP TreeExplainer on the champion LightGBM model to find each
transaction's top-3 contributing features, then maps them to a short,
analyst- and customer-readable sentence. Explanation runs synchronously in
the scoring pipeline but is a single cheap TreeExplainer call (no sampling),
so it stays within the QA3 latency budget (<=10ms p95 overhead).
"""
import numpy as np
import shap

from common.features import FEATURE_NAMES

# Human-readable template per feature. {value} is filled from the raw row.
_TEMPLATES = {
    "amount": "Payment amount is unusually large (₹{value:,.0f})",
    "log_amount": "Payment amount is unusually large (₹{value:,.0f})",
    "is_transfer": "Payment is a direct transfer rather than a cash-out",
    "oldbalance_orig": "Sender's balance before payment was ₹{value:,.0f}",
    "amount_to_balance_ratio": "Payment is {value:.0%} of the sender's balance",
    "drains_account": "Transfer empties the sender's account balance",
    "orig_zero_balance": "Sender's account balance was zero before paying",
    "oldbalance_dest": "Receiver's balance before payment was ₹{value:,.0f}",
    "dest_zero_balance": "Receiver's account balance was zero before payment",
    "hour_of_day": "Payment made at hour {value:.0f} of the day",
    "is_night": "Payment made during night hours (12am-6am)",
    "dest_txn_count_24h": "Receiver got {value:.0f} payments in the last 24h",
    "dest_amount_sum_24h": "Receiver collected ₹{value:,.0f} in the last 24h",
}

TOP_K = 3


class ReasonCodeExplainer:
    def __init__(self, model) -> None:
        # LightGBM's native booster; shap.TreeExplainer is O(depth) per row,
        # independent of training-set size.
        self._explainer = shap.TreeExplainer(model)

    def explain(self, feature_row: dict) -> list[str]:
        x = np.array([[feature_row[name] for name in FEATURE_NAMES]])
        shap_values = self._explainer.shap_values(x)
        # LightGBM binary classifier: shap_values is (n_rows, n_features) and
        # positive values push the prediction toward the fraud class.
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        top_idx = np.argsort(-np.abs(values))[:TOP_K]
        reasons = []
        for i in top_idx:
            if values[i] <= 0:
                continue  # only report features that pushed risk UP
            name = FEATURE_NAMES[i]
            template = _TEMPLATES.get(name, name)
            reasons.append(template.format(value=feature_row[name]))
        if not reasons:
            reasons = ["No single factor dominates; risk is driven by a combination of signals"]
        return reasons
