"""Population Stability Index (PSI) for score drift monitoring (FR8).

PSI compares the distribution of risk scores served in production against
the distribution observed on the validation split at training time. A PSI
over ~0.2 conventionally signals the population has shifted enough to
warrant investigation or retraining (Session 4: Model Drift Monitoring)."""
import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 2:
        return 0.0
    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
