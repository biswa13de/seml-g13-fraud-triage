"""QA1 (PLAN.md Sec 4): Performance. Inference p95 within budget on a
realistic batch, measured against the in-process pipeline (not over HTTP,
so this test needs no running services — see scripts/load_test.py for the
end-to-end HTTP load test)."""
import time

import pandas as pd
import pytest

from common.features import FEATURE_NAMES

MODEL_INFERENCE_P95_BUDGET_MS = 50.0


@pytest.fixture(scope="module")
def sample_rows():
    try:
        df = pd.read_parquet("data/features_test.parquet")
    except FileNotFoundError:
        pytest.skip("features_test.parquet not built")
    return df[FEATURE_NAMES].sample(n=min(500, len(df)), random_state=13)


def test_single_row_inference_p95(champion_model, sample_rows):
    latencies = []
    for i in range(len(sample_rows)):
        row = sample_rows.iloc[[i]]
        t0 = time.perf_counter()
        champion_model.predict_proba(row)
        latencies.append((time.perf_counter() - t0) * 1000)
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    assert p95 <= MODEL_INFERENCE_P95_BUDGET_MS, f"model inference p95={p95:.2f}ms"
