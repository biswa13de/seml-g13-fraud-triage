"""Data-quality gates on the offline feature tables: schema, no duplicate
transaction ids, fraud rate in a sane range, no future leakage."""
import pandas as pd
import pytest

SPLITS = ["train", "valid", "test"]


@pytest.fixture(params=SPLITS)
def split_df(request):
    path = f"data/features_{request.param}.parquet"
    try:
        return pd.read_parquet(path), request.param
    except FileNotFoundError:
        pytest.skip(f"{path} not built; run `python -m training.build_features` first")


def test_no_duplicate_transactions(split_df):
    df, _ = split_df
    assert df["txn_id"].is_unique


def test_fraud_rate_in_sane_range(split_df):
    df, name = split_df
    rate = df["isFraud"].mean()
    assert 0.0005 <= rate <= 0.10, f"{name} fraud rate {rate} looks wrong for PaySim TRANSFER/CASH_OUT"


def test_no_missing_values_in_features(split_df):
    from common.features import FEATURE_NAMES
    df, _ = split_df
    assert not df[FEATURE_NAMES].isna().any().any()


def test_velocity_features_non_negative(split_df):
    df, _ = split_df
    assert (df["dest_txn_count_24h"] >= 0).all()
    assert (df["dest_amount_sum_24h"] >= 0).all()


def test_splits_are_time_ordered():
    """Train must end before valid starts, which must end before test starts —
    guards against random-split leakage across time."""
    from training.build_features import SPLITS as STEP_RANGES
    assert STEP_RANGES["train"][1] < STEP_RANGES["valid"][0]
    assert STEP_RANGES["valid"][1] < STEP_RANGES["test"][0]
