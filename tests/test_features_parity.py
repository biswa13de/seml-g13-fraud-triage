"""Proves there is no training-serving skew: replaying the raw PaySim rows
through the online InMemoryVelocityStore reproduces the offline parquet
features exactly, since both paths share common.features.compute_features."""
import numpy as np
import pandas as pd
import pytest

from common.feature_store import InMemoryVelocityStore
from common.features import FEATURE_NAMES, compute_features
from training.build_features import KEEP, SCORED_TYPES, load_raw

N_ROWS = 20_000  # enough rows to exercise repeat receivers within the window


@pytest.fixture(scope="module")
def raw_sample():
    return load_raw().head(N_ROWS)


def test_online_replay_matches_offline_pipeline(raw_sample):
    from training.build_features import build

    offline = build(raw_sample.copy())

    store = InMemoryVelocityStore()
    online_rows = []
    for row in raw_sample.itertuples(index=False):
        row = row._asdict() if hasattr(row, "_asdict") else dict(zip(raw_sample.columns, row))
        if row["type"] not in SCORED_TYPES:
            store.record(row["txn_id"], row["nameDest"], row["step"], row["amount"])
            continue
        velocity = store.get(row["nameDest"], row["step"])  # read BEFORE recording this txn
        feats = compute_features(row, velocity)
        online_rows.append({**{k: row[k] for k in KEEP}, **feats})
        store.record(row["txn_id"], row["nameDest"], row["step"], row["amount"])

    online = pd.DataFrame(online_rows)[offline.columns]
    pd.testing.assert_frame_equal(
        offline.reset_index(drop=True), online.reset_index(drop=True), check_like=False
    )


def test_feature_names_are_finite_and_ordered(raw_sample):
    from training.build_features import build

    offline = build(raw_sample.copy())
    assert list(FEATURE_NAMES) == [c for c in offline.columns if c in FEATURE_NAMES]
    assert np.isfinite(offline[FEATURE_NAMES].to_numpy()).all()
