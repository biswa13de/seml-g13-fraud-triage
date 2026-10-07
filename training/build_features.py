"""Offline feature pipeline (GR4ML Data Preparation View, implemented).

raw PaySim CSV
  → assign txn_id
  → windowed aggregation: per-receiver 24 h velocity (point-in-time, vectorised)
  → filter type IN (TRANSFER, CASH_OUT)
  → derive features via common.features.compute_features (shared with serving)
  → time-based split on `step`
  → undersample genuine payments in TRAIN only (valid/test keep the true base rate)
  → data/features_{train,valid,test}.parquet

Usage: python -m training.build_features
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.features import FEATURE_NAMES, SCORED_TYPES, VELOCITY_WINDOW_STEPS, compute_features

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "paysim.csv"
SPLITS = {"train": (1, 500), "valid": (501, 600), "test": (601, 743)}
TRAIN_GENUINE_FRACTION = 0.25
SEED = 13
LABEL = "isFraud"
KEEP = ["txn_id", "step", "type", "nameOrig", "nameDest", LABEL]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not df["step"].is_monotonic_increasing:
        raise ValueError("PaySim rows must be ordered by step for point-in-time features")
    df.insert(0, "txn_id", "T" + df.index.astype(str))
    return df


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver count and amount over the previous 24 steps, counting only
    payments that came BEFORE the current row (same semantics as the online
    InMemory/Redis feature stores)."""
    n = len(df)
    dest = pd.factorize(df["nameDest"])[0].astype(np.int64)
    order = np.lexsort((np.arange(n), dest))  # group by receiver, keep arrival order
    d, s = dest[order], df["step"].to_numpy()[order]
    a = df["amount"].to_numpy()[order]

    key = d * 1000 + s  # steps <= 743 < 1000, so keys never cross receiver groups
    first_in_window = np.searchsorted(key, d * 1000 + (s - VELOCITY_WINDOW_STEPS + 1), side="left")
    pos = np.arange(n)
    csum = np.concatenate([[0.0], np.cumsum(a)])

    count, total = np.empty(n), np.empty(n)
    count[order] = pos - first_in_window
    total[order] = csum[pos] - csum[first_in_window]
    return df.assign(dest_txn_count_24h=count, dest_amount_sum_24h=total)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = add_velocity(df)
    df = df[df["type"].isin(SCORED_TYPES)].reset_index(drop=True)
    feats = pd.DataFrame(compute_features(df, df[["dest_txn_count_24h", "dest_amount_sum_24h"]]))
    return pd.concat([df[KEEP], feats[FEATURE_NAMES]], axis=1)


def split(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts = {name: table[table["step"].between(lo, hi)] for name, (lo, hi) in SPLITS.items()}
    train = parts["train"]
    genuine = train[train[LABEL] == 0].sample(frac=TRAIN_GENUINE_FRACTION, random_state=SEED)
    parts["train"] = pd.concat([train[train[LABEL] == 1], genuine]).sort_values("step")
    return parts


def main() -> None:
    table = build(load_raw())
    summary = {}
    for name, part in split(table).items():
        part.to_parquet(DATA_DIR / f"features_{name}.parquet", index=False)
        summary[name] = {
            "rows": len(part),
            "frauds": int(part[LABEL].sum()),
            "fraud_rate": round(float(part[LABEL].mean()), 5),
            "steps": SPLITS[name],
        }
    (DATA_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
