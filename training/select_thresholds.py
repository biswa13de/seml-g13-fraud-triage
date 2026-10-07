"""Cost-based threshold search for the champion model (GR4ML PrescriptionGoal,
implemented): turn a fraud probability into one of ALLOW / STEP_UP / BLOCK.

Expected cost per payment (grid-searched on the VALIDATION split):
  ALLOW a fraud          -> lose the full amount
  STEP_UP a genuine txn  -> COST_FRICTION   (customer friction)
  BLOCK a genuine txn    -> COST_FALSE_BLOCK (customer churn, worse than friction)
  STEP_UP a fraud        -> COST_STEPUP_CATCH (assume OTP stops most attempts,
                             but not all, so this still carries residual risk)

Search t_stepup < t_block over a grid, subject to STEP_UP rate <= MAX_STEPUP_RATE
(business constraint: QA "little friction", see PLAN.md Sec 2.1).
Writes data/thresholds.json, which is registered as a versioned artifact
alongside the model (training/register.py) so serving never hardcodes cutoffs.

Usage: python -m training.select_thresholds [--run-id RUN_ID]
"""
import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from common.features import FEATURE_NAMES

DATA_DIR = Path("data")
LABEL = "isFraud"
MAX_STEPUP_RATE = 0.03
COST_FRICTION = 50.0
COST_FALSE_BLOCK = 500.0
COST_STEPUP_RESIDUAL_FRACTION = 0.20  # fraction of a STEP_UPed fraud still assumed lost


def latest_run_id(run_name: str) -> str:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(
        experiment_names=["fraud-triage"],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise SystemExit(f"No MLflow run named '{run_name}'. Run `python -m training.train` first.")
    return runs.iloc[0]["run_id"]


def expected_cost(y_true: np.ndarray, amount: np.ndarray, scores: np.ndarray,
                   t_stepup: float, t_block: float) -> float:
    decision = np.where(scores >= t_block, "BLOCK", np.where(scores >= t_stepup, "STEP_UP", "ALLOW"))
    fraud, genuine = y_true == 1, y_true == 0

    cost = 0.0
    cost += amount[fraud & (decision == "ALLOW")].sum()  # fraud gets through fully
    cost += COST_STEPUP_RESIDUAL_FRACTION * amount[fraud & (decision == "STEP_UP")].sum()
    cost += COST_FRICTION * (genuine & (decision == "STEP_UP")).sum()
    cost += COST_FALSE_BLOCK * (genuine & (decision == "BLOCK")).sum()
    return cost


def search(y_true: np.ndarray, amount: np.ndarray, scores: np.ndarray) -> dict:
    grid = np.unique(np.quantile(scores, np.linspace(0.80, 0.999, 60)))
    best = None
    for t_block in grid:
        for t_stepup in grid[grid < t_block]:
            stepup_rate = ((scores >= t_stepup) & (scores < t_block)).mean()
            if stepup_rate > MAX_STEPUP_RATE:
                continue
            cost = expected_cost(y_true, amount, scores, t_stepup, t_block)
            if best is None or cost < best["expected_cost"]:
                best = {
                    "t_stepup": float(t_stepup), "t_block": float(t_block),
                    "expected_cost": float(cost), "stepup_rate": float(stepup_rate),
                }
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="MLflow run id; defaults to latest lightgbm run")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run_id = args.run_id or latest_run_id("lightgbm")
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    valid = pd.read_parquet(DATA_DIR / "features_valid.parquet")
    scores = model.predict_proba(valid[FEATURE_NAMES])[:, 1]
    best = search(valid[LABEL].to_numpy(), valid["amount"].to_numpy(), scores)
    best.update({"run_id": run_id, "max_stepup_rate_constraint": MAX_STEPUP_RATE})

    out = DATA_DIR / "thresholds.json"
    out.write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
