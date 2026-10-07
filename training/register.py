"""Quality gate + Model Registry pattern: only promote a model that clears
the bar in PLAN.md Sec 4 (QA2 — Model accuracy). Sets the registry alias
"champion", which scoring-service resolves at startup (models:/<name>@champion).

Usage: python -m training.register [--run-id RUN_ID]
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from common.config import settings
from training.select_thresholds import latest_run_id

MODEL_NAME = "fraud-triage-model"
ALIAS = "champion"
LABEL = "isFraud"

# Quality gate (must match PLAN.md QA2)
MIN_PR_AUC = 0.80
MIN_RECALL_AT_1PCT_FPR = 0.85


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    run_id = args.run_id or latest_run_id("lightgbm")
    run = client.get_run(run_id)
    metrics = run.data.metrics

    print(f"Evaluating run {run_id}: PR-AUC={metrics['pr_auc']:.4f}, "
          f"Recall@1%FPR={metrics['recall_at_1pct_fpr']:.4f}")

    if metrics["pr_auc"] < MIN_PR_AUC or metrics["recall_at_1pct_fpr"] < MIN_RECALL_AT_1PCT_FPR:
        print(f"QUALITY GATE FAILED: needs PR-AUC >= {MIN_PR_AUC} and "
              f"Recall@1%FPR >= {MIN_RECALL_AT_1PCT_FPR}", file=sys.stderr)
        sys.exit(1)

    thresholds_path = Path("data/thresholds.json")
    if not thresholds_path.exists():
        print("Run `python -m training.select_thresholds` first.", file=sys.stderr)
        sys.exit(1)
    thresholds = json.loads(thresholds_path.read_text())

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "t_stepup", str(thresholds["t_stepup"]))
    client.set_model_version_tag(MODEL_NAME, mv.version, "t_block", str(thresholds["t_block"]))
    client.set_model_version_tag(MODEL_NAME, mv.version, "pr_auc", f"{metrics['pr_auc']:.4f}")

    print(f"QUALITY GATE PASSED. Registered {MODEL_NAME} v{mv.version}, "
          f"aliased '{ALIAS}'. Thresholds: {thresholds['t_stepup']:.4f} / {thresholds['t_block']:.4f}")


if __name__ == "__main__":
    main()
