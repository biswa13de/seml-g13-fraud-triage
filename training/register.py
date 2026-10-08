"""Quality gate + Model Registry pattern: only promote a model that clears
the bar in PLAN.md Sec 4 (QA2 — Model accuracy). Sets the registry alias
"champion", which scoring-service resolves at startup (models:/<name>@champion).

The candidate is whichever model training.train selected (data/champion.json),
chosen on the validation split. The gate itself is checked on the held-out
TEST split, which played no part in that choice.

Usage: python -m training.register [--run-id RUN_ID]
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from common.config import settings
from training.select_thresholds import load_champion

# scoring-service loads the registered model with mlflow.lightgbm and explains it
# with SHAP's TreeExplainer, so it can only serve this model type.
SERVABLE_ALGORITHMS = {"lightgbm"}

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
    champion = load_champion()
    run_id = args.run_id or champion["run_id"]
    algorithm = client.get_run(run_id).data.params.get("algorithm", champion["algorithm"])
    if algorithm not in SERVABLE_ALGORITHMS:
        print(f"Selected model '{algorithm}' can't be served: scoring-service only loads "
              f"{sorted(SERVABLE_ALGORITHMS)}. Extend services/scoring before registering it.",
              file=sys.stderr)
        sys.exit(1)

    metrics = client.get_run(run_id).data.metrics
    pr_auc, recall = metrics["test_pr_auc"], metrics["test_recall_at_1pct_fpr"]
    print(f"Evaluating {algorithm} run {run_id} on the held-out test split: "
          f"PR-AUC={pr_auc:.4f}, Recall@1%FPR={recall:.4f}")

    if pr_auc < MIN_PR_AUC or recall < MIN_RECALL_AT_1PCT_FPR:
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
    client.set_model_version_tag(MODEL_NAME, mv.version, "test_pr_auc", f"{pr_auc:.4f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "selection_rule", champion["rule"])

    print(f"QUALITY GATE PASSED. Registered {MODEL_NAME} v{mv.version}, "
          f"aliased '{ALIAS}'. Thresholds: {thresholds['t_stepup']:.4f} / {thresholds['t_block']:.4f}")


if __name__ == "__main__":
    main()
