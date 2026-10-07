"""Train and compare three algorithms for the fraud triage model, logging
every run to MLflow (GR4ML Analytics Design View, implemented).

Algorithms (mirrors the Analytics Design View's softgoal trade-off):
  - Logistic Regression  : interpretable baseline, fast, linear
  - Random Forest        : non-linear, robust, but slower at inference
  - LightGBM             : non-linear, handles missing values natively,
                            fast at inference -> the champion candidate

Usage: python -m training.train
"""
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common.config import settings
from common.features import FEATURE_NAMES

DATA_DIR = Path("data")
LABEL = "isFraud"
EXPERIMENT = "fraud-triage"
TARGET_FPR = 0.01


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA_DIR / f"features_{name}.parquet")
    return df[FEATURE_NAMES], df[LABEL]


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    """Recall achievable while keeping the false-positive rate at/below target_fpr.
    Used instead of plain accuracy because fraud is a 1% minority class."""
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    neg_total = (y_true == 0).sum()
    pos_total = (y_true == 1).sum()
    fp = np.cumsum(y_sorted == 0)
    tp = np.cumsum(y_sorted == 1)
    fpr = fp / max(neg_total, 1)
    ok = fpr <= target_fpr
    return float(tp[ok].max() / pos_total) if ok.any() else 0.0


def pr_curve_artifact(y_true, scores, path: Path, label: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {label}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def eval_and_log(name, model, X_test, y_test, predict_fn, extra_params=None):
    y_test_arr = y_test.to_numpy()

    t0 = time.perf_counter()
    scores = predict_fn(X_test)
    latency_ms_per_row = (time.perf_counter() - t0) * 1000 / len(X_test)

    pr_auc = average_precision_score(y_test_arr, scores)
    roc_auc = roc_auc_score(y_test_arr, scores)
    recall_1fpr = recall_at_fpr(y_test_arr, scores)

    mlflow.log_params({"algorithm": name, **(extra_params or {})})
    mlflow.log_metrics(
        {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            f"recall_at_{int(TARGET_FPR * 100)}pct_fpr": recall_1fpr,
            "inference_latency_ms_per_row": latency_ms_per_row,
        }
    )
    curve_path = Path(f"/tmp/pr_curve_{name}.png")
    pr_curve_artifact(y_test_arr, scores, curve_path, name)
    mlflow.log_artifact(str(curve_path))

    print(f"{name:16s} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
          f"Recall@{TARGET_FPR:.0%}FPR={recall_1fpr:.4f}  latency={latency_ms_per_row:.3f}ms/row")
    return {"algorithm": name, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "recall_at_1pct_fpr": recall_1fpr, "latency_ms_per_row": latency_ms_per_row}


def train_logistic_regression(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="logistic_regression"):
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=13)
        model.fit(scaler.transform(X_train), y_train)
        result = eval_and_log(
            "logistic_regression", model, X_test, y_test,
            predict_fn=lambda X: model.predict_proba(scaler.transform(X))[:, 1],
            extra_params={"class_weight": "balanced", "max_iter": 1000},
        )
        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(3), serialization_format="pickle")
        return result


def train_random_forest(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="random_forest"):
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced_subsample",
            n_jobs=-1, random_state=13,
        )
        model.fit(X_train, y_train)
        result = eval_and_log(
            "random_forest", model, X_test, y_test,
            predict_fn=lambda X: model.predict_proba(X)[:, 1],
            extra_params={"n_estimators": 200, "max_depth": 10},
        )
        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(3), serialization_format="pickle")
        return result


def train_lightgbm(X_train, y_train, X_valid, y_valid, X_test, y_test):
    with mlflow.start_run(run_name="lightgbm") as run:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=13, verbosity=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        result = eval_and_log(
            "lightgbm", model, X_test, y_test,
            predict_fn=lambda X: model.predict_proba(X)[:, 1],
            extra_params={"n_estimators": model.best_iteration_, "max_depth": 6,
                          "learning_rate": 0.05, "scale_pos_weight": round(scale_pos_weight, 2)},
        )
        mlflow.lightgbm.log_model(model, "model", input_example=X_train.head(3))
        result["run_id"] = run.info.run_id
        return result


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train = load_split("train")
    X_valid, y_valid = load_split("valid")
    X_test, y_test = load_split("test")
    print(f"train={len(X_train)} valid={len(X_valid)} test={len(X_test)}  "
          f"test fraud rate={y_test.mean():.4f}")

    results = [
        train_logistic_regression(X_train, y_train, X_test, y_test),
        train_random_forest(X_train, y_train, X_test, y_test),
        train_lightgbm(X_train, y_train, X_valid, y_valid, X_test, y_test),
    ]

    comparison = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n=== Model comparison (sorted by PR-AUC) ===")
    print(comparison.to_string(index=False))
    Path("data/model_comparison.json").write_text(comparison.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
