"""Train and compare three algorithms for the fraud triage model, logging
every run to MLflow (GR4ML Analytics Design View, implemented).

Algorithms (mirrors the Analytics Design View's softgoal trade-off):
  - Logistic Regression  : interpretable baseline, fast, linear
  - Random Forest        : non-linear, robust
  - LightGBM             : non-linear, handles missing values natively

How the champion is chosen (select_champion):
  1. Compare on the VALIDATION split only. The test split is never used to
     pick a model; it is kept for the final quality gate in register.py.
  2. Keep every model whose validation PR-AUC is within PR_AUC_TOLERANCE of
     the best one (i.e. accuracy is effectively tied).
  3. Among those, pick the lowest p95 latency to score ONE payment, because
     the service scores one payment per request inside a latency budget.

Usage: python -m training.train
"""
import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
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
PR_AUC_TOLERANCE = 0.001
LATENCY_SAMPLE_ROWS = 300
SHAP_SAMPLE_ROWS = 50
SEED = 13


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
    plt.title(f"Precision-Recall (validation) — {label}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def p95_ms(fn, rows: pd.DataFrame) -> float:
    """p95 wall-clock time of calling fn on one row at a time, as the service does."""
    timings = []
    for i in range(len(rows)):
        row = rows.iloc[[i]]
        t0 = time.perf_counter()
        fn(row)
        timings.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(timings, 95))


def evaluate(name, model, predict_fn, train_seconds, X_valid, y_valid, X_test, y_test,
             tree_model=None, extra_params=None):
    """Log validation metrics (used to choose), test metrics (used only by the
    quality gate), and the serving costs: one-payment scoring and SHAP latency,
    model size and training time."""
    valid_scores = predict_fn(X_valid)
    test_scores = predict_fn(X_test)
    y_valid_arr, y_test_arr = y_valid.to_numpy(), y_test.to_numpy()

    sample = X_valid.sample(LATENCY_SAMPLE_ROWS, random_state=SEED)
    single_row_p95 = p95_ms(predict_fn, sample)
    shap_p95 = float("nan")
    if tree_model is not None:
        explainer = shap.TreeExplainer(tree_model)
        shap_p95 = p95_ms(explainer.shap_values, sample.head(SHAP_SAMPLE_ROWS))

    metrics = {
        "valid_pr_auc": average_precision_score(y_valid_arr, valid_scores),
        "valid_roc_auc": roc_auc_score(y_valid_arr, valid_scores),
        "valid_recall_at_1pct_fpr": recall_at_fpr(y_valid_arr, valid_scores),
        "test_pr_auc": average_precision_score(y_test_arr, test_scores),
        "test_recall_at_1pct_fpr": recall_at_fpr(y_test_arr, test_scores),
        "single_row_p95_ms": single_row_p95,
        "shap_single_row_p95_ms": shap_p95,
        "model_size_mb": len(pickle.dumps(model)) / 1e6,
        "train_seconds": train_seconds,
    }
    mlflow.log_params({"algorithm": name, **(extra_params or {})})
    mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})

    curve_path = Path(f"/tmp/pr_curve_{name}.png")
    pr_curve_artifact(y_valid_arr, valid_scores, curve_path, name)
    mlflow.log_artifact(str(curve_path))

    print(f"{name:20s} valid PR-AUC={metrics['valid_pr_auc']:.5f}  "
          f"Recall@1%FPR={metrics['valid_recall_at_1pct_fpr']:.4f}  "
          f"1-payment p95={single_row_p95:.2f}ms  SHAP p95={shap_p95:.2f}ms  "
          f"size={metrics['model_size_mb']:.1f}MB  train={train_seconds:.1f}s")
    return {"algorithm": name, **metrics, "run_id": mlflow.active_run().info.run_id}


def train_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test):
    with mlflow.start_run(run_name="logistic_regression"):
        t0 = time.perf_counter()
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        model.fit(scaler.transform(X_train), y_train)
        train_seconds = time.perf_counter() - t0
        result = evaluate(
            "logistic_regression", model,
            lambda X: model.predict_proba(scaler.transform(X))[:, 1],
            train_seconds, X_valid, y_valid, X_test, y_test,
            extra_params={"class_weight": "balanced", "max_iter": 1000},
        )
        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(3), serialization_format="pickle")
        return result


def train_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test):
    with mlflow.start_run(run_name="random_forest"):
        t0 = time.perf_counter()
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED,
        )
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0
        # Serve single-threaded: starting a thread pool for one row costs more
        # than it saves, so measure RF the way it would actually be served.
        model.set_params(n_jobs=1)
        result = evaluate(
            "random_forest", model, lambda X: model.predict_proba(X)[:, 1],
            train_seconds, X_valid, y_valid, X_test, y_test, tree_model=model,
            extra_params={"n_estimators": 200, "max_depth": 10},
        )
        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(3), serialization_format="pickle")
        return result


def train_lightgbm(X_train, y_train, X_valid, y_valid, X_test, y_test):
    with mlflow.start_run(run_name="lightgbm"):
        t0 = time.perf_counter()
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=SEED, verbosity=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        train_seconds = time.perf_counter() - t0
        result = evaluate(
            "lightgbm", model, lambda X: model.predict_proba(X)[:, 1],
            train_seconds, X_valid, y_valid, X_test, y_test, tree_model=model,
            extra_params={"n_estimators": model.best_iteration_, "max_depth": 6,
                          "learning_rate": 0.05, "scale_pos_weight": round(scale_pos_weight, 2)},
        )
        mlflow.lightgbm.log_model(model, "model", input_example=X_train.head(3))
        return result


def select_champion(results: list[dict]) -> dict:
    """Validation PR-AUC within PR_AUC_TOLERANCE of the best counts as a tie;
    among the tied models, the fastest to score one payment wins."""
    best_pr_auc = max(r["valid_pr_auc"] for r in results)
    tied = [r for r in results if r["valid_pr_auc"] >= best_pr_auc - PR_AUC_TOLERANCE]
    champion = min(tied, key=lambda r: r["single_row_p95_ms"])
    return {
        "algorithm": champion["algorithm"],
        "run_id": champion["run_id"],
        "rule": (f"validation PR-AUC within {PR_AUC_TOLERANCE} of the best "
                 f"({best_pr_auc:.5f}), then lowest one-payment p95 latency"),
        "tied_on_accuracy": [r["algorithm"] for r in tied],
    }


def train_all(X_train, y_train, X_valid, y_valid, X_test, y_test) -> list[dict]:
    args = (X_train, y_train, X_valid, y_valid, X_test, y_test)
    return [train_logistic_regression(*args), train_random_forest(*args), train_lightgbm(*args)]


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train = load_split("train")
    X_valid, y_valid = load_split("valid")
    X_test, y_test = load_split("test")
    print(f"train={len(X_train)} valid={len(X_valid)} test={len(X_test)}")

    results = train_all(X_train, y_train, X_valid, y_valid, X_test, y_test)
    champion = select_champion(results)

    comparison = pd.DataFrame(results).sort_values("valid_pr_auc", ascending=False)
    print("\n=== Model comparison (validation split) ===")
    print(comparison.drop(columns=["run_id"]).to_string(index=False))
    print(f"\nChampion: {champion['algorithm']} ({champion['rule']})")

    (DATA_DIR / "model_comparison.json").write_text(comparison.to_json(orient="records", indent=2))
    (DATA_DIR / "champion.json").write_text(json.dumps(champion, indent=2) + "\n")


if __name__ == "__main__":
    # Force a headless backend only for a standalone CLI run (no display to draw
    # to); importing this module into an already-running kernel (e.g. a notebook)
    # must never override that kernel's own matplotlib backend as a side effect.
    import matplotlib

    matplotlib.use("Agg")
    main()
