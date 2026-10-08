"""Builds 13.ipynb cell-by-cell with nbformat. Colab-executable: clones the repo,
installs deps, downloads the dataset, trains inline. Run this script from the repo root."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

REPO_URL = "https://github.com/biswa13de/seml-g13-fraud-triage.git"
REPO_DIR = "seml-g13-fraud-triage"

# ---------------------------------------------------------------- Header
md(f"""\
# Real-Time Digital Payment Fraud Risk Triage System

**AIMLZG546 — Software Engineering for Machine Learning — Assignment I**

| | |
|---|---|
| **Group No.** | 13 |
| **Domain** | Financial Technology (FinTech) |
| **Problem** | Real-Time Digital Payment Fraud Risk Triage |
| **Repository** | {REPO_URL.removesuffix('.git')} |

### Group members

| Sl. No | BITS ID | Name | Contribution |
|---|---|---|---|
| 1 | 2025ae05576 | Anirudh Anand | Report lead. Domain/problem statement, requirements and measurable goals, Business View, final report. (25%) |
| 2 | 2025ae05178 | Aniketh Paul | GR4ML Analytics Design and Data Preparation views, quality requirements. Data pipeline: EDA, ingest, clean, features, split. (25%) |
| 3 | 2025ae05898 | Pushadapu Sanjay Kumar | Architecture diagram, the two patterns. FastAPI service, input validation, prediction logging, automated tests. (25%) |
| 4 | 2025af05111 | Biswajeet Mahato (Group Lead) | Model pipeline: train/evaluate, thresholds, MLflow tracking and registry. End-to-end run, screenshots. (25%) |

---

## How to run this notebook

**On Google Colab:** run every cell top to bottom, with no setup and no Kaggle account or API
token needed. Section 0 clones the repository and installs dependencies; Section 2 downloads the
dataset via `kagglehub`, which fetches this public dataset anonymously; Section 5 trains the three
models **inline** (~1–2 minutes on Colab's CPU runtime) against a local, file-based MLflow store,
since Colab cannot reach our local Docker services. Section 9 (calling the live running system)
only works on the machine that has `docker compose up -d` running — on Colab it is skipped
automatically.

**Locally, with our Docker stack running:** the notebook still works end to end; Section 0's clone
step is skipped if the repository is already present, Section 2 skips the download if the CSV is
already there, and Section 10 will successfully reach `http://localhost:8000`.

## What this notebook does

This notebook walks through the full pipeline we built: downloading the dataset, exploring it,
building point-in-time-correct features, training and comparing three algorithms in MLflow,
choosing cost-based triage thresholds, explaining individual predictions with SHAP, and finally
(when available) calling our live, running services to show the end-to-end system working.

The actual production code lives in `training/`, `common/` and `services/` as proper Python
modules with tests (see `tests/`, 33 passing) — this notebook imports and calls that code rather
than duplicating it.
""")

# ---------------------------------------------------------------- Problem statement
md("""\
## 1. Problem Statement

India's UPI rails process billions of payments a month, and fraud has grown with them: account
takeover after SIM-swap or phishing, "collect-request" scams, mule accounts that receive and
quickly move stolen funds, and rapid low-value probing. A typical PSP (Payment Service Provider)
today relies on static rules (amount limits, blocklists), which **miss new fraud patterns** and
**decline too many genuine payments**.

We built a **Real-Time Fraud Risk Triage System** that, for every payment, before authorisation:

1. Estimates fraud probability with an ML model using transaction and behavioural features
2. Maps that risk to a cost-optimal action — **ALLOW**, **STEP_UP** (extra authentication), or
   **BLOCK** — inside a strict latency budget
3. Returns human-readable reason codes for every decision
4. Opens prioritised cases for fraud analysts, whose verdicts flow back as labels for retraining

**Why ML instead of rules:** fraud patterns change adversarially, the signal comes from many weak
interacting features, labelled history exists, and an occasional wrong answer is tolerable because
STEP_UP and human review act as safety nets.
""")

# ---------------------------------------------------------------- Section 0: clone + install
md("""\
## 0. Environment Setup (Colab-safe)

This cell detects whether it is running on Google Colab. If so, it clones the repository (skipping
the clone if it is already present, e.g. on a re-run) and `cd`s into it so every relative path
(`data/...`, `common/...`) used below resolves correctly, then installs the pinned dependencies.
Locally, it just confirms you are inside the repository.
""")
code(f"""\
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "{REPO_URL}"
REPO_DIR = "{REPO_DIR}"

if IN_COLAB:
    if not os.path.isdir(REPO_DIR):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    os.chdir(REPO_DIR)
    print(f"Colab detected. Working directory: {{os.getcwd()}}")
else:
    # Local run: walk up from the current directory until we find the repo root
    # (identified by the presence of the `common/` package), in case the notebook
    # was launched from a subdirectory.
    cwd = os.getcwd()
    while not os.path.isdir(os.path.join(cwd, "common")) and os.path.dirname(cwd) != cwd:
        cwd = os.path.dirname(cwd)
    assert os.path.isdir(os.path.join(cwd, "common")), (
        "Could not locate the repository root (a `common/` folder). "
        "Run this notebook from inside seml-g13-fraud-triage, or on Colab."
    )
    os.chdir(cwd)
    print(f"Local run detected. Working directory: {{os.getcwd()}}")
""")
code("""\
# Install dependencies. On Colab this takes ~1-2 minutes the first time; locally, if you already
# have the project's virtualenv active, this is a fast no-op confirmation pass.
%pip install -q -r requirements.txt
print("Dependencies installed.")
""")

# ---------------------------------------------------------------- Section: all imports up front
md("""\
## Imports

Every library this notebook uses, imported once, here. Later sections import only project modules
(`common.*`, `training.*`, `services.*`) at the point they are first used, since those modules are
only importable once Section 0 has `cd`'d into the repository and installed dependencies.
""")
code("""\
import logging
import os
import warnings

# These must be set BEFORE `import mlflow`, since mlflow reads them at import time.
warnings.filterwarnings("ignore")
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"  # tqdm's \\r writes can corrupt inline display
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import json
import sys
from pathlib import Path

import httpx
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

matplotlib.use("module://matplotlib_inline.backend_inline")
%matplotlib inline
matplotlib.rcParams["figure.dpi"] = 100
pd.set_option("display.max_columns", 20)
logging.getLogger("mlflow").setLevel(logging.ERROR)  # after import: mlflow sets its own level on import

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

print("Imports OK. Repo root:", REPO_ROOT)
""")

# ---------------------------------------------------------------- Dataset
md("""\
## 2. Dataset

We use the public **PaySim** dataset from Kaggle (`ealaxi/paysim1`): 6.36M simulated mobile-money
transactions, the closest public match to UPI peer-to-peer payments, since it has sender and
receiver account IDs (needed for velocity and mule features).

**Mapping to UPI:** `nameOrig` → payer VPA · `nameDest` → payee VPA (`M…` = merchant) ·
`TRANSFER` → P2P transfer · `CASH_OUT` → cash-out via agent · `step` → hour of the month.

The cell below downloads the CSV if it is not already present (skipped on a re-run, and skipped
locally if you already ran `python data/download_paysim.py`). It uses `kagglehub`
(`data/download_paysim.py`'s preferred path), which downloads this public dataset **with no
Kaggle account, login or API token** — nothing to upload, nothing to configure, works the same on
Colab and locally. (A Kaggle API token is only needed for the `kaggle` CLI fallback, which this
script only falls back to if `kagglehub` is somehow unavailable.)
""")
code("""\
DATA_CSV = REPO_ROOT / "data" / "paysim.csv"

if not DATA_CSV.exists():
    subprocess.run([sys.executable, "data/download_paysim.py"], check=True)
else:
    print(f"Found existing {DATA_CSV}, skipping download.")

provenance = json.loads((REPO_ROOT / "data" / "provenance.json").read_text())
print(json.dumps(provenance, indent=2))
""")
code("""\
raw = pd.read_csv(DATA_CSV)
print(f"{len(raw):,} rows, {raw['isFraud'].sum():,} frauds ({raw['isFraud'].mean():.4%})")
raw.head()
""")

# ---------------------------------------------------------------- EDA
md("""\
## 3. Exploratory Data Analysis

Two questions drive the feature design: **which transaction types actually contain fraud**, and
**what separates a fraudulent transaction from a genuine one**.
""")
code("""\
by_type = raw.groupby("type")["isFraud"].agg(["count", "sum", "mean"])
by_type.columns = ["count", "frauds", "fraud_rate"]
by_type
""")
md("""\
Fraud only occurs in `TRANSFER` and `CASH_OUT`. `PAYMENT`, `DEBIT` and `CASH_IN` have **zero**
fraud in this dataset, so we score only `TRANSFER`/`CASH_OUT` and let the gateway's rule engine
allow the rest without calling the model (see `common/features.py:SCORED_TYPES`).
""")
code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

by_type["count"].plot.bar(ax=axes[0], color="#4C72B0")
axes[0].set_title("Transaction count by type")
axes[0].set_ylabel("count")

by_type["fraud_rate"].plot.bar(ax=axes[1], color="#C44E52")
axes[1].set_title("Fraud rate by type")
axes[1].set_ylabel("fraud rate")

plt.tight_layout()
plt.show()
""")
md("""\
A key design decision for the Data Preparation View: fraudulent transactions are far more likely
to **drain the sender's balance completely** and send to a **brand-new, zero-balance** receiver
account. These are legitimate pre-transaction signals (not leakage — `oldbalanceOrg` and
`oldbalanceDest` are known before the payment is authorised), and they turn out to separate the
classes very cleanly.
""")
code("""\
scored = raw[raw["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
scored["drains_account"] = (scored["amount"] >= 0.99 * scored["oldbalanceOrg"]) & (scored["oldbalanceOrg"] > 0)
scored["dest_zero_balance"] = scored["oldbalanceDest"] == 0

compare = scored.groupby("isFraud")[["drains_account", "dest_zero_balance"]].mean()
compare.index = ["Genuine", "Fraud"]
compare
""")
code("""\
fig, ax = plt.subplots(figsize=(6, 4))
compare.plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_ylabel("proportion of transactions")
ax.set_title("Draining the account / sending to a fresh account,\\ngenuine vs fraud")
ax.legend(["Drains sender balance", "Receiver had zero balance"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------- Feature pipeline
md("""\
## 4. Feature Engineering (Data Preparation View)

We use **one shared function**, `common.features.compute_features()`, for both training and
live serving — this is how we guarantee there is no training/serving skew. A test
(`tests/test_features_parity.py`) proves the two paths produce byte-identical output by replaying
raw transactions through the same in-memory feature store used at training time.

Pipeline: filter to TRANSFER/CASH_OUT → drop post-transaction balance columns (they would leak
the outcome) → compute 24-hour receiver velocity (point-in-time correct — only transactions that
happened *before* the current one count) → derive the 13 model features → time-based
train/valid/test split on `step` (not a random split, since fraud patterns change over time).

The cell below builds the feature tables if they are not already present (first run or on Colab);
this takes a few seconds.
""")
code("""\
from common.features import FEATURE_NAMES
from training.build_features import SPLITS as STEP_RANGES
from training.build_features import main as build_features_main

if not (REPO_ROOT / "data" / "features_train.parquet").exists():
    build_features_main()

print("Features used by the model:")
for f in FEATURE_NAMES:
    print(" -", f)
print()
print("Time-based split (by `step`, i.e. simulated hour):", STEP_RANGES)
""")
code("""\
summary = json.loads((REPO_ROOT / "data" / "split_summary.json").read_text())
pd.DataFrame(summary).T
""")
md("""\
Note the fraud rate rises across train → valid → test: this is realistic concept drift, and it's
exactly why we monitor PSI drift in production (`services/monitor`) rather than assuming a static
model stays accurate forever.
""")

# ---------------------------------------------------------------- Training comparison
md("""\
## 5. Model Training & Comparison (Analytics Design View)

We compare three algorithms, mirroring the Analytics Design View's softgoal trade-offs:

| Algorithm | Softgoal it's strong on | Softgoal it's weak on |
|---|---|---|
| Logistic Regression | Interpretability, latency | Accuracy on non-linear patterns |
| Random Forest | Accuracy | Inference latency, model size |
| LightGBM | Accuracy, latency, missing values (native) | Interpretability (fixed with SHAP, below) |

**How the winner is chosen** (`training/train.py:select_champion`), decided before looking at the
results:

1. Compare on the **validation** split only. The test split is never used to pick a model; it is
   kept for the final quality gate in Section 6.
2. Any model whose validation PR-AUC is within 0.001 of the best counts as tied on accuracy.
3. Among the tied models, pick the one that scores **a single payment** fastest (p95), because the
   service scores one payment per request inside a latency budget.

For each model we also measure SHAP explanation time for one payment, model size and training time.
The cell below trains all three with `training/train.py`'s own functions against a local, file-based
MLflow store, so it runs unmodified on Colab. Expect about a minute on a laptop or Colab CPU.
""")
code("""\
mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'notebook_mlflow.db'}")
mlflow.set_experiment("fraud-triage-notebook")

from training.train import load_split, select_champion, train_all

X_train, y_train = load_split("train")
X_valid, y_valid = load_split("valid")
X_test, y_test = load_split("test")
print(f"train={len(X_train)} valid={len(X_valid)} test={len(X_test)}")

results = train_all(X_train, y_train, X_valid, y_valid, X_test, y_test)
selected = select_champion(results)

comparison = pd.DataFrame(results).sort_values("valid_pr_auc", ascending=False).reset_index(drop=True)
comparison[["algorithm", "valid_pr_auc", "valid_recall_at_1pct_fpr", "single_row_p95_ms",
            "shap_single_row_p95_ms", "model_size_mb", "train_seconds"]].round(5)
""")
code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].barh(comparison["algorithm"], comparison["valid_pr_auc"], color="#55A868")
axes[0].set_xlabel("Validation PR-AUC")
axes[0].set_title("Accuracy (higher is better)")
axes[0].set_xlim(0.9, 1.0)

axes[1].barh(comparison["algorithm"], comparison["single_row_p95_ms"], color="#C44E52")
axes[1].set_xlabel("p95 time to score one payment (ms)")
axes[1].set_title("Serving latency (lower is better)")

plt.tight_layout()
plt.show()

print(f"Tied on accuracy: {selected['tied_on_accuracy']}")
print(f"Champion: {selected['algorithm']}  ({selected['rule']})")
""")
md("""\
**Why LightGBM wins.** Random Forest and LightGBM are tied on validation accuracy (both PR-AUC
about 1.000 and recall 1.0 at 1% FPR), so accuracy can't separate them. Logistic Regression is
clearly behind. Between the two tied models, LightGBM scores one payment **6–12x faster** across our
runs (about 0.3 ms vs 1.7–3.5 ms p95, with Random Forest served single-threaded), explains it with SHAP
about **7x faster**, is about **6x smaller** (0.7 MB vs 4.2 MB) and retrains **5–7x faster**. Those
are the costs our quality requirements depend on: the latency budget (QA1), cheap explanations
(QA3) and cheap retraining when the data drifts.

One note for honesty: on the **test** split, Random Forest is marginally ahead (PR-AUC 0.99998 vs
0.99974). We deliberately did not use the test split to choose. Choosing on test data would leak it
into the decision and leave no unbiased final check. The test split is used only for the quality
gate below.

Exact numbers vary slightly between runs and machines (latency especially). Our own run against the
project's MLflow server is recorded in `data/model_comparison.json` and `data/champion.json`.
""")

# ---------------------------------------------------------------- Load champion + evaluate
md("""\
## 6. The Champion Model and the Quality Gate

`training/register.py` only promotes the selected model to the registry's `champion` alias if it
clears a quality gate on the held-out **test** split: PR-AUC &ge; 0.80 and Recall@1%FPR &ge; 0.85
(see also `tests/test_quality_model.py`). We apply the same gate here to the model chosen above.
""")
code("""\
from training.select_thresholds import load_run_model
from training.train import recall_at_fpr

champion = load_run_model(selected["run_id"], selected["algorithm"])

scores = champion.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, scores)
roc_auc = roc_auc_score(y_test, scores)
recall_1pct = recall_at_fpr(y_test.to_numpy(), scores, target_fpr=0.01)

print(f"Champion: {selected['algorithm']}")
print(f"Test PR-AUC:            {pr_auc:.4f}")
print(f"Test ROC-AUC:           {roc_auc:.4f}")
print(f"Test Recall @ 1% FPR:   {recall_1pct:.4f}")
print()
print("Quality gate (PLAN.md Sec 4, QA2): PR-AUC >= 0.80 and Recall@1%FPR >= 0.85")
print("PASSED" if pr_auc >= 0.80 and recall_1pct >= 0.85 else "FAILED")
""")
code("""\
precision, recall, _ = precision_recall_curve(y_test, scores)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(recall, precision, color="#4C72B0")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"{selected['algorithm']} (champion) — test Precision-Recall curve\\nPR-AUC = {pr_auc:.4f}")
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------- Thresholds
md("""\
## 7. Cost-Based Triage Thresholds (GR4ML PrescriptionGoal)

A risk score alone isn't a decision. `training/select_thresholds.py` grid-searches two cutoffs —
`t_stepup` and `t_block` — that minimise an **expected cost** function on the validation set:

- Letting fraud through (ALLOW) costs the full transaction amount
- A false STEP_UP costs customer friction (we use ₹50 as a proxy)
- A false BLOCK costs more — customer churn (₹500 proxy)
- A STEP_UP on real fraud still carries some residual risk (we assume 20% still gets through)

The search is constrained to keep the STEP_UP rate under 3% of all payments, so we don't annoy too
many genuine customers. We re-run the same search function here, against this notebook's own
champion model.
""")
code("""\
from training.select_thresholds import search as search_thresholds

valid_df = pd.read_parquet(REPO_ROOT / "data" / "features_valid.parquet")
valid_scores = champion.predict_proba(valid_df[FEATURE_NAMES])[:, 1]
thresholds = search_thresholds(valid_df["isFraud"].to_numpy(), valid_df["amount"].to_numpy(), valid_scores)
print(json.dumps(thresholds, indent=2))
""")
code("""\
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 1, 60)
ax.hist(scores[y_test == 0], bins=bins, alpha=0.6, label="Genuine", color="#4C72B0", log=True)
ax.hist(scores[y_test == 1], bins=bins, alpha=0.6, label="Fraud", color="#C44E52", log=True)
ax.axvline(thresholds["t_stepup"], color="#DD8452", linestyle="--", label=f"t_stepup={thresholds['t_stepup']:.4f}")
ax.axvline(thresholds["t_block"], color="#55A868", linestyle="--", label=f"t_block={thresholds['t_block']:.4f}")
ax.set_xlabel("Risk score")
ax.set_ylabel("Count (log scale)")
ax.set_title("Score distribution with triage thresholds")
ax.legend()
plt.tight_layout()
plt.show()
""")
md("""\
Because LightGBM separates the classes so cleanly on this dataset, both thresholds sit very close
to zero — only a thin slice of ambiguous scores falls into the STEP_UP band at all. We flag this
honestly: PaySim's fraud signal (draining the account into a fresh receiver) is strong and easy
to learn, so real-world performance on noisier data would likely need a wider STEP_UP band.
""")

# ---------------------------------------------------------------- SHAP explainability
md("""\
## 8. Explainability — SHAP (QA3)

Every STEP_UP or BLOCK decision ships with up to 3 plain-language reason codes, generated by a
SHAP `TreeExplainer` on the champion model (`services/scoring/explain.py`). This satisfies the
"Explainability" quality attribute: analysts and regulators can see *why* a payment was flagged,
not just that it was.
""")
code("""\
explainer = shap.TreeExplainer(champion)

sample = X_test.sample(300, random_state=13)
shap_values = explainer.shap_values(sample)
values = shap_values[1] if isinstance(shap_values, list) else shap_values

shap.summary_plot(values, sample, feature_names=FEATURE_NAMES, show=False)
plt.tight_layout()
plt.show()
""")
code("""\
from services.scoring.explain import ReasonCodeExplainer

reason_explainer = ReasonCodeExplainer(champion)

# One genuine-looking row and one fraud-looking row from the test set, for a side-by-side example
genuine_row = X_test[y_test == 0].iloc[0].to_dict()
fraud_row = X_test[y_test == 1].iloc[0].to_dict()

for label, row in [("Genuine example", genuine_row), ("Fraud example", fraud_row)]:
    score = champion.predict_proba([[row[f] for f in FEATURE_NAMES]])[0][1]
    reasons = reason_explainer.explain(row)
    print(f"{label}  (risk_score={score:.4f})")
    for r in reasons:
        print("  -", r)
    print()
""")

# ---------------------------------------------------------------- Live API calls
md("""\
## 9. Calling the Live, Running System (local only)

Everything above runs anywhere. This section additionally calls our **actually running**
services — `triage-api` on port 8000 — the same containers behind the screenshots in our report
(`docs/screenshots/`). This only works on the machine that has our Docker stack up
(`docker compose up -d`); **on Colab, or if the stack isn't running locally, this section is
skipped automatically** rather than failing the notebook.
""")
code("""\
TRIAGE_URL = "http://localhost:8000/v1/payments/triage"
HEADERS = {"X-API-Key": "demo-key-g13", "Content-Type": "application/json"}

def triage(payment: dict) -> dict:
    resp = httpx.post(TRIAGE_URL, json=payment, headers=HEADERS, timeout=5.0)
    resp.raise_for_status()
    return resp.json()

try:
    health = httpx.get("http://localhost:8000/health", timeout=2.0).json()
    LIVE_STACK_AVAILABLE = True
    print("Gateway health:", health)
except Exception as exc:
    LIVE_STACK_AVAILABLE = False
    print("Live stack not reachable from this environment (expected on Colab).")
    print(f"Reason: {exc}")
    print("Skipping Section 9's live calls. Run `docker compose up -d` locally to try them.")
""")
md("### Scenario 1 — genuine payment")
code("""\
if LIVE_STACK_AVAILABLE:
    genuine_payment = {
        "txn_id": "NB-GENUINE-1", "step": 120, "type": "TRANSFER", "amount": 500.0,
        "nameOrig": "C-NB-1", "oldbalanceOrg": 5000.0,
        "nameDest": "C-NB-2", "oldbalanceDest": 3000.0,
    }
    result = triage(genuine_payment)
    print(json.dumps(result, indent=2))
else:
    print("Skipped: live stack not available.")
""")
md("""\
### Scenario 2 — a borderline case: large night-time cash-out

This one sits right at the edge. Because the model's score distribution is so cleanly separated
on PaySim (Section 5), our cost-optimal STEP_UP band is narrow (Section 7) and a payment with a
score just a hair on either side of it can flip between ALLOW and STEP_UP from one run to the
next, since `step` affects the velocity features and small amount/balance jitter moves the score
slightly. We're keeping this example precisely because it's a realistic edge case, not because it
always lands the same way.
""")
code("""\
if LIVE_STACK_AVAILABLE:
    stepup_payment = {
        "txn_id": "NB-STEPUP-1", "step": 150, "type": "CASH_OUT", "amount": 15000.0,
        "nameOrig": "C-NB-3", "oldbalanceOrg": 20000.0,
        "nameDest": "C-NB-4", "oldbalanceDest": 200.0,
    }
    result = triage(stepup_payment)
    print(json.dumps(result, indent=2))
else:
    print("Skipped: live stack not available.")
""")
md("### Scenario 3 — account takeover (drains the full balance to a fresh account)")
code("""\
if LIVE_STACK_AVAILABLE:
    fraud_payment = {
        "txn_id": "NB-FRAUD-1", "step": 50, "type": "TRANSFER", "amount": 9500.0,
        "nameOrig": "C-NB-5", "oldbalanceOrg": 9500.0,
        "nameDest": "C-NB-NEW", "oldbalanceDest": 0.0,
    }
    result = triage(fraud_payment)
    print(json.dumps(result, indent=2))
else:
    print("Skipped: live stack not available.")
""")
md("""\
Note how the reason codes change between scenarios: the system isn't returning a canned message
per decision type, it's reporting whichever features actually drove *that specific* score, via
SHAP, computed fresh on every request. (See `docs/screenshots/04a_swagger_allow.png` and
`04b_swagger_block.png` in the report for the same calls captured against our deployed stack.)
""")

# ---------------------------------------------------------------- Summary
md("""\
## 10. Summary

| Quality attribute | Target | Result (our deployed system) |
|---|---|---|
| Model accuracy (QA2) | PR-AUC &ge; 0.80, Recall@1%FPR &ge; 0.85 | **0.9997 / 0.9994** — gate passed |
| Performance (QA1) | p95 &le; 150ms, p99 &le; 300ms (load test, 2000 req, concurrency 50) | **148.0ms / 212.2ms** — passed |
| Explainability (QA3) | &le;3 reason codes, &le;10ms overhead | Passed (see `tests/test_quality_explainability.py`) |

**Links:**
- Code: https://github.com/biswa13de/seml-g13-fraud-triage
- MLflow UI (when the stack is running locally): http://localhost:5000
- API docs (when the stack is running locally): http://localhost:8000/docs
- Analyst console (when the stack is running locally): http://localhost:8501

See `PLAN.md` for the full requirements, GR4ML views, architecture and pattern discussion, and
`docs/diagrams/` and `docs/screenshots/` for the diagrams and implementation evidence referenced
in the report.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"name": "13.ipynb", "provenance": []},
}

out_path = "13.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path} with {len(cells)} cells")
