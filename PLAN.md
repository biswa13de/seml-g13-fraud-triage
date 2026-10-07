# Group 13 — Assignment I Implementation Plan
## Real-Time Digital Payment Fraud Risk Triage System (FinTech / UPI)

AIMLZG546 Software Engineering for Machine Learning · Weightage 10 marks
**Deadline: Fri 09 Oct 2026, 23:49 (no extensions, no makeup).** Target internal submission: **Fri 09 Oct, 20:00**.

---

## 0. Review of the assignment: what earns full marks

| Q | Ask | Marks (est.) | What a full-marks answer needs | Common way to lose marks |
|---|-----|-----|-----|-----|
| 1 | Domain + problem statement | ~0.5 | Specific actors, a quantified pain point, the scope (in/out), and the reason ML is needed instead of rules | A vague line like "detect fraud using ML" |
| 2 | Requirement spec + measurable goals (GR4ML concepts) | ~1.5 | Numbered FRs and NFRs; layered goals (organisational → system → user → model) with **SMART** measures; assumptions | Goals with no numbers, or numbers with no measurement method |
| 3 | GR4ML Business / Analytics Design / Data Preparation views | ~2 | **The instructor's exact notation** (legend on Session 3 slides 17/20/23), every element type used, consistent names across the views | Drawing plain flowcharts, or leaving out Indicators, Softgoals or Influences |
| 4 | Top-3 quality requirements + justification | ~1 | SMART QAs tied to the domain, a trade-off discussion, and each QA **verified by a test** | Generic "accuracy, scalability, security" with no numbers |
| 5 | Architecture diagram, ML + non-ML | ~1.5 | Every component colour-coded ML vs non-ML; data, event and model flows; legend | Showing only the model and an API |
| 6 | Two architectural patterns | ~1.5 | Each written as a **{Context, Problem, Solution}** triple (the instructor's format, Session 4 slide 29), with the reason it fits *this* system | Naming patterns without justifying them |
| 7 | Implement the patterns | ~2 | Runnable code, `docker compose up`, **screenshots with explanations**, tests that pass | Code that only re-skins the instructor's `fraud_demo` |
| — | Admin | — | `13.pdf` / `13.docx`, `13.ipynb`, group no. and names in every file, contribution table | A missing contribution table, or the wrong file name |

**Differentiation (important).** The instructor's `fraud_demo` already shows a binary fraud classifier, a 4-feature model, FastAPI, MLflow and pipe-and-filter. Many groups will copy it. We go further in three ways:
1. **Triage, not just classification.** The output is a 3-way *action* (ALLOW / STEP_UP / BLOCK) chosen by an expected-cost policy, plus an analyst case queue. This adds a GR4ML **PrescriptionGoal**.
2. **Behavioural features on a public dataset (PaySim)**: payee fan-in (mule) and velocity windows served from an **online feature store**, plus explicit **leakage control**. The demo uses 4 hand-made synthetic columns.
3. **Two distributed patterns, Microservices and Event-Driven Architecture**, plus supporting tactics (heartbeat, fallback rule engine, model registry).

---

## 1. Domain and problem statement (Q1)

**Domain:** Financial Technology. Real-time digital payments (UPI / wallet P2P and P2M) at a mid-size Payment Service Provider (PSP).

**Problem statement (report-ready draft):**
> India's UPI rails process billions of payments a month, and fraud has grown with them: account takeover after SIM-swap or phishing, "collect-request" scams, mule accounts that receive and quickly move stolen funds, and rapid low-value probing. Our PSP currently uses static rules (amount limits, blocklists). These rules **miss new fraud patterns** and **decline too many genuine payments**. Every flagged payment goes to an undifferentiated manual-review queue, so analysts spend their time on low-risk alerts while high-value fraud settles before anyone looks at it.
>
> We propose a **Real-Time Fraud Risk Triage System**. For each payment, before authorisation, it (a) estimates fraud probability with an ML model using transaction and behavioural features, (b) maps that risk to a cost-optimal action, **ALLOW**, **STEP_UP** (extra authentication such as OTP or biometric) or **BLOCK**, inside a strict latency budget, (c) returns human-readable reason codes, and (d) opens prioritised cases for fraud analysts, whose verdicts flow back as labels for retraining.

**Why ML instead of rules (Kästner ch. 4: "when to use ML"):** fraud patterns change adversarially, the signal comes from many weak interacting features, labelled history exists (chargebacks and complaints), and an occasional wrong answer is tolerable because STEP_UP and human review act as safety nets.

**Scope.** In scope: scoring, triage decision, reason codes, case queue, monitoring. Out of scope: the actual NPCI switch integration, customer KYC, chargeback processing (simulated).

---

## 2. Requirements and measurable goals (Q2)

### 2.1 Layered goals (Kästner ch. 6, "Setting and measuring goals")
| Layer | Goal | Measure (SMART) | How measured |
|---|---|---|---|
| Organisational | Reduce fraud losses | Fraud loss ≤ **2 bps** of payment value within 6 months (baseline 5 bps) | Monthly finance report |
| Organisational | Protect customer experience | False-decline rate ≤ **0.5%** of genuine payments | Declines later confirmed genuine |
| System | Intervene before money moves | ≥ **85%** of fraud *value* receives STEP_UP or BLOCK | Offline replay on test month |
| System | Decide in real time | p95 end-to-end decision latency ≤ **150 ms**, p99 ≤ 300 ms | Load test plus `/metrics` |
| User (analyst) | Work the riskiest cases first | ≥ 80% of confirmed-fraud cases appear in the top 20% of the queue by priority | Case-service analytics |
| User (customer) | Little friction | STEP_UP rate ≤ **3%** of all payments | Decision-mix metric |
| Model | Rank fraud well on imbalanced data | PR-AUC ≥ **0.80**; Recall ≥ **85% at FPR ≤ 1%** | Time-based hold-out test |
| Model | Stay current | Retrain when score PSI > 0.2 or weekly | Monitoring service |

### 2.2 Functional requirements
- **FR1** Score each payment request and return `{decision, risk_score, reason_codes[3], model_version, latency_ms}`.
- **FR2** Apply deterministic hard rules first: blocklisted payee VPA → BLOCK; amount above the per-transaction limit → BLOCK.
- **FR3** Map the score to ALLOW / STEP_UP / BLOCK using configurable thresholds, chosen by expected cost.
- **FR4** Publish every decision as a `payment.decided` event.
- **FR5** Create an analyst case for every BLOCK and for STEP_UP above ₹10,000. Priority = score × amount.
- **FR6** Let analysts list cases and resolve them (`CONFIRMED_FRAUD` / `GENUINE`). Each resolution becomes a label.
- **FR7** Update the payer's behavioural (velocity) features after every payment.
- **FR8** Monitor decision mix, score distribution drift (PSI) and latency, and raise an alert when thresholds are breached.
- **FR9** If the ML scorer is unhealthy, fall back to rules only and return `degraded=true`. Never hang.
- **FR10** Version every model in a registry. Serving loads the `champion` alias.

### 2.3 Assumptions (world vs machine, Kästner ch. 7)
- A1 Fraud labels arrive late (up to 30 days), so training uses only mature labels.
- A2 Fraudsters adapt, so the data distribution drifts and retraining is required.
- A3 Upstream sends a well-formed payment message with payer/payee account IDs, type, amount and pre-transaction balances.
- A4 Fraud base rate is about 0.1–1%. In PaySim it is ~0.13% overall and ~0.3% within TRANSFER/CASH_OUT.
- A5 PaySim (simulated mobile money) is a reasonable public proxy for UPI P2P payments. Device and geo signals are not available, so a real deployment would add them.

### 2.4 Requirements → GR4ML traceability
Map each FR to a GR4ML element: FR1 → Question goal Q1 / Insight; FR3 → PrescriptionGoal; FR5–6 → Decision goal D2; FR7 → Data Preparation (windowed aggregation). This table earns marks for consistency.

---

## 3. GR4ML views (Q3)

**Tool:** draw.io (diagrams.net). Copy the shapes **exactly** from the legends on Session 3 slides 17, 20 and 23. Store `.drawio` and exported `.png` files in `docs/diagrams/`. Put a legend box on each diagram, as the instructor does.

### 3.1 Business View
| Element | Content |
|---|---|
| **Actors** (stick figure) | Fraud Risk Manager; Fraud Analyst; Customer (Payer); Compliance Officer |
| **Business goals** (ellipse) | G0 *Grow digital payments safely* → **AND** → G1 *Reduce fraud losses*, G2 *Keep payments frictionless*, G3 *Resolve flagged cases quickly* |
| **desires** | Risk Manager desires G1; Customer desires G2; Analyst desires G3; Compliance desires G0 |
| **Indicators** (traffic light, `evaluates` ‖‖) | G1: Fraud loss (bps), target 2 / threshold 3 / worst 5. G2: False-decline rate, 0.5%. G3: Mean case resolution time, ≤ 4 h |
| **Decision goals** (D) | D1 *Decide action for an incoming payment (Allow / Step-up / Block)* (AND under G1 and G2); D2 *Decide which flagged case to review first* (under G3); D3 *Decide whether to block the payee VPA* |
| **Question goals** (Q) | Q1 *What is the probability that this payment is fraudulent?* (→ D1). Q2 *Which factors make this payment risky?* (→ D1, D3). Q3 *What is the expected loss of this flagged case?* (→ D2) |
| **Insights** (box with attributes, `answers` --▷) | **Payment Fraud Risk Predictive Model**: type Predictive; input Payment + behavioural profile; output Fraud probability (0–1); usageFrequency Per transaction (real-time); updateFrequency Weekly; learningPeriod Last 30 days (rolling) → answers Q1. **Risk Reason Codes**: type Explanatory; output Top-3 feature contributions → answers Q2. **Case Priority Score**: score × amount → answers Q3 |

### 3.2 Analytics Design View
| Element | Content |
|---|---|
| **Analytics goals** | AG1 (PredictionGoal) *Classification of payment as fraud/genuine*. AG2 (PrescriptionGoal) *Recommend triage action minimising expected cost*. AG3 (DescriptionGoal) *Explain the drivers of each score* |
| **Indicators on AG1** | PR-AUC (≥ 0.80), Recall@1%FPR (≥ 85%), Inference latency p95 (≤ 50 ms) |
| **Algorithms** (hexagon, `performs` ▶) | Logistic Regression; Random Forest; **LightGBM** (gradient-boosted trees); Isolation Forest (unsupervised). For AG2: cost-based threshold optimisation. For AG3: SHAP TreeExplainer |
| **Softgoals** (cloud) | Detection accuracy on imbalanced data; Low inference latency; Interpretability; Tolerance to missing values; Adaptability to drift (cheap retraining) |
| **Influences** (dotted, +/−/++/−−) | LR: interpretability ++, latency ++, accuracy −. RF: accuracy +, latency −, missing values −. LightGBM: accuracy ++, latency +, missing values ++ (native), interpretability − (raised to + by SHAP). IsoForest: accuracy −− (no labels used), adaptability + |
| **Satisfied / Denied** | ✔ LightGBM + SHAP (champion); ✔ LR (baseline/challenger); ✘ RF (too slow); ✘ IsoForest (too weak alone) |

The algorithms in this view must be the **same ones trained and compared in MLflow**. Screenshot the MLflow comparison table and refer to it from this view. That closes the loop between design and implementation.

### 3.3 Data Preparation View
| Element | Content |
|---|---|
| **Entities** (PK bold) | `PaySimTransaction` (**txn_id** (row index), step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud). Derived reference entities: `PayerAccount` (**nameOrig**, first_seen_step) and `PayeeAccount` (**nameDest**, is_merchant = prefix `M`, on_blocklist). `AnalystLabel` (**txn_id**, verdict, resolved_at), the feedback loop from the case service |
| **Output entity** | `TrainingFeatureTable` (**txn_id**, ~16 pre-transaction features, is_fraud) |
| **Operators** (box, data flow →) | **Filter**: `type IN ('TRANSFER','CASH_OUT')` (note: SQL `WHERE`; fraud exists only in these types). **Projection / leakage removal**: drop `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud`. **Clean**: dedupe, check amount > 0, check non-negative balances. **Sampling**: keep all fraud plus a time-stratified sample of genuine transactions (train only). **Windowed aggregation** (point-in-time correct, no future leakage): dest_txn_count_24h, dest_amount_sum_24h, dest_distinct_senders_24h, orig_txn_count_24h. **Derive**: hour_of_day, is_night, amount_to_balance_ratio, drains_account, zero-balance flags, dest_is_merchant, log_amount. **Encode**: one-hot type. **Split**: time-based on `step`. **Imbalance**: class weights (`scale_pos_weight`) |
| **Notes** (dog-ear) | Point-in-time correctness. Post-transaction balance columns removed because they leak. The same feature code is used for training and serving (no training–serving skew). Source: Kaggle PaySim, with checksum recorded |

---

## 4. Top-3 quality requirements (Q4)

Format each one as a **QA scenario** (source, stimulus, environment, response, measure). Each has a pytest that fails if the QA is violated.

1. **Performance (latency and throughput).** *Under 100 payments/s on normal load, the system returns a triage decision with p95 ≤ 150 ms and p99 ≤ 300 ms. Model inference p95 ≤ 50 ms.*
   **Why:** the decision sits inside the payment authorisation path. A slow risk check makes payments time out, which hurts the customer and the PSP's success-rate metrics. That is a business loss even with perfect accuracy.
   **Design response:** model loaded in memory once, feature store in Redis (O(1) lookups), async side-effects through events (FR4) so cases and monitoring add no latency, compact tree model.
   **Verified by:** `tests/test_quality_performance.py` and `scripts/load_test.py`.
2. **Model accuracy (cost-sensitive detection).** *On a time-based hold-out month, Recall ≥ 85% at FPR ≤ 1% and PR-AUC ≥ 0.80. Precision of BLOCK decisions ≥ 90%.*
   **Why:** a missed fraud is a direct monetary loss. A false BLOCK loses a customer. Plain accuracy is meaningless at a 1% base rate, so we use PR-AUC and recall at a fixed FPR.
   **Design response:** LightGBM with class weights, expected-cost thresholds, a 3-way decision (STEP_UP absorbs uncertainty), an analyst feedback loop.
   **Verified by:** `tests/test_quality_model.py`, a quality gate that blocks registration as `champion`.
3. **Explainability.** *Every STEP_UP or BLOCK decision returns the top-3 reason codes in plain language (for example, "Transfer drains 100% of the sender's balance"), adding ≤ 10 ms p95.*
   **Why:** analysts must justify actions on customer complaints and disputes. Regulators and auditors expect decisions on customer funds to be explainable. Reason codes also speed up case resolution (G3).
   **Design response:** SHAP TreeExplainer and a feature-to-text mapping.
   **Verified by:** `tests/test_quality_explainability.py`.

**Also considered, ranked lower** (write a short trade-off table): availability/reliability (handled by the architecture: heartbeat and fallback rules, FR9), robustness, fairness (no demographic features are used; we still check for a region proxy), privacy (hash VPAs in logs), scalability. Explain the tensions: accuracy vs latency (a bigger model is slower), explainability vs latency (SHAP cost), accuracy vs friction (the threshold trade-off).

---

## 5. System architecture (Q5)

```
                         ┌──────────────────────── NON-ML ───────────────────────────┐
 Payment App / PSP  ──▶  │ triage-api (Gateway + Decision Orchestrator)  :8000         │
 (simulator / UI)        │  • API-key auth, request-id, schema validation (Pydantic)  │
                         │  • Rule engine (blocklist, limits)        [non-ML]         │
                         │  • Heartbeat check on scorer → fallback   [non-ML]         │
                         │  • Triage policy (thresholds → action)    [non-ML]         │
                         └────────┬──────────────────────────────┬────────────────────┘
                     sync HTTP    │                              │ publish payment.decided
                                  ▼                              ▼
                ┌──────── ML ───────────────────┐     ┌──── Event bus: Redis Streams ────┐
                │ scoring-service :8001          │     │ stream: payment.decided           │
                │ validate → features → predict  │     └──┬─────────┬──────────┬──────────┘
                │ → explain (pipe-and-filter)    │        │         │          │ consumer groups
                │ loads model@champion (MLflow)  │        ▼         ▼          ▼
                └───┬───────────────┬────────────┘   case-service feature-  monitor-service
                    │ read features │ load model      :8002 (SQLite updater   :8003 (PSI drift,
                    ▼               ▼                 cases, analyst (writes   decision mix,
            ┌─ Online Feature ─┐ ┌─ MLflow Tracking  verdicts =    velocity latency, alerts)
            │ Store (Redis)    │ │  + Model Registry  labels)      features)    [ML-ops]
            └──────────────────┘ └─▲───────────────┘
                                   │ register (if quality gate passes)
             Offline: data generator → feature pipeline → training (LR / RF / LGBM) → evaluation
             Analyst Console (Streamlit :8501): submit payments, case queue, metrics   [non-ML]
```

Draw this properly in draw.io for the report. **Colour code**: ML components (scoring, training pipeline, model registry, feature store, drift monitor) in one colour, non-ML (gateway, rules, policy, case management, event bus, UI, auth, logging) in another. Mark sync vs async arrows differently. Add a legend.

---

## 6. Architectural patterns (Q6)

Write each as **{Context, Problem, Solution}**, add "Why it fits this system", then "Consequences / trade-offs".

### Pattern 1: Microservices (Session 5)
- **Context:** a fraud platform with parts that change and scale at different rates. The model is retrained weekly, rules change daily, case management changes rarely.
- **Problem:** in a monolith, a model update means redeploying everything, scoring load can't be scaled on its own, and a bug in case management can take down authorisation (single point of failure, Session 5 slide 27).
- **Solution:** split by single responsibility into `triage-api`, `scoring-service`, `case-service`, `feature-updater` and `monitor-service`, each with its own API or data and its own container.
- **Fits because:** scoring is the hot path, so we scale it on its own (`docker compose up --scale scoring=3`). The model can be swapped without touching the gateway. Different owners: the data science team owns scoring, the fraud ops team owns cases.
- **Trade-offs:** network hops add latency (measured, within budget), more operational overhead, distributed failures. That last one is why we add the **heartbeat tactic** and a **fallback**.

### Pattern 2: Event-Driven Architecture (publish-subscribe) (Session 6)
- **Context:** each decision triggers several follow-up actions: open a case, update velocity features, update monitoring, (notify the customer).
- **Problem:** doing these synchronously adds latency to the authorisation path (violating QA1) and couples the gateway to every downstream service.
- **Solution:** `triage-api` publishes one `payment.decided` event to Redis Streams. Independent **consumer groups** subscribe, with at-least-once delivery, acknowledgement (`XACK`) and replay of pending messages after a crash.
- **Fits because:** it keeps the hot path fast (QA1), lets us add consumers without changing the producer (for example a future notification service), and tolerates a consumer being down.
- **Trade-offs:** eventual consistency. Velocity features lag by milliseconds; we say so explicitly and it's acceptable. Consumers must be idempotent: dedupe on `txn_id`.

### Supporting patterns and tactics (mention briefly; don't call them the "two patterns")
Pipe-and-filter inside the scoring service · Feature Store pattern · Model Registry pattern (MLflow aliases) · Heartbeat tactic · Real-time serving pattern · Fallback/degradation (rules-only mode).

---

## 7. Implementation (Q7)

### 7.1 Tech stack
Python 3.11 (Docker images) · FastAPI + Pydantic v2 · LightGBM, scikit-learn, SHAP · MLflow (tracking + registry with aliases) · Redis 7 (Streams = event bus, Hashes/Sorted sets = online feature store) · SQLite (cases) · Streamlit (analyst console) · pytest · Docker Compose · ruff · GitHub Actions (CI: lint + tests).

### 7.2 Repository layout
```
seml-g13-fraud-triage/
├── README.md                  # run instructions, architecture image, group details
├── PLAN.md                    # this file
├── docker-compose.yml         # redis, mlflow, triage-api, scoring, case, feature-updater, monitor, console
├── Makefile                   # make data | train | up | test | load
├── pyproject.toml / requirements*.txt
├── 13.ipynb                   # implementation notebook (data → EDA → training → eval → API demo)
├── common/                    # shared package (prevents training–serving skew)
│   ├── schemas.py             # PaymentRequest, TriageDecision, PaymentDecidedEvent
│   ├── features.py            # ONE feature function used by training and serving
│   ├── feature_store.py       # Redis read/write of velocity features
│   ├── events.py              # Redis Streams publish/consume helpers
│   └── config.py, logging.py  # pydantic-settings, JSON logs
├── data/download_paysim.py    # Kaggle download + checksum + sampling
├── training/
│   ├── build_features.py      # point-in-time windowed features, time-based split
│   ├── train.py               # LR / RF / LightGBM → MLflow runs
│   ├── select_thresholds.py   # expected-cost threshold search
│   └── register.py            # quality gate → register + alias "champion"
├── services/
│   ├── triage_api/            # gateway: auth, rules, heartbeat, policy, publish
│   ├── scoring/               # pipe-and-filter: validate → features → predict → explain
│   ├── case_service/          # consumer + REST (cases, resolve)
│   ├── feature_updater/       # consumer → feature store
│   └── monitor/               # consumer → PSI, decision mix, latency, alerts
├── console/app.py             # Streamlit analyst console
├── scripts/load_test.py       # async httpx load test → p50/p95/p99
├── tests/                     # data, model, contract, QA, event, fallback tests
└── docs/diagrams/             # GR4ML views + architecture (.drawio + .png), screenshots/
```

### 7.3 Step-by-step build

**Step 1: Public dataset, PaySim** (`data/download_paysim.py`). Kaggle `ealaxi/paysim1`, ~470 MB CSV, 6.36M mobile-money transactions over 743 hourly steps (30 days), ~0.13% fraud.
- **Why PaySim:** it is the best-known public *mobile-money / P2P digital payment* dataset. It has sender and receiver account IDs (needed for velocity and mule features), transaction types and time. The credit-card dataset (`mlg-ulb`) has anonymised PCA features, so it can't produce meaningful reason codes. IEEE-CIS is card-not-present e-commerce and very heavy.
- **Mapping to UPI (state this in the report):** `nameOrig` → payer VPA, `nameDest` → payee VPA (`M…` = merchant), `TRANSFER` → P2P transfer, `CASH_OUT` → cash-out via agent, `PAYMENT` → P2M, `step` → hour of the month.
- **Download:** the Kaggle API token goes in `~/.kaggle/kaggle.json` (chmod 600), never in the repo. A SHA-256 checksum is recorded for provenance. The CSV stays out of git (`data/*.csv` is ignored).
- **Leakage control (an important SE-for-ML point to write up):** `newbalanceOrig` and `newbalanceDest` are **post-transaction** values that are not known at authorisation time, so we **drop them**. `isFlaggedFraud` is the legacy rule's output, so it isn't used as a feature. We keep it only as the **baseline rule** to compare against.
- **Known data limitations:** payers rarely repeat in PaySim, so payer velocity is weak and payee fan-in (mule) features carry the behavioural signal. Fraud appears only in TRANSFER and CASH_OUT. PAYMENT, DEBIT and CASH_IN are ALLOWed by rule and excluded from training (document this as a design decision).
- **Working sample** for laptop speed: all fraud plus a time-stratified sample of genuine TRANSFER/CASH_OUT (~1M rows). Keep the true base rate in validation and test so the metrics are honest.

**Step 2: Features** (`common/features.py`, `training/build_features.py`)
- Pre-transaction features only: `amount`, `log_amount`, `type_TRANSFER`, `oldbalanceOrg`, `amount_to_balance_ratio` (amount / oldbalanceOrg), `drains_account` (amount ≥ 0.99 × oldbalanceOrg), `orig_zero_balance`, `oldbalanceDest`, `dest_zero_balance`, `dest_is_merchant`, `hour_of_day` (step mod 24), `is_night`, plus point-in-time velocity from the online feature store: `dest_txn_count_24h`, `dest_amount_sum_24h`, `dest_distinct_senders_24h`, `orig_txn_count_24h`.
- One `compute_features(payment, store_state) -> dict` function, called by both training (replaying history in step order) and serving (state read from Redis). A test checks that they produce identical outputs.
- Time-based split: steps 1–500 train, 501–600 validation, 601–743 test.
- If PR-AUC comes out near 0.99, check for leakage first. Then report honestly that PaySim is simulated and relatively separable, and lean on the cost and latency results.

**Step 3: Training and experiment tracking** (`training/train.py`)
- Train LR (scaled, `class_weight='balanced'`), RF, LightGBM (`scale_pos_weight`). Log params, PR-AUC, ROC-AUC, recall@1%FPR, inference latency per row, and the PR curve image to MLflow.
- SHAP on LightGBM: a global summary plot as an artifact.

**Step 4: Cost-based thresholds** (`training/select_thresholds.py`)
- Expected cost = Σ fraud amounts ALLOWed + ₹50 × genuine STEP_UPs (friction) + ₹500 × genuine BLOCKs (churn) + ₹20 × fraud caught only by STEP_UP. Grid-search `t_stepup < t_block` on validation, subject to STEP_UP rate ≤ 3%. Save to `thresholds.json` as a model artifact so thresholds are versioned with the model.

**Step 5: Quality gate and registry** (`training/register.py`)
- If test PR-AUC ≥ 0.80 and recall@1%FPR ≥ 0.85, register `fraud-triage-model` and set alias `champion`. Otherwise exit non-zero.

**Step 6: scoring-service** (pipe-and-filter): `validate → load_features (Redis) → compute_features → predict_proba → explain (SHAP top-3 → reason text)`. Endpoints: `POST /score`, `GET /health` (status, model_version, run_id, trained_at, uptime). The heartbeat payload completes the instructor's take-home challenge as well.

**Step 7: triage-api** (gateway): `POST /v1/payments/triage`
1. API-key check → 2. Pydantic validation → 3. Hard rules → 4. Is the scorer healthy (heartbeat cache, 2 s interval)? If yes, call `/score` with a 100 ms timeout; on timeout or unhealthy, use rules-only with `degraded=true` → 5. Policy → decision → 6. `XADD payment.decided` → 7. Respond.
Also `GET /health`.

Example response:
```json
{"txn_id":"T123","decision":"STEP_UP","risk_score":0.62,
 "reason_codes":["Transfer drains 100% of sender balance","Receiver got 14 payments from 11 senders in 24h","Receiver balance was 0 before payment"],
 "model_version":"3","degraded":false,"latency_ms":41.7}
```

**Step 8: Consumers** (each in its own consumer group, idempotent on txn_id)
- `case-service`: creates a case when the FR5 rule matches. `GET /cases?status=OPEN` is sorted by priority. `POST /cases/{id}/resolve` stores the label in a `labels` table, which is the feedback loop.
- `feature-updater`: updates Redis sorted sets for 1 h / 24 h windows, per-payee distinct senders and amount sums (fan-in) and per-payer counts.
- `monitor-service`: rolling decision mix, score histogram, PSI against the training baseline, p95 latency. `GET /metrics`. Logs a `DRIFT_ALERT` when PSI > 0.2.

**Step 9: Analyst console** (Streamlit). Tabs: *Simulate Payment* (form plus presets: genuine, ATO, collect scam, probing), *Case Queue* (resolve buttons), *Monitoring* (decision mix, PSI, latency). This is where most screenshots come from.

**Step 10: Run modes (Docker is not mandatory).**
- **Primary: Docker Compose.** Services: redis, mlflow, triage-api, scoring, case-service, feature-updater, monitor, console. Add healthchecks and `depends_on: condition: service_healthy`. Demo `--scale scoring=2`.
- **Fallback: `make run-local`.** Starts every service with `uvicorn` / `streamlit` in a local venv. It needs only Redis (`brew install redis`). Tests use `fakeredis`, so `pytest` runs with no infrastructure at all.

**Step 11: Tests** (`pytest -v`. The output screenshot goes in the report.)
| Test file | What it proves |
|---|---|
| `test_data_quality.py` | Schema, no duplicate txn_id, fraud rate within 0.5–2%, no future leakage in windowed features |
| `test_features_parity.py` | Training and serving feature functions give identical output (no skew) |
| `test_quality_model.py` | QA2: PR-AUC and recall@1%FPR gates; invariance (changing payer ID doesn't change the score); directional (draining the balance into a high fan-in receiver raises risk) |
| `test_quality_performance.py` | QA1: p95 latency of the triage endpoint |
| `test_quality_explainability.py` | QA3: 3 reason codes for every STEP_UP/BLOCK; overhead ≤ 10 ms |
| `test_contract.py` | Bad payload → 422; response schema |
| `test_fallback.py` | Scorer down → rules-only, `degraded=true`, still < 150 ms |
| `test_events.py` | A published decision creates a case and updates features (fakeredis) |

**Step 12: Load test** (`scripts/load_test.py`): 5,000 requests at concurrency 50 → p50/p95/p99 table. This is the QA1 evidence.

**Step 13: CI.** A GitHub Actions workflow runs ruff and pytest on every push. The green tick is worth a screenshot.

**Step 14: Notebook `13.ipynb`.** Group details cell → problem → dataset download, provenance and EDA → features → 3-model comparison (MLflow) → thresholds and cost curve → SHAP → calls to the running API (3 scenario payments) → links to the services.

### 7.4 Screenshot checklist for the report (each needs 2–3 lines of explanation)
1. `docker compose ps`, all services healthy
2. MLflow experiment comparison (3 algorithms) and the model registry showing the `champion` alias
3. Swagger `/docs` for triage-api, plus a live ALLOW, STEP_UP and BLOCK response
4. 422 validation error
5. Redis Stream (`XINFO GROUPS payment.decided` or `XRANGE`), showing events and consumer groups
6. Case queue in the console, plus resolving a case
7. Monitoring tab / `/metrics` (decision mix, PSI, latency)
8. Fallback demo: `docker compose stop scoring` → `degraded=true` response
9. Scaling demo: `--scale scoring=2`
10. `pytest -v` all green; load-test p95 table; GitHub Actions green
11. SHAP summary plot and a per-transaction reason-code example

---

## 8. Report structure (`13.pdf`, also export `13.docx`)
1. Cover: course, Assignment I, **Group 13**, member table (BITS ID, name, qualitative contribution, %).
2. Q1 Domain and problem statement.
3. Q2 Requirements: goal table, FR/NFR, assumptions, traceability.
4. Q3 GR4ML: 3 diagrams, each with an element-by-element explanation.
5. Q4 Top-3 QAs: scenarios, justification, trade-off table, linked tests.
6. Q5 Architecture diagram and component table (component | ML/non-ML | responsibility | tech).
7. Q6 Two patterns: C-P-S triples, fit, trade-offs.
8. Q7 Implementation: repo link, how to run, code excerpts, screenshots with explanations, test and load results.
9. Limitations and future work (real data, Kafka, Kubernetes, canary/shadow deployment, Evidently drift reports).
10. Appendix: key source code (the submission needs code in the document).

---

## 9. Git workflow
- Private GitHub repo `seml-g13-fraud-triage`. Add teammates as collaborators. Make it public only after the deadline, if at all, so other groups can't copy it.
- Small, frequent commits in conventional style (`feat(scoring): ...`, `test: ...`, `docs(gr4ml): ...`). Every completed step in §7.3 gets a commit and a push.
- Tags: `v0.1-requirements`, `v0.5-services`, `v1.0-submission`.
- Never commit `mlruns/`, `*.db`, generated CSVs over 50 MB, `.env`, or the `venv`.

## 10. Schedule (≈ 2.5 days)
| When | Work |
|---|---|
| **Wed 7 Oct (evening)** | Answer the open questions, set up the repo and remote. Steps 1–5 (data, features, training, thresholds, registry). Draft the GR4ML views in draw.io |
| **Thu 8 Oct** | Steps 6–11 (services, events, console, compose, tests). Architecture diagram |
| **Fri 9 Oct** | Steps 12–14, screenshots, write the report, contribution table, review. **Submit by 20:00**, leaving buffer before 23:49 |

## 11. Team contribution
The contribution table must be **truthful**. Even if the lead builds most of it, give each member a real, verifiable piece, for example: Member 2 draws and reviews the GR4ML views, Member 3 does the QA tests, load test and screenshots, Member 4 writes the report sections and reviews. Their commits in the repo back up the percentages.
