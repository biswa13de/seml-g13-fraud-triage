# Screenshot Guide — Group 13

The stack is running right now in a clean demo state (seeded with 4 scenario payments: 1 ALLOW, 1 resolved STEP_UP, 2 open BLOCK cases). Take these 11 screenshots in order and save each as `docs/screenshots/<number>_<name>.png`. Each one maps to an item in PLAN.md §7.4 and should get 2-3 lines of your own explanation in the report — don't just paste the image.

**Before you start:** confirm the stack is up —
```bash
cd seml-g13-fraud-triage && docker compose ps
```
All 8 services should show `Up`. If anything is missing, run `docker compose up -d`.

---

### 1. `docker compose ps` — all services healthy
Terminal screenshot. Run:
```bash
docker compose ps
```
Save as `01_compose_ps.png`. **Explain:** 8 independent containers, each a separate deployable unit (Microservices pattern); `scoring` and `triage-api` show `(healthy)` from their own healthchecks.

### 2. MLflow — model comparison
Open **http://localhost:5000** → click the `fraud-triage` experiment → select all 3 runs (logistic_regression, random_forest, lightgbm) → click **Compare**.
Save as `02_mlflow_comparison.png`. **Explain:** PR-AUC and inference latency side by side across the 3 algorithms trained in `training/train.py` — this is the evidence behind the Analytics Design View's softgoal choices.

### 3. MLflow — registered model with "champion" alias
In MLflow, go to **Models → fraud-triage-model**. You should see version 1 with the alias `champion`.
Save as `03_mlflow_registry.png`. **Explain:** the Model Registry pattern — `training/register.py` only promotes a model here if it clears the PR-AUC ≥ 0.80 / Recall@1%FPR ≥ 0.85 quality gate.

### 4. Swagger — triage-api live calls
Open **http://localhost:8000/docs**. Click **Authorize**, enter `demo-key-g13` as the API key. Expand `POST /v1/payments/triage`, click **Try it out**, and submit this genuine payment:
```json
{"txn_id":"T-DEMO-ALLOW","step":120,"type":"TRANSFER","amount":500,"nameOrig":"C1","oldbalanceOrg":5000,"nameDest":"C2","oldbalanceDest":3000}
```
You should see `"decision":"ALLOW"`. Then submit this one and show the `BLOCK` response:
```json
{"txn_id":"T-DEMO-BLOCK","step":50,"type":"TRANSFER","amount":9500,"nameOrig":"C3","oldbalanceOrg":9500,"nameDest":"C4","oldbalanceDest":0}
```
Save both as `04a_swagger_allow.png` and `04b_swagger_block.png`. **Explain:** same endpoint, two different reason-code sets depending on the transaction's own numbers — nothing hardcoded per scenario.

### 5. Swagger — 422 validation error
In the same Swagger page, submit an invalid payload (negative amount):
```json
{"txn_id":"T-BAD","step":50,"type":"TRANSFER","amount":-500,"nameOrig":"C1","oldbalanceOrg":1000,"nameDest":"C2","oldbalanceDest":0}
```
Save as `05_swagger_422.png`. **Explain:** Pydantic schema validation rejects malformed input before it reaches the model — part of QA1/robustness.

### 6. Redis Stream — events and consumer groups
Terminal screenshot. Run:
```bash
docker compose exec redis redis-cli XLEN payment.decided
docker compose exec redis redis-cli XINFO GROUPS payment.decided
```
Save as `06_redis_stream.png`. **Explain:** three independent consumer groups (`case-service`, `feature-updater`, `monitor`) all reading the same stream — the Event-Driven / publish-subscribe pattern; `lag: 0` on each shows nothing was missed.

### 7. Case queue — console
Open **http://localhost:8501**, click the **Case Queue** tab, leave filter on "OPEN". You should see 2 open BLOCK cases sorted by priority (the ₹45,000 mule case above the ₹9,500 account-takeover case).
Save as `07_case_queue.png`. Then click **Confirm fraud** on one of them and screenshot the queue again (now 1 open) as `07b_case_resolved.png`. **Explain:** priority = risk_score × amount, so the highest-value fraud surfaces first for the analyst; resolving a case writes the verdict as a label (FR6 feedback loop).

### 8. Monitoring tab — console
In the console, click the **Monitoring** tab.
Save as `08_monitoring.png`. **Explain:** decision mix bar chart and PSI drift number, both computed from the same event stream in real time by `monitor-service`.

### 9. Fallback demo — scoring service down
Terminal screenshots, two steps:
```bash
docker compose stop scoring
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/payments/triage -H "Content-Type: application/json" -H "X-API-Key: demo-key-g13" -d '{"txn_id":"T-DEGRADED","step":60,"type":"TRANSFER","amount":1000,"nameOrig":"C1","oldbalanceOrg":5000,"nameDest":"C2","oldbalanceDest":2000}'
```
Save as `09a_fallback_degraded.png` — note `"degraded":true` and the sub-millisecond latency (never hangs). Then:
```bash
docker compose start scoring
sleep 8
curl -s http://localhost:8000/health
```
Save as `09b_fallback_recovered.png` — `scoring_service_healthy` flips back to `true` with no restart. **Explain:** the heartbeat tactic (FR9) detects the outage within a few seconds and the gateway falls back to a safe rules-only decision instead of timing out.

### 10. Scaling demo — Microservices pattern
Terminal screenshot:
```bash
docker compose up -d --scale scoring=2 scoring
docker compose ps scoring
```
Save as `10_scaling.png`. **Explain:** the scoring service scales independently of the rest of the system — this is also what measurably improved the load-test p95 from 158.8ms to 132.8ms (see load test below).

### 11. Tests, load test, and CI — all green
Three terminal screenshots:
```bash
make test        # or: MLFLOW_TRACKING_URI=http://localhost:5000 .venv/bin/python -m pytest tests/ -v
```
Save as `11a_pytest.png` — 33 passed.
```bash
.venv/bin/python scripts/load_test.py --n 2000 --concurrency 50
```
Save as `11b_load_test.png` — p95/p99 numbers against the QA1 budget.
Then push to GitHub and open the **Actions** tab on the repo page, screenshot the green check as `11c_ci.png`.

---

## After capturing

Scale back down and reset to a normal single-replica state before your next work session:
```bash
docker compose up -d --scale scoring=1 scoring
```

Save all images into `docs/screenshots/`, then let me know and I'll fold them into the report draft with the explanations above as a starting point for each caption.
