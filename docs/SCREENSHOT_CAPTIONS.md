# Screenshot Captions

Short captions for the report (Section 7, Implementation). Paste each image with its caption directly under it.

**01_compose_ps.png**
All 8 containers are up, and `scoring`, `triage-api`, `redis` and `mlflow` show `(healthy)` from their own healthchecks. Each service can be built, deployed and scaled on its own.

**02_mlflow_comparison.png**
MLflow comparing the 3 trained models on validation PR-AUC (left) and the time to score one payment (right). LightGBM and Random Forest are both at about 1.000 PR-AUC, but LightGBM scores a payment in 0.28 ms vs 3.49 ms for Random Forest, so LightGBM wins under our selection rule.

**03_mlflow_registry.png**
Version 3 of `fraud-triage-model` holds the `champion` alias. The tags record why it was picked (`selection_rule`), the triage thresholds, and the test PR-AUC (0.9997) that passed the quality gate. `scoring-service` loads whichever version holds this alias.

**04a_swagger_allow.png**
A normal ₹500 transfer sent through Swagger. The model gives it a very low risk score and the gateway returns `ALLOW` in 48ms.

**04b_swagger_block.png**
A transfer that empties the sender's full balance into a brand-new, zero-balance account. The model scores it 0.9999 and the gateway returns `BLOCK`, with reason codes that explain why in plain English.

**05_swagger_422.png**
Sending a negative amount is rejected with a 422 before it ever reaches the model — Pydantic schema validation catching bad input at the edge.

**06_redis_stream.png**
`XINFO GROUPS` on the `payment.decided` stream shows all three consumers (`case-service`, `feature-updater`, `monitor`) with `lag: 0`, meaning every event published by the gateway has been picked up by all three independently.

**07_case_queue.png**
The analyst console's case queue, sorted by priority. The ₹45,000 mule case (priority 44,999) is ranked above the ₹9,500 account-takeover case (priority 9,500), so the analyst sees the bigger risk first.

**07b_case_resolved.png**
After clicking "Confirm fraud" on the mule case, it moves to the Resolved tab with the verdict `CONFIRMED_FRAUD`. This verdict is saved as a label for the next retraining round.

**08_monitoring.png**
The monitoring tab after a few test payments: 6 decisions total, 2 ALLOW / 3 BLOCK / 1 STEP_UP, no drift alert yet (PSI needs more than 50 scored payments before it reports a number).

**09a_fallback_degraded.png**
After stopping the scoring container, a new payment still gets a response — `STEP_UP` with `"degraded": true` — instead of the request hanging or failing.

**09b_fallback_recovered.png**
A few seconds after restarting the scoring container, the gateway's own heartbeat check detects it's back up (`scoring_service_healthy: true`) with no restart of the gateway needed.

**10_scaling.png**
`docker compose up --scale scoring=2` adds a second scoring container while everything else keeps running — the model-serving part of the system scales on its own, separately from the rest.

**11a_pytest.png**
All 33 tests pass, covering data quality, the model quality gate, explainability, performance, API contracts, and the event pipeline.

**11b_load_test.png**
2000 requests at 50 concurrent, p95 148.0ms and p99 212.2ms — both inside our 150ms/300ms latency target (QA1).
