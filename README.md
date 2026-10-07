# Real-Time Digital Payment Fraud Risk Triage System

AIMLZG546 Software Engineering for Machine Learning · Assignment I · **Group 13**

## Group details

| Sl. No | BITS ID | Name | Contribution (qualitative) | % Contribution |
|---|---|---|---|---|
| 1 | `<BITS_ID_1>` | `<NAME_1>` (Lead) | `<TBD>` | `<TBD>` |
| 2 | `<BITS_ID_2>` | `<NAME_2>` | `<TBD>` | `<TBD>` |
| 3 | `<BITS_ID_3>` | `<NAME_3>` | `<TBD>` | `<TBD>` |
| 4 | `<BITS_ID_4>` | `<NAME_4>` | `<TBD>` | `<TBD>` |

## What it does

For every digital payment (UPI-style P2P / cash-out), the system scores fraud risk in real time and triages the payment to **ALLOW**, **STEP_UP** (extra authentication) or **BLOCK**. It returns the top-3 reason codes and opens prioritised cases for fraud analysts.

Architectural patterns: **Microservices** and **Event-Driven Architecture** (Redis Streams). See [PLAN.md](PLAN.md) for the full design.

## Dataset

Kaggle [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1), a public simulated mobile-money transaction log (6.36M rows, 8,213 frauds, licence CC-BY-SA-4.0). The CSV is not committed to git; its SHA-256 checksum is recorded in [data/provenance.json](data/provenance.json).

1. Kaggle → Settings → API → create a token. Save it as `~/.kaggle/access_token` (or a legacy `~/.kaggle/kaggle.json`).
2. Protect it: `chmod 600 ~/.kaggle/access_token`
3. Download:
   ```bash
   pip install kaggle
   python data/download_paysim.py
   ```

## Architecture

```
Payment App/Console → triage-api (gateway: rules, heartbeat, policy) → scoring (ML: pipe-and-filter + SHAP)
                              │                                              │
                              └──────────── publish payment.decided ────────┘
                                         (Redis Streams, pub-sub)
                                    ↓              ↓              ↓
                              case-service   feature-updater   monitor
```
Patterns: **Microservices** (independently deployable/scalable services) + **Event-Driven Architecture**
(Redis Streams pub-sub). Full design in [PLAN.md](PLAN.md).

## Running (Docker — primary path)

```bash
docker compose up -d redis mlflow        # 1. bring up infra first
make train                               # 2. train 3 models, pick cost-based thresholds, register champion
docker compose up -d --build             # 3. bring up every service
```

- Gateway: http://localhost:8000/docs (header `X-API-Key: demo-key-g13`)
- Analyst console: http://localhost:8501
- Case service: http://localhost:8002/cases
- Monitoring: http://localhost:8003/metrics
- MLflow UI: http://localhost:5000

Scale the hot path independently (Microservices pattern demo):
```bash
docker compose up -d --scale scoring=2 scoring
```

Simulate the fallback tactic (Event-Driven + heartbeat demo):
```bash
docker compose stop scoring   # triage-api keeps answering, degraded=true, rules-only
docker compose start scoring  # heartbeat self-heals within ~2-6s, no restart needed
```

Run the full QA test suite and the latency load test:
```bash
make test     # needs `docker compose up -d mlflow` for model-quality tests; the rest run with zero infra
make load     # needs the full stack up; checks QA1 (p95<=150ms, p99<=300ms)
```

## Running without Docker

```bash
brew install redis && redis-server &
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlruns --serve-artifacts &
make train
make run-local
```
