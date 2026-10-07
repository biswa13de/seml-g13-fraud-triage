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

Kaggle [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1), a public simulated mobile-money transaction log (6.36M rows). The dataset is not committed to git.

1. Kaggle → Settings → API → *Create New Token*. This downloads `kaggle.json`.
2. Move it into place:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
3. Download:
   ```bash
   pip install kaggle
   python data/download_paysim.py
   ```

## Running

Instructions will be added as the services are built (Docker Compose, or `make run-local` without Docker).
