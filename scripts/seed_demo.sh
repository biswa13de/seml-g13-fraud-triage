#!/usr/bin/env bash
# Seeds the running stack with a clean, small set of scenario payments for
# screenshots (PLAN.md Sec 7.4 / docs/SCREENSHOT_GUIDE.md). Safe to re-run:
# each txn_id is unique so re-running just adds more of the same scenarios.
set -euo pipefail

API="http://localhost:8000/v1/payments/triage"
KEY="demo-key-g13"

post() {
  curl -s -X POST "$API" -H "Content-Type: application/json" -H "X-API-Key: $KEY" -d "$1"
  echo
}

echo "--- genuine payment -> ALLOW ---"
post '{"txn_id":"T-GENUINE-1","step":120,"type":"TRANSFER","amount":500.0,"nameOrig":"C-PAYER-1","oldbalanceOrg":5000.0,"nameDest":"C-PAYEE-1","oldbalanceDest":3000.0}'

echo "--- large night cash-out -> STEP_UP ---"
post '{"txn_id":"T-STEPUP-1","step":150,"type":"CASH_OUT","amount":15000.0,"nameOrig":"C-PAYER-4","oldbalanceOrg":20000.0,"nameDest":"C-PAYEE-4","oldbalanceDest":200.0}'

echo "--- account takeover (drains balance) -> BLOCK ---"
post '{"txn_id":"T-FRAUD-1","step":50,"type":"TRANSFER","amount":9500.0,"nameOrig":"C-PAYER-2","oldbalanceOrg":9500.0,"nameDest":"C-NEW-PAYEE","oldbalanceDest":0.0}'

echo "--- mule fan-in cash-out -> BLOCK ---"
post '{"txn_id":"T-MULE-1","step":200,"type":"CASH_OUT","amount":45000.0,"nameOrig":"C-PAYER-3","oldbalanceOrg":45000.0,"nameDest":"C-MULE-1","oldbalanceDest":0.0}'

echo "--- invalid payload -> 422 ---"
post '{"txn_id":"T-BAD-1","step":50,"type":"TRANSFER","amount":-500,"nameOrig":"C1","oldbalanceOrg":1000,"nameDest":"C2","oldbalanceDest":0}'

echo
echo "Seeded. Check http://localhost:8501 (console), http://localhost:8002/cases, http://localhost:8003/metrics"
