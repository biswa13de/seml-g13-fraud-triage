"""Load test for QA1 (PLAN.md Sec 4): triage-api must hold p95 <= 150ms and
p99 <= 300ms under concurrent load. Hits the live running triage-api, so
`docker compose up` (or `make run-local`) must be running first.

Usage: python scripts/load_test.py [--n 5000] [--concurrency 50]
"""
import argparse
import asyncio
import random
import time

import httpx

URL = "http://localhost:8000/v1/payments/triage"
API_KEY = "demo-key-g13"


def random_payment(i: int) -> dict:
    drains = random.random() < 0.3
    old_bal = random.uniform(100, 50000)
    amount = old_bal if drains else random.uniform(10, old_bal * 0.5)
    return {
        "txn_id": f"LOAD-{i}", "step": random.randint(1, 743),
        "type": random.choice(["TRANSFER", "CASH_OUT"]),
        "amount": round(amount, 2),
        "nameOrig": f"C-LOAD-{i % 5000}", "oldbalanceOrg": round(old_bal, 2),
        "nameDest": f"C-LOAD-DEST-{random.randint(0, 2000)}",
        "oldbalanceDest": round(random.uniform(0, 20000), 2),
    }


async def fire(client: httpx.AsyncClient, i: int, latencies: list, sem: asyncio.Semaphore) -> None:
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(URL, json=random_payment(i), headers={"X-API-Key": API_KEY}, timeout=5.0)
            resp.raise_for_status()
        except Exception as exc:
            print(f"request {i} failed: {exc}")
            return
        latencies.append((time.perf_counter() - t0) * 1000)


async def main(n: int, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[fire(client, i, latencies, sem) for i in range(n)])
        total_s = time.perf_counter() - t0

    latencies.sort()
    def pct(p):
        return latencies[min(int(p * len(latencies)), len(latencies) - 1)]

    print(f"\n=== Load test: {len(latencies)}/{n} succeeded, concurrency={concurrency} ===")
    print(f"Throughput: {len(latencies) / total_s:.1f} req/s over {total_s:.1f}s")
    print(f"p50={pct(0.50):.1f}ms  p95={pct(0.95):.1f}ms  p99={pct(0.99):.1f}ms  max={latencies[-1]:.1f}ms")
    print(f"QA1 budget: p95<=150ms {'PASS' if pct(0.95) <= 150 else 'FAIL'}, "
          f"p99<=300ms {'PASS' if pct(0.99) <= 300 else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main(args.n, args.concurrency))
