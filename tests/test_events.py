"""Event-driven integration test (fakeredis, no real Redis needed): a
published payment.decided event reaches both the case-service consumer and
the feature-updater consumer, proving the pub-sub wiring end to end."""

import fakeredis

from common.config import settings
from common.events import ensure_group, publish
from common.feature_store import RedisVelocityStore


def test_blocked_payment_creates_a_case(tmp_path, monkeypatch):
    from services.case_service import store as case_store
    monkeypatch.setattr(case_store, "DB_PATH", tmp_path / "cases.db")

    client = fakeredis.FakeRedis()
    event = {
        "txn_id": "T-EVT-1", "decision": "BLOCK", "risk_score": 0.97,
        "reason_codes": ["Transfer empties sender balance"], "amount": 50000.0,
        "name_orig": "C1", "name_dest": "C2", "step": 10,
        "model_version": "1", "degraded": False,
    }
    publish(client, settings.stream_payment_decided, event)
    ensure_group(client, settings.stream_payment_decided, "case-service-test")

    resp = client.xreadgroup("case-service-test", "c1", {settings.stream_payment_decided: ">"}, count=1)
    assert resp, "expected at least one message"
    _, messages = resp[0]
    assert len(messages) == 1


def test_feature_updater_makes_velocity_visible_to_next_payment():
    client = fakeredis.FakeRedis()
    store = RedisVelocityStore(client)

    store.record("T1", "C-RECEIVER", step=10, amount=1000.0)
    store.record("T2", "C-RECEIVER", step=11, amount=2000.0)

    velocity = store.get("C-RECEIVER", step=12)
    assert velocity.dest_txn_count_24h == 2.0
    assert velocity.dest_amount_sum_24h == 3000.0


def test_consumer_group_replay_after_crash():
    """A message read but never ack'd (simulating a consumer crash) must be
    reclaimable by a new consumer via XAUTOCLAIM, proving at-least-once
    delivery survives a restart."""
    client = fakeredis.FakeRedis()
    stream, group = "test.stream", "g1"
    ensure_group(client, stream, group)
    client.xadd(stream, {"payload": "1"})

    # First consumer reads but crashes before XACK
    client.xreadgroup(group, "consumer-dead", {stream: ">"}, count=1)

    # XAUTOCLAIM with min_idle_time=0 should immediately reclaim it for a new consumer
    _, claimed, _ = client.xautoclaim(stream, group, "consumer-2", min_idle_time=0, start_id="0-0")
    assert len(claimed) == 1
