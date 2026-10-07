"""
services/feature_updater/main.py
----------------------------------------------------------------
feature-updater — ML-adjacent, non-ML component (no model inside).

Consumes payment.decided and writes the payer/payee velocity state into the
online feature store (Redis), so the NEXT payment from/to that account sees
up-to-date behavioural features (FR7). This runs asynchronously so updating
features never adds latency to the authorisation path (QA1).
----------------------------------------------------------------
"""
from common.config import settings
from common.events import consume_forever, redis_client_for_consumer
from common.feature_store import RedisVelocityStore
from common.logger import get_logger

logger = get_logger(__name__)

GROUP = "feature-updater"


def run() -> None:
    client = redis_client_for_consumer(settings.redis_url)
    store = RedisVelocityStore(client)

    def handle(msg_id, event: dict) -> None:
        # Idempotent: record() keys on txn_id, so replays of the same event
        # (at-least-once delivery) don't double-count velocity.
        store.record(event["txn_id"], event["name_dest"], event["step"], event["amount"])

    consume_forever(client, settings.stream_payment_decided, GROUP, "feature-consumer-1", handle)


if __name__ == "__main__":
    run()
