"""Event-Driven Architecture helpers: publish-subscribe over Redis Streams.

- publish(): XADD a decision event.
- consume_forever(): a consumer-group loop with at-least-once delivery,
  idempotent handlers (callers dedupe on txn_id), and pending-message replay
  after a crash (XAUTOCLAIM), so a restarted consumer picks up unacked work.
"""
import json
import time
from collections.abc import Callable

from common.logger import get_logger

logger = get_logger(__name__)


def publish(client, stream: str, event: dict) -> str:
    fields = {k: json.dumps(v, default=str) for k, v in event.items()}
    return client.xadd(stream, fields)


def ensure_group(client, stream: str, group: str) -> None:
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as exc:  # BUSYGROUP = already exists, which is fine
        if "BUSYGROUP" not in str(exc):
            raise


def consume_forever(
    client, stream: str, group: str, consumer: str,
    handler: Callable[[str, dict], None], block_ms: int = 5000,
) -> None:
    """NOTE: `client` must be constructed with a socket_timeout comfortably
    LARGER than block_ms (see common.events.redis_client_for_consumer).
    XREADGROUP with block=N tells Redis to hold the connection open for up to
    N ms server-side; if redis-py's own socket_timeout is <= N, the CLIENT
    gives up and raises redis.exceptions.TimeoutError before Redis ever
    replies, which looks like the consumer "died" with no message read."""
    ensure_group(client, stream, group)
    logger.info("Consumer started", extra={"stream": stream, "group": group, "consumer": consumer})
    while True:
        # Replay anything claimed by a dead consumer before reading new messages
        try:
            _, claimed, _ = client.xautoclaim(stream, group, consumer, min_idle_time=30_000, start_id="0-0")
            for msg_id, fields in claimed:
                _handle(handler, stream, group, client, msg_id, fields)
        except Exception as exc:
            logger.warning("Autoclaim failed", extra={"error": str(exc)})

        try:
            resp = client.xreadgroup(group, consumer, {stream: ">"}, count=10, block=block_ms)
        except Exception as exc:
            logger.warning("xreadgroup failed, retrying", extra={"error": str(exc)})
            continue
        for _, messages in resp or []:
            for msg_id, fields in messages:
                _handle(handler, stream, group, client, msg_id, fields)


def redis_client_for_consumer(redis_url: str, block_ms: int = 5000):
    """Build a redis-py client whose socket_timeout exceeds block_ms, so a
    blocking XREADGROUP call doesn't trip the client-side read timeout
    before Redis's own server-side block window elapses."""
    import redis as _redis
    return _redis.from_url(redis_url, socket_timeout=(block_ms / 1000) + 10, socket_keepalive=True)


def _handle(handler, stream, group, client, msg_id, fields) -> None:
    event = {k.decode() if isinstance(k, bytes) else k:
             json.loads(v) for k, v in fields.items()}
    try:
        handler(msg_id, event)
        client.xack(stream, group, msg_id)
    except Exception as exc:
        logger.error("Handler failed, will retry", extra={"msg_id": str(msg_id), "error": str(exc)})
        time.sleep(0.1)
