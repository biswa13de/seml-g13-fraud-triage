"""Online feature store (Feature Store pattern).

Holds per-receiver velocity state so the scoring service can compute
behavioural features in O(log n) at request time. Two interchangeable
backends share one interface:

* InMemoryVelocityStore: used for offline replay and unit tests
* RedisVelocityStore: used by the running services

Read-before-write semantics: `get` for a payment is called before that
payment is `record`ed, so a payment never sees itself (point-in-time correct).
"""
from collections import defaultdict, deque
from typing import Protocol

from common.features import VELOCITY_WINDOW_STEPS, Velocity


class VelocityStore(Protocol):
    def get(self, dest: str, step: int) -> Velocity: ...

    def record(self, txn_id: str, dest: str, step: int, amount: float) -> None: ...


def _window_start(step: int) -> int:
    return step - VELOCITY_WINDOW_STEPS + 1


class InMemoryVelocityStore:
    def __init__(self) -> None:
        self._events: dict[str, deque[tuple[int, float]]] = defaultdict(deque)

    def _evict(self, dest: str, step: int) -> deque[tuple[int, float]]:
        events = self._events[dest]
        while events and events[0][0] < _window_start(step):
            events.popleft()
        return events

    def get(self, dest: str, step: int) -> Velocity:
        events = self._evict(dest, step)
        return Velocity(float(len(events)), float(sum(a for _, a in events)))

    def record(self, txn_id: str, dest: str, step: int, amount: float) -> None:
        self._evict(dest, step).append((step, amount))


class RedisVelocityStore:
    """One sorted set per receiver: member = "<txn_id>|<amount>", score = step.

    Using txn_id in the member makes `record` idempotent, which the
    at-least-once event consumers rely on.
    """

    KEY = "fs:dest:{dest}"
    TTL_SECONDS = 7 * 24 * 3600

    def __init__(self, client) -> None:
        self._r = client

    def get(self, dest: str, step: int) -> Velocity:
        members = self._r.zrangebyscore(self.KEY.format(dest=dest), _window_start(step), step)
        amounts = [float(m.decode().rsplit("|", 1)[1]) for m in members]
        return Velocity(float(len(amounts)), float(sum(amounts)))

    def record(self, txn_id: str, dest: str, step: int, amount: float) -> None:
        key = self.KEY.format(dest=dest)
        pipe = self._r.pipeline()
        pipe.zadd(key, {f"{txn_id}|{amount}": step})
        pipe.zremrangebyscore(key, "-inf", _window_start(step) - 1)
        pipe.expire(key, self.TTL_SECONDS)
        pipe.execute()
