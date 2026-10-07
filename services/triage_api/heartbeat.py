"""Heartbeat tactic (Session 3/architecture-tactics): triage-api polls
scoring-service's /health on a timer instead of probing it on every request
(which would double the latency of every payment). A small failure streak
threshold avoids flapping on a single slow health check."""
import threading

import httpx

from common.logger import get_logger

logger = get_logger(__name__)


class Heartbeat:
    def __init__(self, url: str, interval_s: float, failure_threshold: int) -> None:
        self._url = url
        self._interval_s = interval_s
        self._failure_threshold = failure_threshold
        self._consecutive_failures = 0
        self._healthy = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def _check_once(self) -> None:
        try:
            resp = httpx.get(self._url, timeout=1.0)
            resp.raise_for_status()
            self._consecutive_failures = 0
            self._healthy = True
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                if self._healthy:
                    logger.warning("scoring-service unhealthy, switching to rules-only",
                                    extra={"error": str(exc)})
                self._healthy = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._check_once()
            self._stop.wait(self._interval_s)

    def start(self) -> None:
        self._check_once()  # don't start degraded if scorer is already up
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
