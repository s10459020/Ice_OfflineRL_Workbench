from __future__ import annotations

import time


def now_ns() -> int:
    """Return a monotonic high-resolution timestamp in nanoseconds."""
    return time.perf_counter_ns()


def now_s() -> float:
    """Return a monotonic high-resolution timestamp in seconds."""
    return time.perf_counter()


def ns_to_ms(duration_ns: int) -> float:
    """Convert nanoseconds to milliseconds."""
    return duration_ns / 1_000_000.0


class Stopwatch:
    """Simple start/elapsed helper based on perf_counter_ns."""

    def __init__(self) -> None:
        self._start_ns: int | None = None

    def start(self) -> None:
        self._start_ns = now_ns()

    def elapsed_ns(self) -> int:
        if self._start_ns is None:
            raise RuntimeError("stopwatch has not been started")
        return now_ns() - self._start_ns

    def elapsed_ms(self) -> float:
        return ns_to_ms(self.elapsed_ns())
