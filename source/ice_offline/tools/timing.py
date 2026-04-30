
import time
from collections.abc import Callable
from typing import Any

class Timer:
    """Minimal in-memory timing recorder keyed by string."""

    _records_ms: dict[str, float] = {}
    _stopwatch_ns: dict[str, int] = {}

    @classmethod
    def get(cls, key: str) -> float:
        return cls._records_ms.get(key, 0.0)

    @classmethod
    def set(cls, key: str, value_ms: float) -> None:
        cls._records_ms[key] = float(value_ms)

    @classmethod
    def record(cls, key: str, callback: Callable[[], Any]) -> tuple[float, Any]:
        t0 = time.perf_counter_ns()
        value = callback()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        cls._records_ms[key] = elapsed_ms
        return elapsed_ms, value

    @classmethod
    def stopwatch(cls, key: str) -> float:
        now_ns = time.perf_counter_ns()
        prev_ns = cls._stopwatch_ns.get(key)
        if prev_ns is None:
            cls._stopwatch_ns[key] = now_ns
            cls._records_ms[key] = 0.0
            return 0.0
        elapsed_ms = (now_ns - prev_ns) / 1_000_000.0
        cls._stopwatch_ns[key] = now_ns
        cls._records_ms[key] = elapsed_ms
        return elapsed_ms
