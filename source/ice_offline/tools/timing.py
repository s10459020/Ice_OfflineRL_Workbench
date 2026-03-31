from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

class Timer:
    """Minimal in-memory timing recorder keyed by string."""

    _records_ms: dict[str, float] = {}

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
