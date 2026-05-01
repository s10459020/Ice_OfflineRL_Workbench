import time
from collections import deque
from collections.abc import Callable
from typing import Any


class _Timer:
    def __init__(self, buffer_size: int = 100) -> None:
        self._records_ms: dict[str, float] = {}
        self._buffers_ms: dict[str, deque[float]] = {}
        self._stopwatch_ns: dict[str, int] = {}
        self._buffer_size = int(buffer_size)

    def __call__(self, buffer_size: int = 100) -> "_Timer":
        return _Timer(buffer_size=buffer_size)

    def get(self, key: str) -> float:
        return self._records_ms.get(key, 0.0)

    def set(self, key: str, value_ms: float) -> None:
        self._records_ms[key] = float(value_ms)
        self._buffers_ms.pop(key, None)

    def _record_value(self, key: str, value_ms: float) -> float:
        size = self._buffer_size
        value = float(value_ms)
        if size == 0:
            self._records_ms[key] = value
            self._buffers_ms.pop(key, None)
            return value

        buffer = self._buffers_ms.setdefault(key, deque(maxlen=size))
        buffer.append(value)
        avg_ms = sum(buffer) / float(len(buffer))
        self._records_ms[key] = avg_ms
        return avg_ms

    def record(self, key: str, callback: Callable[[], Any]) -> tuple[float, Any]:
        t0 = time.perf_counter_ns()
        value = callback()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        avg_ms = self._record_value(key=key, value_ms=elapsed_ms)
        return avg_ms, value

    def stopwatch(self, key: str) -> float:
        now_ns = time.perf_counter_ns()
        prev_ns = self._stopwatch_ns.get(key)
        if prev_ns is None:
            self._stopwatch_ns[key] = now_ns
            self._records_ms[key] = 0.0
            return 0.0
        elapsed_ms = (now_ns - prev_ns) / 1_000_000.0
        self._stopwatch_ns[key] = now_ns
        return self._record_value(key=key, value_ms=elapsed_ms)


Timer = _Timer()
