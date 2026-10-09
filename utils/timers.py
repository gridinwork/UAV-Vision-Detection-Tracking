"""Small wall-clock helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager


class LapTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def lap_ms(self) -> float:
        now = time.perf_counter()
        ms = (now - self._t0) * 1000.0
        self._t0 = now
        return ms


@contextmanager
def measure_ms():
    start = time.perf_counter()
    bucket = {"ms": 0.0}
    try:
        yield bucket
    finally:
        bucket["ms"] = (time.perf_counter() - start) * 1000.0


class FpsMeter:
    def __init__(self) -> None:
        self._count = 0
        self._t0 = time.perf_counter()
        self.value = 0.0

    def tick(self) -> float:
        self._count += 1
        now = time.perf_counter()
        elapsed = now - self._t0
        if self._count >= 2 and elapsed >= 0.2:
            self.value = self._count / elapsed
            if elapsed >= 0.5:
                self._count = 0
                self._t0 = now
        return self.value
