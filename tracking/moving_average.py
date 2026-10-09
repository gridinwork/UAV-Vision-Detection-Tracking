"""FIFO moving-average filter.

ORIGINAL 20 matches offboard/track_and_follow.py: a 20-point buffer, oldest
sample dropped, result rounded to an integer pixel.
"""

from __future__ import annotations


SMOOTHING_WINDOWS = {
    "OFF": 1,
    "LOW": 5,
    "MEDIUM": 10,
    "HIGH": 30,
    "ORIGINAL_20": 20,
}


class MovingAverageFilter:
    def __init__(self, window: int = 20) -> None:
        self.window = max(1, int(window))
        self._xs: list[float] = []
        self._ys: list[float] = []

    def set_window(self, window: int) -> None:
        window = max(1, int(window))
        if window != self.window:
            self.window = window
            self._xs = self._xs[-window:]
            self._ys = self._ys[-window:]

    def reset(self) -> None:
        self._xs.clear()
        self._ys.clear()

    def push(self, x: float, y: float) -> tuple[int, int]:
        self._xs.append(float(x))
        self._ys.append(float(y))
        if len(self._xs) > self.window:
            self._xs = self._xs[-self.window :]
            self._ys = self._ys[-self.window :]
        n = len(self._xs)
        return int(round(sum(self._xs) / n)), int(round(sum(self._ys) / n))

    @property
    def length(self) -> int:
        return len(self._xs)
