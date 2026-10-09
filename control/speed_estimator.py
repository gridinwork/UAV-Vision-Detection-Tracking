"""Virtual speed suggestion from how the target moves over time.

This is a heuristic for the demo. It is not a measured vehicle speed.
Bird's-eye reading used here:
- target drifting up the frame and/or shrinking  -> pulling away -> SPEED UP
- target growing quickly and drifting down       -> closing in  -> SLOW DOWN
- size and offset steady                         -> HOLD SPEED
"""

from __future__ import annotations


class SpeedEstimator:
    def __init__(self) -> None:
        self._samples: list[tuple[float, float, float]] = []

    def reset(self) -> None:
        self._samples.clear()

    def update(self, now: float, target_y: float | None, area_ratio: float | None, valid: bool) -> tuple[str, str]:
        if not valid or target_y is None or area_ratio is None:
            self._samples.clear()
            return "HOLD SPEED", "STABLE"
        self._samples.append((now, float(target_y), float(area_ratio)))
        self._samples = [sample for sample in self._samples if now - sample[0] <= 1.2]
        trend = self._trend()
        if len(self._samples) < 5:
            return "HOLD SPEED", trend
        t0, y0, a0 = self._samples[0]
        t1, y1, a1 = self._samples[-1]
        dt = max(1e-3, t1 - t0)
        y_slope = (y1 - y0) / dt  # px/s, positive = moving down the frame
        area_slope = (a1 - a0) / dt
        if abs(y_slope) < 12.0 and abs(area_slope) < 0.006:
            speed = "HOLD SPEED"
        elif area_slope > 0.012 and y_slope > 18.0:
            speed = "SLOW DOWN"
        elif y_slope < -14.0 or area_slope < -0.008:
            speed = "SPEED UP"
        elif area_slope > 0.02:
            speed = "SLOW DOWN"
        else:
            speed = "HOLD SPEED"
        return speed, trend

    def _trend(self) -> str:
        if len(self._samples) < 4:
            return "STABLE"
        areas = [sample[2] for sample in self._samples]
        delta = areas[-1] - areas[0]
        if delta > 0.004:
            return "APPROACHING"
        if delta < -0.004:
            return "RECEDING"
        return "STABLE"
