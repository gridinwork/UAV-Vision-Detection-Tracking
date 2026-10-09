"""Virtual pan/tilt used until a real stabilized camera is connected.

This adapter only updates an internal angle. It does not open a serial port,
does not speak MAVLink, and does not address an aircraft or an effector.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PtzCommand:
    pan_deg: float
    tilt_deg: float
    rate_pan_dps: float
    rate_tilt_dps: float
    stopped: bool
    backend: str = "virtual"


class VirtualPtz:
    name = "virtual"

    def __init__(self) -> None:
        self.pan = 0.0
        self.tilt = 0.0
        self.pan_min = -90.0
        self.pan_max = 90.0
        self.tilt_min = -30.0
        self.tilt_max = 45.0
        self.pan_offset = 0.0
        self.tilt_offset = 0.0
        self.deg_per_norm = 35.0
        self.max_rate_dps = 40.0
        self._stopped = False
        self._rate_pan = 0.0
        self._rate_tilt = 0.0
        self.last = self.snapshot()

    def set_limits(self, pan_min: float, pan_max: float, tilt_min: float, tilt_max: float) -> None:
        self.pan_min = min(pan_min, pan_max)
        self.pan_max = max(pan_min, pan_max)
        self.tilt_min = min(tilt_min, tilt_max)
        self.tilt_max = max(tilt_min, tilt_max)
        self.pan = self._clamp(self.pan, self.pan_min, self.pan_max)
        self.tilt = self._clamp(self.tilt, self.tilt_min, self.tilt_max)
        self.last = self.snapshot()

    def calibrate(self, pan_offset: float, tilt_offset: float, deg_per_norm: float) -> None:
        self.pan_offset = float(pan_offset)
        self.tilt_offset = float(tilt_offset)
        self.deg_per_norm = max(1.0, float(deg_per_norm))
        self.last = self.snapshot()

    def home(self) -> PtzCommand:
        self._stopped = False
        self._rate_pan = 0.0
        self._rate_tilt = 0.0
        self.pan = self._clamp(self.pan_offset, self.pan_min, self.pan_max)
        self.tilt = self._clamp(self.tilt_offset, self.tilt_min, self.tilt_max)
        self.last = self.snapshot()
        return self.last

    def stop(self) -> PtzCommand:
        """Safe stop: zero the commanded rate and hold the current angle."""
        self._stopped = True
        self._rate_pan = 0.0
        self._rate_tilt = 0.0
        self.last = self.snapshot()
        return self.last

    def release_stop(self) -> None:
        self._stopped = False

    def slew_toward(self, norm_x: float, norm_y: float, dt: float) -> PtzCommand:
        """Move toward a normalized image error. Positive Y looks up.

        ``norm_x`` and ``norm_y`` are in approximately [-1, 1]. While stopped,
        the angle does not change.
        """
        if self._stopped:
            self._rate_pan = 0.0
            self._rate_tilt = 0.0
            self.last = self.snapshot()
            return self.last
        target_pan = self._clamp(self.pan_offset + float(norm_x) * self.deg_per_norm, self.pan_min, self.pan_max)
        target_tilt = self._clamp(self.tilt_offset + float(norm_y) * self.deg_per_norm, self.tilt_min, self.tilt_max)
        step = max(0.0, self.max_rate_dps * max(0.0, float(dt)))
        self.pan, self._rate_pan = self._approach(self.pan, target_pan, step, dt)
        self.tilt, self._rate_tilt = self._approach(self.tilt, target_tilt, step, dt)
        self.last = self.snapshot()
        return self.last

    def snapshot(self) -> PtzCommand:
        return PtzCommand(self.pan, self.tilt, self._rate_pan, self._rate_tilt, self._stopped, self.name)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _approach(current: float, target: float, step: float, dt: float) -> tuple[float, float]:
        delta = target - current
        if abs(delta) <= step or step == 0.0:
            moved = delta
            current = target
        else:
            moved = step if delta > 0 else -step
            current += moved
        rate = moved / dt if dt > 1e-6 else 0.0
        return current, rate
