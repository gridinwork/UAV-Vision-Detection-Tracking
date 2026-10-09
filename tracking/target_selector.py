"""Switches between the original centre tracker and the enhanced tracker."""

from __future__ import annotations

from tracking.enhanced_tracker import EnhancedTracker
from tracking.moving_average import SMOOTHING_WINDOWS
from tracking.original_tracker import OriginalTracker
from tracking.target_state import TargetState


class TargetSelector:
    def __init__(self) -> None:
        self.original = OriginalTracker(20)
        self.enhanced = EnhancedTracker(20)
        self.algorithm = "ENHANCED"

    def reset(self) -> None:
        self.original.reset()
        self.enhanced.reset()

    def clear_target(self) -> None:
        self.enhanced.clear_target()

    def cycle(self, step: int) -> None:
        self.enhanced.cycle(step)

    def update(
        self,
        detections: list,
        frame_w: int,
        frame_h: int,
        frame_id: int,
        dead_zone: float,
        click_xy: tuple[float, float] | None,
        mode: str,
        algorithm: str,
        smoothing: str,
        lost_timeout: float,
        auto_reacquire: bool,
        trail_length: int,
        now: float | None = None,
    ) -> TargetState:
        if algorithm != self.algorithm:
            self.reset()
            self.algorithm = algorithm
        half = (frame_w * dead_zone * 0.5, frame_h * dead_zone * 0.5)
        if algorithm == "ORIGINAL":
            self.original.set_window(SMOOTHING_WINDOWS["ORIGINAL_20"])
            return self.original.update(detections, frame_w, frame_h, frame_id, half)
        window = SMOOTHING_WINDOWS.get(smoothing, 20)
        self.enhanced.set_window(window)
        return self.enhanced.update(
            detections,
            frame_w,
            frame_h,
            frame_id,
            half,
            click_xy,
            mode,
            lost_timeout,
            auto_reacquire,
            trail_length,
            now=now,
        )
