"""Centre-offset tracker from the upstream project.

Each frame the car closest to the frame centre is the target. Its centre is
smoothed with the moving-average filter. When nothing is detected the target
is pinned to the frame centre so the virtual controller holds, and that pin
is not written into the filter buffer (same idea as track_and_follow.py).
"""

from __future__ import annotations

from tracking.moving_average import MovingAverageFilter
from tracking.target_state import TargetState


class OriginalTracker:
    def __init__(self, window: int = 20) -> None:
        self.filter = MovingAverageFilter(window)
        self._had_target = False

    def reset(self) -> None:
        self.filter.reset()
        self._had_target = False

    def set_window(self, window: int) -> None:
        self.filter.set_window(window)

    def update(
        self,
        detections: list,
        frame_w: int,
        frame_h: int,
        frame_id: int,
        dead_half: tuple[float, float],
    ) -> TargetState:
        state = TargetState(
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
            detections=list(detections),
            dead_zone_half=dead_half,
        )
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        if not detections:
            state.status = "SEARCHING"
            state.filtered_xy = (cx, cy)
            state.raw_xy = (cx, cy)
            state.inside_dead_zone = True
            state.target_label = ""
            return state

        def dist(det) -> float:
            dx, dy = det.center
            return (dx - cx) ** 2 + (dy - cy) ** 2

        target = min(detections, key=dist)
        raw_x, raw_y = target.center
        filt_x, filt_y = self.filter.push(raw_x, raw_y)
        self._had_target = True
        half_w, half_h = dead_half
        inside = abs(filt_x - cx) <= half_w and abs(filt_y - cy) <= half_h
        area = target.area / float(frame_w * frame_h) if frame_w and frame_h else 0.0
        state.target_id = 1
        state.target_label = "CAR"
        state.confidence = target.confidence
        state.raw_xy = (raw_x, raw_y)
        state.filtered_xy = (float(filt_x), float(filt_y))
        state.box = target.box
        state.inside_dead_zone = inside
        state.status = "CENTERED" if inside else "CORRECTING"
        state.area_ratio = area
        state.tracks = [
            {
                "id": index + 1,
                "box": det.box,
                "confidence": det.confidence,
                "class_name": det.class_name,
                "selected": det is target,
            }
            for index, det in enumerate(detections)
        ]
        return state
