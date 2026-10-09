"""Frame result consumed by the controller, overlay, and panels."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TargetState:
    frame_id: int = 0
    frame_w: int = 0
    frame_h: int = 0
    detections: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    target_id: int | None = None
    target_label: str = ""
    confidence: float | None = None
    raw_xy: tuple[float, float] | None = None
    filtered_xy: tuple[float, float] | None = None
    box: tuple[int, int, int, int] | None = None
    status: str = "SEARCHING"
    lost: bool = False
    trail: list = field(default_factory=list)
    raw_detection_count: int = 0
    area_ratio: float = 0.0
    scale_trend: str = "STABLE"
    dead_zone_half: tuple[float, float] = (0.0, 0.0)
    inside_dead_zone: bool = False
