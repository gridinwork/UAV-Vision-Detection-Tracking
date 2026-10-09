"""Directional virtual commands from the filtered target offset."""

from __future__ import annotations

import math


def offsets(
    target_x: float,
    target_y: float,
    center_x: float,
    center_y: float,
    frame_w: int,
    frame_h: int,
) -> dict:
    dx = target_x - center_x
    dy = target_y - center_y
    half_w = frame_w / 2.0 or 1.0
    half_h = frame_h / 2.0 or 1.0
    magnitude = math.hypot(dx, dy)
    norm_span = math.hypot(half_w, half_h) or 1.0
    return {
        "dx": dx,
        "dy": dy,
        "x_norm": dx / half_w,
        "y_norm": dy / half_h,
        "magnitude": magnitude,
        "centering_error": magnitude / norm_span * 100.0,
    }


def directional_command(dx: float, dy: float, half_w: float, half_h: float, active: bool) -> str:
    if not active:
        return "HOLD"
    horizontal = None
    vertical = None
    if abs(dx) > half_w:
        horizontal = "RIGHT" if dx > 0 else "LEFT"
    if abs(dy) > half_h:
        vertical = "BACK" if dy > 0 else "FORWARD"
    if vertical and horizontal:
        return f"{vertical} + {horizontal}"
    if vertical:
        return vertical
    if horizontal:
        return horizontal
    return "HOLD"


def control_state(status: str, command: str) -> str:
    if status == "LOST":
        return "LOST"
    if status == "SEARCHING":
        return "SEARCHING"
    if command == "HOLD":
        return "HOLDING"
    return "CORRECTING"
