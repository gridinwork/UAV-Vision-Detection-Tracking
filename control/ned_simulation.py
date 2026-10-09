"""Pixel offset to a virtual North/East command.

This repeats the geometry in offboard/track_and_follow.py:

    E = (target_x - frame_centre_x) * scale
    N = (frame_centre_y - target_y) * scale

`scale` is a simulation coefficient (default 0.1). It is not a calibrated
metres-per-pixel value for an arbitrary video.
"""

from __future__ import annotations


def virtual_ned(
    target_x: float,
    target_y: float,
    center_x: float,
    center_y: float,
    scale: float,
    inside_dead_zone: bool,
) -> tuple[float, float]:
    if inside_dead_zone:
        return 0.0, 0.0
    east = (target_x - center_x) * scale
    north = (center_y - target_y) * scale
    return north, east
