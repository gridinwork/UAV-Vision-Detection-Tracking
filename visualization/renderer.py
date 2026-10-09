"""Compose every overlay onto a copy of the camera frame."""

from __future__ import annotations

import cv2

from visualization.boxes import draw_detections
from visualization.flight_director import draw_flight_director
from visualization.overlays import command_lines, debug_lines, draw_fps, draw_rec_badge, draw_text_block
from visualization.target_vector import draw_crosshair, draw_dead_zone, draw_trail, draw_vector


def render_frame(frame, state, packet, config, backend_label: str, recording: bool = False, rec_label: str = ""):
    view = frame.copy()
    overlays = config.overlays or {}
    if overlays.get("dead_zone", True):
        draw_dead_zone(view, state.dead_zone_half)
    if overlays.get("crosshair", True):
        draw_crosshair(view)
    if overlays.get("trail", True):
        draw_trail(view, state.trail)
    draw_detections(view, state, overlays)
    if overlays.get("target_vector", True):
        draw_vector(view, state)
    if overlays.get("flight_director", True):
        draw_flight_director(view, packet["command"])
    if overlays.get("command_text", True):
        show_ned = config.control_display == "NED"
        lines = command_lines(state, packet, "VIRTUAL", show_ned)
        draw_text_block(view, lines, (12, view.shape[0] - 18 * (len(lines) + 1)))
    if overlays.get("fps", True):
        fps = config.proc_fps or config.source_fps
        draw_fps(view, f"FPS {fps:.1f}  {backend_label}")
    if recording and rec_label:
        draw_rec_badge(view, rec_label)
    _ = cv2
    return view


def render_debug(view, info: dict) -> None:
    draw_text_block(
        view,
        debug_lines(info),
        (max(8, view.shape[1] - 250), 42),
        accent=(180, 220, 255),
    )
