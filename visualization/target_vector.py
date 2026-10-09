"""Centre crosshair, dead zone, target vector, and trail."""

from __future__ import annotations

import cv2


def draw_crosshair(frame) -> None:
    height, width = frame.shape[:2]
    cx, cy = int(width / 2), int(height / 2)
    color = (230, 230, 230)
    cv2.line(frame, (cx - 14, cy), (cx + 14, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 14), (cx, cy + 14), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (80, 220, 255), -1, cv2.LINE_AA)


def draw_dead_zone(frame, half) -> None:
    height, width = frame.shape[:2]
    cx, cy = int(width / 2), int(height / 2)
    half_w, half_h = int(half[0]), int(half[1])
    p1 = (cx - half_w, cy - half_h)
    p2 = (cx + half_w, cy + half_h)
    cv2.rectangle(frame, p1, p2, (70, 200, 120), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "DEAD ZONE",
        (p1[0] + 6, min(height - 8, p1[1] + 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (70, 200, 120),
        1,
        cv2.LINE_AA,
    )


def draw_vector(frame, state) -> None:
    if state.filtered_xy is None or state.status in ("LOST", "SEARCHING", "DETECTED"):
        return
    if state.target_id is None:
        return
    height, width = frame.shape[:2]
    cx, cy = int(width / 2), int(height / 2)
    tx, ty = int(state.filtered_xy[0]), int(state.filtered_xy[1])
    cv2.line(frame, (cx, cy), (tx, ty), (40, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 5, (40, 180, 255), -1, cv2.LINE_AA)


def draw_trail(frame, trail) -> None:
    if len(trail) < 2:
        return
    points = [(int(x), int(y)) for x, y in trail]
    for index in range(1, len(points)):
        shade = int(80 + 160 * index / len(points))
        cv2.line(frame, points[index - 1], points[index], (shade, 230, 140), 2, cv2.LINE_AA)
