"""Compact flight-director rose. The active virtual command is highlighted."""

from __future__ import annotations

import cv2


def draw_flight_director(frame, command: str) -> None:
    height, width = frame.shape[:2]
    box_w, box_h = 156, 132
    x0 = max(4, width - box_w - 8)
    y0 = max(4, height - box_h - 8)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (10, 14, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (50, 70, 90), 1)
    cx = x0 + box_w // 2
    cy = y0 + 68
    active = _active_dirs(command)
    dim = (130, 140, 150)
    hot = (40, 210, 255)
    hold = (80, 220, 140)
    cv2.putText(frame, "FLIGHT DIRECTOR", (x0 + 18, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 200), 1, cv2.LINE_AA)
    _arm(frame, (cx, cy), (cx, cy - 28), "FORWARD", hot if "FORWARD" in active else dim, (cx - 32, cy - 36))
    _arm(frame, (cx, cy), (cx, cy + 28), "BACK", hot if "BACK" in active else dim, (cx - 18, cy + 44))
    _arm(frame, (cx, cy), (cx - 36, cy), "LEFT", hot if "LEFT" in active else dim, (x0 + 6, cy - 8))
    _arm(frame, (cx, cy), (cx + 36, cy), "RIGHT", hot if "RIGHT" in active else dim, (cx + 40, cy - 8))
    center = hold if command == "HOLD" else hot
    cv2.circle(frame, (cx, cy), 6, center, -1, cv2.LINE_AA)


def _active_dirs(command: str) -> set[str]:
    if not command or command == "HOLD":
        return set()
    return set(part.strip() for part in command.split("+"))


def _arm(frame, start, end, label, color, label_pos) -> None:
    cv2.arrowedLine(frame, start, end, color, 2, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
