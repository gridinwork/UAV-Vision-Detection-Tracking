"""Car boxes, ids, and confidence."""

from __future__ import annotations

import cv2


def draw_detections(frame, state, overlays: dict) -> None:
    if not overlays.get("boxes", True):
        return
    show_conf = overlays.get("confidence", True)
    show_ids = overlays.get("ids", True)
    occupied: list[tuple[int, int, int, int]] = []
    ordered = sorted(state.tracks, key=lambda track: bool(track.get("selected")))
    for track in ordered:
        if track.get("missed", 0):
            continue
        x, y, w, h = track["box"]
        selected = bool(track.get("selected"))
        if selected:
            color = (40, 190, 255)
            thickness = 3
        else:
            color = (255, 210, 70)
            thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if selected:
            cv2.rectangle(frame, (x - 1, y - 1), (x + w + 1, y + h + 1), (20, 140, 255), 1)
        parts = []
        if show_ids and track.get("id"):
            parts.append(f"CAR #{track['id']}")
        else:
            parts.append("CAR")
        if show_conf:
            parts.append(f"{int(round(float(track['confidence']) * 100))}%")
        _draw_label(frame, " ".join(parts), x, y, w, h, color, occupied)


def _draw_label(frame, text, x, y, w, h, color, occupied) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    box_w, box_h = tw + 8, th + base + 6
    height, width = frame.shape[:2]
    candidates = [
        (x, y - box_h - 2),
        (x, y + h + 2),
        (x + w + 2, y),
        (x - box_w - 2, y),
    ]
    left, top = candidates[0]
    for cx, cy in candidates:
        left = min(max(0, cx), max(0, width - box_w))
        top = min(max(0, cy), max(0, height - box_h))
        rect = (left, top, left + box_w, top + box_h)
        if not any(_intersects(rect, other) for other in occupied):
            break
    occupied.append((left, top, left + box_w, top + box_h))
    cv2.rectangle(frame, (left, top), (left + box_w, top + box_h), (12, 16, 22), -1)
    cv2.rectangle(frame, (left, top), (left + box_w, top + box_h), color, 1)
    cv2.putText(frame, text, (left + 4, top + th + 2), font, scale, color, thickness, cv2.LINE_AA)


def _intersects(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
