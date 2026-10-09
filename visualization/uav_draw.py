"""UAV overlay: class, track id, confidence, status colour, trail and speed."""

from __future__ import annotations

import os

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - Pillow ships with torchvision
    Image = ImageDraw = ImageFont = None


_FONTS: dict[int, object] = {}


STATUS_COLOR = {
    "DETECTED": (0, 220, 255),
    "TRACKING": (80, 210, 90),
    "OCCLUDED": (0, 150, 255),
    "LOST": (70, 70, 235),
    "REACQUIRED": (230, 190, 40),
}


def render_uav(frame, tracks: list[dict], info: dict, overlays: dict):
    view = frame.copy()
    if overlays.get("crosshair", False):
        h, w = view.shape[:2]
        cv2.drawMarker(view, (w // 2, h // 2), (180, 190, 200), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
    for track in tracks:
        if overlays.get("trail", True):
            _trail(view, track.get("trail") or [], STATUS_COLOR.get(track.get("status"), (180, 180, 180)))
        if overlays.get("boxes", True):
            _box(view, track, overlays, str(info.get("unit_px") or "px/s"), str(info.get("unit_kmh") or "km/h"))
    if info.get("demo"):
        _banner(view, str(info.get("banner") or "DEMO / SIMULATED DATA"))
    elif info.get("warning_text"):
        _banner(view, info["warning_text"], (40, 90, 160))
    unit_px = str(info.get("unit_px") or "px/s")
    unit_kmh = str(info.get("unit_kmh") or "km/h")
    lines = []
    if overlays.get("fps", True):
        lines.append(f"FPS {float(info.get('proc_fps') or 0):.1f}  {info.get('backend') or ''}")
    target = info.get("target_label") or ""
    if target:
        speed = info.get("speed_px_s")
        speed_text = f"{float(speed):.1f} {unit_px}" if speed is not None else ""
        kmh = info.get("speed_kmh")
        if kmh is not None:
            speed_text = f"{speed_text}  {float(kmh):.1f} {unit_kmh}".strip()
        elif speed is not None and info.get("kmh_na"):
            speed_text = f"{speed_text}  {info.get('kmh_na')}".strip()
        status = info.get("status_label") or info.get("status") or ""
        lines.append(f"{target}  {status}  {speed_text}".strip())
    if info.get("ptz_text"):
        lines.append(str(info["ptz_text"]))
    elif info.get("ptz"):
        ptz = info["ptz"]
        lines.append(
            f"PTZ virtual  pan {ptz.get('pan', 0):+.1f}  tilt {ptz.get('tilt', 0):+.1f}"
            + ("  STOP" if ptz.get("stopped") else "")
        )
    if lines:
        _text_block(view, lines, (12, 28))
    return view


def _box(view, track: dict, overlays: dict, unit_px: str = "px/s", unit_kmh: str = "km/h") -> None:
    x, y, w, h = [int(v) for v in track["box"]]
    status = str(track.get("status") or "DETECTED")
    color = STATUS_COLOR.get(status, (200, 200, 200))
    selected = bool(track.get("selected"))
    if status in ("OCCLUDED", "LOST"):
        _dashed(view, x, y, w, h, color)
    else:
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 3 if selected else 2, cv2.LINE_AA)
    if selected:
        cv2.rectangle(view, (x - 2, y - 2), (x + w + 2, y + h + 2), (245, 245, 245), 1, cv2.LINE_AA)
    parts = []
    if overlays.get("ids", True):
        parts.append(f"#{track.get('id')}")
    parts.append(str(track.get("class_label") or track.get("class_name") or ""))
    if overlays.get("confidence", True):
        parts.append(f"{float(track.get('confidence') or 0):.2f}")
    if overlays.get("status", True):
        parts.append(str(track.get("status_label") or status))
    if overlays.get("speed", True):
        parts.append(f"{float(track.get('speed_px_s') or 0):.0f} {unit_px}")
        if track.get("speed_kmh") is not None:
            parts.append(f"{float(track['speed_kmh']):.1f} {unit_kmh}")
    _label(view, " ".join(part for part in parts if part), x, max(0, y - 8), color)


def _trail(view, points, color) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points, points[1:]):
        cv2.line(view, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 2, cv2.LINE_AA)


def _dashed(view, x, y, w, h, color) -> None:
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for start, end in zip(points, points[1:] + points[:1]):
        _dash_line(view, start, end, color)


def _dash_line(view, start, end, color) -> None:
    x1, y1 = start
    x2, y2 = end
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    gap = 8
    for step in range(0, steps, gap * 2):
        a = step / steps
        b = min(steps, step + gap) / steps
        cv2.line(
            view,
            (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)),
            (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)),
            color,
            2,
            cv2.LINE_AA,
        )


def _font(size: int):
    if ImageFont is None:
        return None
    size = max(12, int(size))
    if size not in _FONTS:
        path = next(
            (item for item in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf") if os.path.isfile(item)),
            "",
        )
        _FONTS[size] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return _FONTS[size]


def _measure(text: str, size: int) -> tuple[int, int]:
    font = _font(size)
    if font is None:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        return tw, th + base
    box = font.getbbox(text)
    return max(1, box[2] - box[0]), max(1, box[3] - box[1])


def _paint(view, text: str, x: int, y: int, color_bgr, size: int) -> None:
    font = _font(size)
    if font is None or Image is None:
        cv2.putText(view, text, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
        return
    height, width = view.shape[:2]
    tw, th = _measure(text, size)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + tw + 2), min(height, y + th + 4)
    if x1 <= x0 or y1 <= y0:
        return
    crop = view[y0:y1, x0:x1]
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text((x - x0, y - y0), text, font=font, fill=(int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])))
    view[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _label(view, text, x, y, color) -> None:
    tw, th = _measure(text, 16)
    top = max(0, y - th - 8)
    cv2.rectangle(view, (x, top), (min(view.shape[1] - 1, x + tw + 8), top + th + 8), (12, 16, 22), -1)
    _paint(view, text, x + 4, top + 2, color, 16)


def _banner(view, text: str, color=(40, 40, 180)) -> None:
    tw, th = _measure(text, 22)
    height, width = view.shape[:2]
    x = max(8, (width - tw) // 2)
    y = 8
    cv2.rectangle(view, (x - 8, y), (min(width - 1, x + tw + 12), min(height - 1, y + th + 10)), color, -1)
    _paint(view, text, x, y + 2, (245, 245, 245), 22)


def _text_block(view, lines, origin) -> None:
    x, y = origin
    for line in lines:
        _paint(view, line, x + 1, y + 1, (10, 14, 18), 16)
        _paint(view, line, x, y, (236, 240, 244), 16)
        y += 22
