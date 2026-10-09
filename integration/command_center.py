"""JSON events for Counter-UAS Sensor Fusion Command Center.

The public command center accepts one JSON object per UDP datagram and treats
``id`` as the track key. This adapter adds the vision fields required by the
technical specification. Screen coordinates are not converted to latitude,
longitude, range or speed in kilometres.
"""

from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


FORBIDDEN = {
    "lat",
    "lon",
    "latitude",
    "longitude",
    "range_m",
    "range",
    "distance_m",
    "gps",
}


def build_event(
    *,
    sensor_id: str,
    track_id: int,
    class_name: str,
    confidence: float,
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    source: str,
    status: str,
    speed_px_s: float = 0.0,
    when: datetime | None = None,
) -> dict:
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("Размер кадра должен быть положительным.")
    x, y, w, h = [float(value) for value in box]
    stamp = when or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    event = {
        "id": f"{sensor_id}-{int(track_id)}",
        "sensor_id": str(sensor_id),
        "track_id": int(track_id),
        "time_utc": stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "class": str(class_name or "unknown"),
        "confidence": round(max(0.0, min(1.0, float(confidence))) * 100.0, 2),
        "confidence_01": round(max(0.0, min(1.0, float(confidence))), 4),
        "cv": True,
        "bbox_norm": {
            "x": round(x / frame_w, 6),
            "y": round(y / frame_h, 6),
            "w": round(w / frame_w, 6),
            "h": round(h / frame_h, 6),
        },
        "source": str(source or ""),
        "status": str(status or ""),
        "speed_px_s": round(float(speed_px_s), 3),
    }
    validate_event(event)
    return event


def validate_event(event: dict) -> None:
    missing = [key for key in ("id", "sensor_id", "track_id", "time_utc", "class", "confidence", "bbox_norm", "source", "status") if key not in event]
    if missing:
        raise ValueError("В событии нет полей: " + ", ".join(missing))
    if event.get("cv") is not True:
        raise ValueError("Событие зрения должно содержать cv=true.")
    banned = FORBIDDEN.intersection(event)
    if banned:
        raise ValueError("Событие не должно содержать географию или дальность: " + ", ".join(sorted(banned)))
    box = event["bbox_norm"]
    for key in ("x", "y", "w", "h"):
        value = float(box[key])
        if value < -0.05 or value > 1.05:
            raise ValueError(f"Нормализованная координата {key} вне кадра: {value}")
    if "kmh" in event or "speed_kmh" in event:
        raise ValueError("Физическая скорость не включается в событие без отдельной калибровки на стороне потребителя.")


def events_from_tracks(tracks: list[dict], frame_w: int, frame_h: int, sensor_id: str, source: str) -> list[dict]:
    events = []
    for track in tracks:
        box = track.get("box")
        if not box:
            continue
        events.append(
            build_event(
                sensor_id=sensor_id,
                track_id=int(track.get("id") or 0),
                class_name=str(track.get("class_name") or ""),
                confidence=float(track.get("confidence") or 0.0),
                box=tuple(box),
                frame_w=frame_w,
                frame_h=frame_h,
                source=source,
                status=str(track.get("status") or ""),
                speed_px_s=float(track.get("speed_px_s") or 0.0),
            )
        )
    return events


class CommandCenterClient:
    """Write JSONL locally and optionally send one UDP datagram per event."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def send_events(
        self,
        events: list[dict],
        host: str,
        port: int,
        log_path: str | Path,
        min_interval: float = 0.2,
    ) -> int:
        now = time.monotonic()
        ready = []
        for event in events:
            validate_event(event)
            key = str(event["id"])
            if min_interval > 0 and now - self._last.get(key, 0.0) < min_interval:
                continue
            self._last[key] = now
            ready.append(event)
        if not ready:
            return 0
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for event in ready:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if host and int(port) > 0:
            payload_errors = []
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                for event in ready:
                    data = json.dumps(event, ensure_ascii=False).encode("utf-8")
                    sock.sendto(data, (host, int(port)))
            except OSError as exc:
                payload_errors.append(str(exc))
            finally:
                sock.close()
            if payload_errors:
                raise OSError(payload_errors[0])
        return len(ready)
