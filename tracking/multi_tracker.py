"""ByteTrack-style and BoT-SORT-style multi-object tracking.

BoT-SORT here is the motion model: a Kalman filter on the box plus IoU
association. An appearance / ReID encoder is not included.

Track ids stay on the same object through short detection gaps. Status values
are DETECTED, TRACKING, OCCLUDED, LOST and REACQUIRED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]
    status: str = "DETECTED"
    hits: int = 1
    missed: int = 0
    age: int = 1
    trail: list = field(default_factory=list)
    speed_px_s: float = 0.0
    speed_kmh: float | None = None
    selected: bool = False
    reacquire_ttl: int = 0
    kalman: object | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box": self.box,
            "status": self.status,
            "missed": self.missed,
            "selected": self.selected,
            "trail": [(point[1], point[2]) for point in self.trail[-80:]],
            "speed_px_s": self.speed_px_s,
            "speed_kmh": self.speed_kmh,
        }


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = float(aw * ah + bw * bh) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def associate(tracks: list[Track], detections, threshold: float):
    candidates = []
    for track_index, track in enumerate(tracks):
        for det_index, detection in enumerate(detections):
            score = iou_xywh(track.box, detection.box)
            if score >= threshold:
                candidates.append((score, track_index, det_index))
    candidates.sort(reverse=True)
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matched = []
    for _score, track_index, det_index in candidates:
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        matched.append((track_index, det_index))
    unmatched_tracks = [index for index in range(len(tracks)) if index not in used_tracks]
    unmatched_dets = [index for index in range(len(detections)) if index not in used_dets]
    return matched, unmatched_tracks, unmatched_dets


def _center(box) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def speed_px_s(trail, now: float) -> float:
    window = [point for point in trail if now - point[0] <= 0.6]
    if len(window) < 2:
        return 0.0
    dt = window[-1][0] - window[0][0]
    if dt <= 1e-3:
        return 0.0
    return math.hypot(window[-1][1] - window[0][1], window[-1][2] - window[0][2]) / dt


def speed_kmh(px_s: float, meters_per_pixel: float) -> float | None:
    if meters_per_pixel is None or meters_per_pixel <= 0.0:
        return None
    return px_s * float(meters_per_pixel) * 3.6


class BoxKalman:
    """Constant-velocity box filter used by the BoT-SORT motion stage."""

    def __init__(self, box) -> None:
        cx, cy = _center(box)
        _x, _y, w, h = box
        self.x = np.array([cx, cy, float(w), float(h), 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.p = np.eye(8) * 20.0
        self.q = np.eye(8)
        self.q[4:, 4:] *= 0.4
        self.r = np.eye(4) * 2.0

    def predict(self, dt: float):
        transition = np.eye(8)
        transition[0, 4] = dt
        transition[1, 5] = dt
        transition[2, 6] = dt
        transition[3, 7] = dt
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + self.q
        return self.box()

    def update(self, box):
        cx, cy = _center(box)
        measurement = np.array([cx, cy, float(box[2]), float(box[3])], dtype=float)
        observation = np.zeros((4, 8))
        observation[0, 0] = observation[1, 1] = observation[2, 2] = observation[3, 3] = 1.0
        innovation = measurement - observation @ self.x
        covariance = observation @ self.p @ observation.T + self.r
        gain = self.p @ observation.T @ np.linalg.inv(covariance)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(8) - gain @ observation) @ self.p
        return self.box()

    def box(self) -> tuple[int, int, int, int]:
        cx, cy, w, h = self.x[:4]
        w = max(2.0, float(w))
        h = max(2.0, float(h))
        return (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))


class ByteTracker:
    """Two-stage IoU association: high-confidence detections, then low-confidence."""

    use_motion = False

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self._next_id = 1
        self.high_conf = 0.5
        self.low_conf = 0.1
        self.match_iou = 0.3
        self.min_hits = 3

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections,
        now: float,
        fps: float,
        lost_timeout: float,
        meters_per_pixel: float = 0.0,
        high_conf: float | None = None,
        low_conf: float | None = None,
        match_iou: float | None = None,
    ) -> list[Track]:
        fps = max(1.0, float(fps or 25.0))
        if high_conf is not None:
            self.high_conf = float(high_conf)
        if low_conf is not None:
            self.low_conf = float(low_conf)
        if match_iou is not None:
            self.match_iou = float(match_iou)
        dt = 1.0 / fps
        if self.use_motion:
            for track in self.tracks:
                if track.kalman is not None:
                    track.box = track.kalman.predict(dt)
        high = [item for item in detections if item.confidence >= self.high_conf]
        low = [item for item in detections if self.low_conf <= item.confidence < self.high_conf]
        matched, unmatched_tracks, unmatched_high = associate(self.tracks, high, self.match_iou)
        for track_index, det_index in matched:
            self._hit(self.tracks[track_index], high[det_index], now, fps, meters_per_pixel)
        rest = [self.tracks[index] for index in unmatched_tracks]
        pairs2, still_open, _unused = associate(rest, low, self.match_iou)
        recovered = set()
        for track_index, det_index in pairs2:
            source = unmatched_tracks[track_index]
            self._hit(self.tracks[source], low[det_index], now, fps, meters_per_pixel)
            recovered.add(source)
        occ_frames = max(2, int(round(0.45 * fps)))
        lost_frames = max(occ_frames + 1, int(round(float(lost_timeout) * fps)))
        for track_index in unmatched_tracks:
            if track_index in recovered:
                continue
            self._miss(self.tracks[track_index], occ_frames)
        for det_index in unmatched_high:
            self._spawn(high[det_index], now, meters_per_pixel)
        self.tracks = [track for track in self.tracks if track.missed <= lost_frames]
        return list(self.tracks)

    def _spawn(self, detection, now: float, meters_per_pixel: float) -> None:
        cx, cy = detection.center
        track = Track(
            track_id=self._next_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=float(detection.confidence),
            box=tuple(int(v) for v in detection.box),
            status="DETECTED",
            hits=1,
            trail=[(now, cx, cy)],
        )
        if self.use_motion:
            track.kalman = BoxKalman(track.box)
        track.speed_kmh = speed_kmh(0.0, meters_per_pixel)
        self._next_id += 1
        self.tracks.append(track)

    def _hit(self, track: Track, detection, now: float, fps: float, meters_per_pixel: float) -> None:
        was_gap = track.missed > 0 or track.status in ("OCCLUDED", "LOST")
        box = tuple(int(v) for v in detection.box)
        if track.kalman is not None:
            box = track.kalman.update(box)
        elif self.use_motion:
            track.kalman = BoxKalman(box)
        track.box = box
        track.class_id = detection.class_id
        track.class_name = detection.class_name
        track.confidence = float(detection.confidence)
        track.missed = 0
        track.hits += 1
        track.age += 1
        cx, cy = _center(track.box)
        track.trail.append((now, cx, cy))
        if len(track.trail) > 120:
            track.trail = track.trail[-120:]
        track.speed_px_s = speed_px_s(track.trail, now)
        track.speed_kmh = speed_kmh(track.speed_px_s, meters_per_pixel)
        if was_gap and track.hits > 2:
            track.status = "REACQUIRED"
            track.reacquire_ttl = max(3, int(round(0.25 * fps)))
        elif track.reacquire_ttl > 0:
            track.reacquire_ttl -= 1
            track.status = "REACQUIRED" if track.reacquire_ttl > 0 else "TRACKING"
        elif track.hits < self.min_hits:
            track.status = "DETECTED"
        else:
            track.status = "TRACKING"

    def _miss(self, track: Track, occ_frames: int) -> None:
        track.missed += 1
        track.age += 1
        track.reacquire_ttl = 0
        if track.missed <= occ_frames:
            track.status = "OCCLUDED"
        else:
            track.status = "LOST"


class BotSortTracker(ByteTracker):
    """BoT-SORT motion association without a ReID appearance model."""

    use_motion = True


class TargetLock:
    def __init__(self) -> None:
        self.locked_id: int | None = None

    def reset(self) -> None:
        self.locked_id = None

    def clear(self) -> None:
        self.locked_id = None

    def cycle(self, step: int, tracks: list[Track]) -> None:
        ids = [track.track_id for track in tracks if track.status != "LOST"]
        if not ids:
            return
        if self.locked_id not in ids:
            self.locked_id = ids[0]
            return
        index = ids.index(self.locked_id)
        self.locked_id = ids[(index + int(step)) % len(ids)]

    def choose(self, tracks: list[Track], frame_w: int, frame_h: int, click, mode: str) -> Track | None:
        if mode == "CLICK" and click is not None:
            x, y = click
            for track in tracks:
                bx, by, bw, bh = track.box
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.locked_id = track.track_id
                    break
        elif mode == "FIRST":
            alive = [track for track in tracks if track.status in ("DETECTED", "TRACKING", "REACQUIRED")]
            if self.locked_id is None or not any(track.track_id == self.locked_id for track in tracks):
                if alive:
                    self.locked_id = alive[0].track_id
        elif mode == "AUTO_CENTER":
            pool = [track for track in tracks if track.status in ("DETECTED", "TRACKING", "REACQUIRED", "OCCLUDED")]
            if pool:
                cx, cy = frame_w / 2.0, frame_h / 2.0
                self.locked_id = min(pool, key=lambda track: _distance(track, cx, cy)).track_id
        selected = None
        for track in tracks:
            track.selected = track.track_id == self.locked_id
            if track.selected:
                selected = track
        return selected


def _distance(track: Track, x: float, y: float) -> float:
    cx, cy = _center(track.box)
    return math.hypot(cx - x, cy - y)
