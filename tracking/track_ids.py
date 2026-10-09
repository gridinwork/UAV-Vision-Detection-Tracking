"""Lightweight identity: IoU, then centroid distance."""

from __future__ import annotations

from dataclasses import dataclass, field


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def center_of(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    confidence: float
    class_name: str = "car"
    missed: int = 0
    hits: int = 1
    vx: float = 0.0
    vy: float = 0.0
    trail: list = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return center_of(self.box)

    def predicted_center(self) -> tuple[float, float]:
        cx, cy = self.center
        steps = max(1, self.missed)
        return cx + self.vx * steps, cy + self.vy * steps

    def update(self, box: tuple[int, int, int, int], confidence: float, class_name: str) -> None:
        old_x, old_y = self.center
        new_x, new_y = center_of(box)
        self.vx = 0.65 * self.vx + 0.35 * (new_x - old_x)
        self.vy = 0.65 * self.vy + 0.35 * (new_y - old_y)
        self.box = box
        self.confidence = confidence
        self.class_name = class_name
        self.missed = 0
        self.hits += 1

    def mark_missed(self) -> None:
        self.missed += 1


def match_tracks(
    tracks: list[Track],
    detections: list,
    iou_threshold: float = 0.25,
    centroid_px: float = 90.0,
) -> list[tuple[int, int]]:
    """Return pairs of (track_index, detection_index)."""
    if not tracks or not detections:
        return []
    pairs: list[tuple[float, int, int]] = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            score = iou(track.box, det.box)
            if score >= iou_threshold:
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))
    for ti, track in enumerate(tracks):
        if ti in used_t:
            continue
        pcx, pcy = track.predicted_center()
        best_d = -1
        best_dist = centroid_px
        for di, det in enumerate(detections):
            if di in used_d:
                continue
            cx, cy = det.center
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_d = di
        if best_d >= 0:
            used_t.add(ti)
            used_d.add(best_d)
            matches.append((ti, best_d))
    return matches
