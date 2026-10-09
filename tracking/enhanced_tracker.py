"""Identity-preserving tracker: click lock, loss timeout, reacquire."""

from __future__ import annotations

import time

from tracking.moving_average import MovingAverageFilter
from tracking.target_state import TargetState
from tracking.track_ids import Track, match_tracks


class EnhancedTracker:
    def __init__(self, window: int = 20) -> None:
        self.filter = MovingAverageFilter(window)
        self.tracks: list[Track] = []
        self._next_id = 1
        self.selected_id: int | None = None
        self.manual_override = False
        self._lost_since: float | None = None
        self._reacquired_until = 0.0
        self._last_status = "SEARCHING"

    def reset(self) -> None:
        self.filter.reset()
        self.tracks.clear()
        self._next_id = 1
        self.selected_id = None
        self.manual_override = False
        self._lost_since = None
        self._reacquired_until = 0.0
        self._last_status = "SEARCHING"

    def set_window(self, window: int) -> None:
        self.filter.set_window(window)

    def clear_target(self) -> None:
        self.selected_id = None
        self.manual_override = False
        self._lost_since = None
        self.filter.reset()

    def cycle(self, step: int) -> None:
        live = [track for track in self.tracks if track.missed == 0]
        if not live:
            return
        live.sort(key=lambda track: track.track_id)
        ids = [track.track_id for track in live]
        if self.selected_id in ids:
            index = ids.index(self.selected_id)
        else:
            index = -1
        self.selected_id = ids[(index + step) % len(ids)]
        self.manual_override = True
        self._lost_since = None
        self.filter.reset()

    def update(
        self,
        detections: list,
        frame_w: int,
        frame_h: int,
        frame_id: int,
        dead_half: tuple[float, float],
        click_xy: tuple[float, float] | None,
        mode: str,
        lost_timeout: float,
        auto_reacquire: bool,
        trail_length: int,
        now: float | None = None,
    ) -> TargetState:
        now = time.perf_counter() if now is None else now
        self._associate(detections)
        if click_xy is not None:
            self._apply_click(click_xy)
        self._select(mode, frame_w, frame_h, auto_reacquire, lost_timeout, now)
        self._trim(trail_length)
        return self._build_state(
            detections, frame_w, frame_h, frame_id, dead_half, trail_length, now, lost_timeout
        )

    def _associate(self, detections: list) -> None:
        matches = match_tracks(self.tracks, detections)
        matched_tracks = {ti for ti, _di in matches}
        matched_dets = {di for _ti, di in matches}
        for ti, di in matches:
            det = detections[di]
            self.tracks[ti].update(det.box, det.confidence, det.class_name)
        for index, track in enumerate(self.tracks):
            if index not in matched_tracks:
                track.mark_missed()
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    box=det.box,
                    confidence=det.confidence,
                    class_name=det.class_name,
                )
            )
            self._next_id += 1

    def _apply_click(self, click_xy: tuple[float, float]) -> None:
        x, y = click_xy
        chosen: Track | None = None
        for track in self.tracks:
            if track.missed > 0:
                continue
            bx, by, bw, bh = track.box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                chosen = track
                break
        if chosen is None:
            best = 1e18
            for track in self.tracks:
                if track.missed > 0:
                    continue
                cx, cy = track.center
                dist = (cx - x) ** 2 + (cy - y) ** 2
                if dist < best and dist <= 80 ** 2:
                    best = dist
                    chosen = track
        if chosen is not None:
            self.selected_id = chosen.track_id
            self.manual_override = True
            self._lost_since = None
            self.filter.reset()

    def _select(
        self,
        mode: str,
        frame_w: int,
        frame_h: int,
        auto_reacquire: bool,
        lost_timeout: float,
        now: float,
    ) -> None:
        visible = [track for track in self.tracks if track.missed == 0]
        selected = self._track_by_id(self.selected_id) if self.selected_id else None
        selected_visible = selected is not None and selected.missed == 0

        if selected is not None and not selected_visible:
            if self._lost_since is None:
                self._lost_since = now
            elapsed = now - self._lost_since
            recovered = self._recover_same_id(selected)
            if recovered is not None:
                self.selected_id = recovered.track_id
                self._lost_since = None
                self._reacquired_until = now + 1.0
                return
            if elapsed < lost_timeout:
                return
            allow = auto_reacquire and (mode != "CLICK" or auto_reacquire)
            if mode == "CLICK" and not auto_reacquire:
                allow = False
            if allow and visible:
                self.selected_id = self._pick(visible, mode, frame_w, frame_h).track_id
                self._lost_since = None
                self._reacquired_until = now + 1.0
                self.manual_override = False
                self.filter.reset()
            return

        if selected_visible:
            if self._lost_since is not None:
                self._reacquired_until = now + 1.0
            self._lost_since = None
            return

        if mode == "CLICK" or self.manual_override:
            return
        if not visible:
            return
        pick = self._pick(visible, mode, frame_w, frame_h)
        if pick is not None:
            if self.selected_id != pick.track_id:
                self.filter.reset()
            self.selected_id = pick.track_id

    def _recover_same_id(self, selected: Track) -> Track | None:
        px, py = selected.predicted_center()
        best: Track | None = None
        best_dist = 110.0
        for track in self.tracks:
            if track.missed > 0 or track.track_id == selected.track_id:
                continue
            cx, cy = track.center
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = track
        if best is None:
            return None
        # Keep the locked identity when the car reappears near the prediction.
        best.track_id = selected.track_id
        self.tracks = [track for track in self.tracks if track is not selected]
        return best

    def _pick(self, visible: list[Track], mode: str, frame_w: int, frame_h: int) -> Track:
        if mode == "FIRST":
            return max(visible, key=lambda track: (track.hits, track.confidence))
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        return min(visible, key=lambda track: (track.center[0] - cx) ** 2 + (track.center[1] - cy) ** 2)

    def _track_by_id(self, track_id: int | None) -> Track | None:
        if track_id is None:
            return None
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def _trim(self, trail_length: int) -> None:
        kept: list[Track] = []
        for track in self.tracks:
            if track.track_id == self.selected_id:
                kept.append(track)
                continue
            if track.missed <= 12:
                kept.append(track)
        self.tracks = kept
        _ = trail_length

    def _build_state(
        self,
        detections: list,
        frame_w: int,
        frame_h: int,
        frame_id: int,
        dead_half: tuple[float, float],
        trail_length: int,
        now: float,
        lost_timeout: float,
    ) -> TargetState:
        state = TargetState(
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
            detections=list(detections),
            dead_zone_half=dead_half,
        )
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        public_tracks = []
        for track in self.tracks:
            if track.missed > 0 and track.track_id != self.selected_id:
                continue
            public_tracks.append(
                {
                    "id": track.track_id,
                    "box": track.box,
                    "confidence": track.confidence,
                    "class_name": track.class_name,
                    "selected": track.track_id == self.selected_id and track.missed == 0,
                    "missed": track.missed,
                }
            )
        state.tracks = public_tracks
        selected = self._track_by_id(self.selected_id)
        if selected is None or (selected.missed > 0):
            if not detections and selected is None:
                state.status = "SEARCHING"
            elif selected is not None and selected.missed > 0:
                state.status = "LOST"
                state.lost = True
                state.target_id = selected.track_id
                state.target_label = f"CAR #{selected.track_id}"
                state.confidence = selected.confidence
                state.trail = list(selected.trail[-trail_length:])
            elif detections:
                state.status = "DETECTED"
            else:
                state.status = "SEARCHING"
            state.filtered_xy = (cx, cy)
            state.raw_xy = (cx, cy)
            state.inside_dead_zone = True
            self._last_status = state.status
            return state

        raw_x, raw_y = selected.center
        filt_x, filt_y = self.filter.push(raw_x, raw_y)
        selected.trail.append((float(filt_x), float(filt_y)))
        if len(selected.trail) > max(trail_length, 100):
            selected.trail = selected.trail[-max(trail_length, 100) :]
        half_w, half_h = dead_half
        inside = abs(filt_x - cx) <= half_w and abs(filt_y - cy) <= half_h
        area = (selected.box[2] * selected.box[3]) / float(frame_w * frame_h)
        state.target_id = selected.track_id
        state.target_label = f"CAR #{selected.track_id}"
        state.confidence = selected.confidence
        state.raw_xy = (raw_x, raw_y)
        state.filtered_xy = (float(filt_x), float(filt_y))
        state.box = selected.box
        state.trail = list(selected.trail[-trail_length:])
        state.inside_dead_zone = inside
        state.area_ratio = area
        state.lost = False
        if now < self._reacquired_until:
            state.status = "REACQUIRED"
        elif inside:
            state.status = "CENTERED"
        else:
            state.status = "CORRECTING"
        self._last_status = state.status
        _ = lost_timeout
        return state
