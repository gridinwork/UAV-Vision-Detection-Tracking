"""Record exactly 15 seconds of processed frames, timed with perf_counter.

Output FPS follows the source. If the detector is slower than that FPS, the
latest processed frame is repeated so the file duration stays 15 seconds.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from recording.video_encoder import open_writer


RECORD_SECONDS = 15.0


class SessionRecorder:
    def __init__(self) -> None:
        self.writer = None
        self.path: Path | None = None
        self.fps = 30.0
        self.size = (0, 0)
        self.codec = ""
        self.t0 = 0.0
        self.next_write = 0.0
        self.frames_written = 0
        self.active = False
        self._last = None
        self.limit = RECORD_SECONDS

    def start(self, path: Path, fps: float, frame_shape: tuple[int, int], duration: float | None = None) -> None:
        height, width = frame_shape
        self.writer, self.codec, self.size, self.fps = open_writer(str(path), fps, (width, height))
        self.path = path
        self.limit = float(duration) if duration and duration > 0 else RECORD_SECONDS
        self.t0 = time.perf_counter()
        self.next_write = self.t0
        self.frames_written = 0
        self.active = True
        self._last = None

    @property
    def remaining(self) -> float:
        if not self.active:
            return 0.0
        return max(0.0, self.limit - (time.perf_counter() - self.t0))

    @property
    def elapsed(self) -> float:
        if not self.active:
            return 0.0
        return min(self.limit, time.perf_counter() - self.t0)

    def submit(self, frame) -> bool:
        """Write every source-FPS slot up to now. Returns False when the 15s ends."""
        if not self.active or self.writer is None:
            return False
        self._last = self._fit(frame)
        return self._drain()

    def _fit(self, frame):
        width, height = self.size
        if frame.shape[1] != width or frame.shape[0] != height:
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame

    def _drain(self) -> bool:
        if self._last is None or self.writer is None:
            return self.active
        now = time.perf_counter()
        interval = 1.0 / self.fps
        if now - self.t0 >= self.limit:
            self.stop()
            return False
        while self.next_write <= now and (self.next_write - self.t0) < self.limit:
            self.writer.write(self._last)
            self.frames_written += 1
            self.next_write += interval
            if self.frames_written > int(self.fps * self.limit) + 2:
                break
        if time.perf_counter() - self.t0 >= self.limit:
            self.stop()
            return False
        return True

    def stop(self) -> Path | None:
        if self.writer is not None:
            # Fill the timeline up to 15s so the container duration matches the clock.
            if self._last is not None and self.t0:
                interval = 1.0 / self.fps
                limit = self.t0 + self.limit
                while self.next_write < limit and self.frames_written < int(round(self.fps * self.limit)):
                    self.writer.write(self._last)
                    self.frames_written += 1
                    self.next_write += interval
            self.writer.release()
        self.writer = None
        self.active = False
        return self.path


class RecordingWorker(QThread):
    """Writes processed frames on a side thread for exactly 15 seconds."""

    tick = Signal(float)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._pending: queue.Queue = queue.Queue()
        self._stop = False
        self.recorder = SessionRecorder()

    def stop(self) -> None:
        self._stop = True
        self._pending.put(("stop", None))

    def request_start(self, path: Path, fps: float, shape: tuple[int, int], duration: float | None = None) -> None:
        self._pending.put(("start", (path, fps, shape, duration)))

    def submit_frame(self, frame) -> None:
        if not self.recorder.active and self._pending.empty():
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                return

    def run(self) -> None:
        last = None
        while not self._stop:
            self._drain_pending()
            try:
                last = self._queue.get(timeout=0.03)
            except queue.Empty:
                pass
            if not self.recorder.active or last is None:
                continue
            try:
                alive = self.recorder.submit(last)
            except Exception as exc:
                self.recorder.stop()
                self.failed.emit(str(exc))
                continue
            self.tick.emit(self.recorder.remaining)
            if not alive:
                path = self.recorder.path
                self.finished_ok.emit(str(path) if path else "")
                last = None

    def _drain_pending(self) -> None:
        while True:
            try:
                name, payload = self._pending.get_nowait()
            except queue.Empty:
                return
            if name == "start" and payload is not None:
                path, fps, shape, duration = payload if len(payload) == 4 else (*payload, None)
                try:
                    if self.recorder.active:
                        self.recorder.stop()
                    self.recorder.start(path, fps, shape, duration)
                except Exception as exc:
                    self.failed.emit(str(exc))
            elif name == "stop" and self.recorder.active:
                done = self.recorder.stop()
                if done is not None:
                    self.finished_ok.emit(str(done))
