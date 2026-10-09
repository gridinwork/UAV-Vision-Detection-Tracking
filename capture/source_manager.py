"""Capture thread. Live sources keep only the latest frame."""

from __future__ import annotations

import queue
import threading
import time

from PySide6.QtCore import QThread, Signal

from capture.rtsp_source import RtspSource
from capture.video_source import VideoFileSource
from capture.webcam_source import WebcamSource
from utils.logger import get_logger


log = get_logger()


class FrameHub:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.frame = None
        self.meta: dict = {}
        self.seq = 0
        self.consumed = 0
        self.published = 0

    def publish(self, frame, meta: dict, wait_consume: bool) -> None:
        with self._cond:
            self.seq += 1
            self.published += 1
            self.frame = frame
            self.meta = dict(meta)
            token = self.seq
            self._cond.notify_all()
            if wait_consume:
                while self.consumed < token:
                    self._cond.wait(timeout=0.3)

    def latest(self):
        with self._cond:
            return self.seq, self.frame, dict(self.meta)

    def ack(self, seq: int) -> None:
        with self._cond:
            if seq > self.consumed:
                self.consumed = seq
            self._cond.notify_all()

    def dropped(self) -> int:
        with self._cond:
            return max(0, self.published - self.consumed)


class CaptureWorker(QThread):
    info_ready = Signal(dict)
    error = Signal(str)
    ended = Signal()
    position = Signal(dict)

    def __init__(self, hub: FrameHub) -> None:
        super().__init__()
        self.hub = hub
        self._commands: queue.Queue = queue.Queue()
        self._stop = False
        self._source = None
        self._playing = False
        self._mode = "REALTIME"
        self._ended = False

    def post(self, name: str, **payload) -> None:
        self._commands.put((name, payload))

    def stop(self) -> None:
        self._stop = True
        self.post("noop")
        self.hub.ack(10**12)

    def run(self) -> None:
        while not self._stop:
            self._drain()
            if self._source is None or not self._playing:
                time.sleep(0.02)
                continue
            self._read_once()

    def _drain(self) -> None:
        while True:
            try:
                name, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            self._handle(name, payload)

    def _handle(self, name: str, payload: dict) -> None:
        if name == "open_video":
            self._open(lambda: VideoFileSource(payload["path"]), "Video opened")
        elif name == "open_webcam":
            self._open(
                lambda: WebcamSource(payload["index"], payload["width"], payload["height"]),
                "Webcam opened",
            )
        elif name == "open_rtsp":
            self._open(lambda: RtspSource(payload["url"]), "RTSP connected")
        elif name == "close":
            self._close()
        elif name == "play":
            self._playing = self._source is not None
            self._ended = False
        elif name == "pause":
            self._playing = False
        elif name == "stop":
            self._playing = False
            if self._source is not None:
                self._source.seek_frame(0)
                self._publish_seek_frame()
        elif name == "restart":
            if self._source is not None:
                self._source.seek_frame(0)
                self._ended = False
                self._playing = True
        elif name == "seek":
            if self._source is not None and self._source.kind == "video":
                self._source.seek_ratio(float(payload.get("ratio", 0.0)))
                self._ended = False
                self._publish_seek_frame()
        elif name == "mode":
            self._mode = payload.get("mode", "REALTIME")
        elif name == "noop":
            return

    def _open(self, factory, log_message: str) -> None:
        self._close()
        try:
            source = factory()
        except Exception as exc:
            log.info("Source open failed: %s", exc)
            self.error.emit(str(exc))
            return
        self._source = source
        self._playing = False
        self._ended = False
        info = source.info()
        log.info("%s | %s | %sx%s @ %.2f fps", log_message, info.get("filename"), info.get("width"), info.get("height"), info.get("fps") or 0)
        self.info_ready.emit(info)
        self._publish_seek_frame()

    def _close(self) -> None:
        self._playing = False
        if self._source is not None:
            self._source.release()
            self._source = None

    def _publish_seek_frame(self) -> None:
        if self._source is None:
            return
        ok, frame = self._source.read()
        if not ok or frame is None:
            return
        # Seek consumes the frame at the new position; step the index back so
        # the position label matches the frame just shown.
        if self._source.kind == "video":
            self._source.index = max(0, self._source.index - 1)
        self._emit_frame(frame)

    def _read_once(self) -> None:
        source = self._source
        if source is None:
            return
        deadline = None
        if source.kind == "video" and self._mode == "REALTIME":
            deadline = time.perf_counter() + (1.0 / source.fps)
        ok, frame = source.read()
        if not ok or frame is None:
            if source.kind == "video" and not self._ended:
                self._ended = True
                self._playing = False
                self.ended.emit()
            elif source.kind != "video":
                self.error.emit("The live source stopped sending frames.")
                self._playing = False
            return
        self._emit_frame(frame)
        if source.kind == "video" and self._mode == "REALTIME" and deadline is not None:
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def _emit_frame(self, frame) -> None:
        source = self._source
        if source is None:
            return
        meta = source.info()
        if source.kind == "video":
            meta["frame_index"] = int(getattr(source, "content_index", max(0, int(getattr(source, "index", 1)) - 1)))
        else:
            meta["frame_index"] = int(getattr(source, "index", 0))
        wait = source.kind == "video" and self._mode == "EVERY_FRAME"
        self.hub.publish(frame.copy(), meta, wait_consume=wait)
        if source.kind == "video":
            index = int(meta["frame_index"])
            count = int(meta.get("frame_count") or 0)
            fps = float(meta.get("fps") or 30.0)
            self.position.emit(
                {
                    "index": index,
                    "count": count,
                    "time": index / fps if fps else 0.0,
                    "duration": float(meta.get("duration") or 0.0),
                    "ratio": (index / count) if count else 0.0,
                }
            )
