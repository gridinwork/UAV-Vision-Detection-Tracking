"""RTSP source. Open and read timeouts keep a dead URL from freezing the UI."""

from __future__ import annotations

import cv2


class RtspSource:
    kind = "rtsp"

    def __init__(self, url: str, timeout_ms: int = 8000) -> None:
        self.url = url.strip()
        if not self.url:
            raise ConnectionError("RTSP URL is empty.")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_ms))
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_ms))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.cap.release()
            raise ConnectionError(
                "Could not connect to the RTSP stream.\n"
                "Check the URL, the network, and that the camera is publishing."
            )
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps < 1.0 or self.fps > 240.0:
            self.fps = 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = 0

    def read(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.height, self.width = frame.shape[:2]
        return ok, frame

    def seek_frame(self, index: int) -> None:
        return None

    def seek_ratio(self, ratio: float) -> None:
        return None

    def info(self) -> dict:
        return {
            "kind": self.kind,
            "filename": self.url,
            "path": self.url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": 0,
            "duration": 0.0,
        }

    def release(self) -> None:
        self.cap.release()
