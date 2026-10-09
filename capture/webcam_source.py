"""Local camera. DirectShow is used on Windows."""

from __future__ import annotations

import sys

import cv2


class WebcamSource:
    kind = "webcam"

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        api = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, api)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.index}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps < 1.0 or self.fps > 240.0:
            self.fps = 30.0
        self.frame_count = 0
        self.index_frame = 0

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.index_frame += 1
            self.height, self.width = frame.shape[:2]
        return ok, frame

    def seek_frame(self, index: int) -> None:
        return None

    def seek_ratio(self, ratio: float) -> None:
        return None

    def info(self) -> dict:
        return {
            "kind": self.kind,
            "filename": f"Camera {self.index}",
            "path": str(self.index),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": 0,
            "duration": 0.0,
        }

    def release(self) -> None:
        self.cap.release()
