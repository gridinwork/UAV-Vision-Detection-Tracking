"""Video-file source with seek."""

from __future__ import annotations

from pathlib import Path

import cv2


class VideoFileSource:
    kind = "video"

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file:\n{self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps < 1.0 or self.fps > 240.0:
            self.fps = 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.index = 0
        self.content_index = 0

    @property
    def duration(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return self.frame_count / self.fps

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.content_index = self.index
            self.index += 1
        return ok, frame

    def seek_frame(self, index: int) -> None:
        index = max(0, int(index))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.index = index
        self.content_index = index

    def seek_ratio(self, ratio: float) -> None:
        if self.frame_count <= 0:
            return
        self.seek_frame(int(max(0.0, min(1.0, ratio)) * max(0, self.frame_count - 1)))

    def info(self) -> dict:
        return {
            "kind": self.kind,
            "filename": Path(self.path).name,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
        }

    def release(self) -> None:
        self.cap.release()
