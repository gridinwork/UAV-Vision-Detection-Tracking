"""Aspect-fit video surface. Clicks are mapped back into frame pixels."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("VideoSurface")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background:#070b10; color:#8ea0b5;")
        self.setText("Open a video file to begin")
        self._image: QImage | None = None
        self._frame_size = (0, 0)

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self._frame_size = (image.width(), image.height())
        self.setText("")
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._image is None or self._image.isNull():
            return
        pixmap = QPixmap.fromImage(self._image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            mapped = self._map(event.position().toPoint())
            if mapped is not None:
                self.clicked.emit(mapped[0], mapped[1])
        super().mousePressEvent(event)

    def _map(self, pos: QPoint) -> tuple[float, float] | None:
        pixmap = self.pixmap()
        frame_w, frame_h = self._frame_size
        if pixmap is None or pixmap.isNull() or frame_w <= 0 or frame_h <= 0:
            return None
        pw, ph = pixmap.width(), pixmap.height()
        x0 = (self.width() - pw) / 2.0
        y0 = (self.height() - ph) / 2.0
        x, y = pos.x() - x0, pos.y() - y0
        if x < 0 or y < 0 or x > pw or y > ph:
            return None
        return x * frame_w / pw, y * frame_h / ph
