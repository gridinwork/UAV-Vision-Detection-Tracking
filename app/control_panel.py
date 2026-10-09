"""Source, playback, and detector controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QWidget):
    open_video = Signal()
    connect_webcam = Signal()
    connect_rtsp = Signal()
    disconnect_source = Signal()
    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    restart_clicked = Signal()
    seek_released = Signal(float)
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self._source_box())
        root.addWidget(self._playback_box())
        root.addWidget(self._detect_box())

    def _source_box(self) -> QGroupBox:
        box = QGroupBox("SOURCE")
        layout = QVBoxLayout(box)
        self.source_kind = QComboBox()
        self.source_kind.addItem("Video File", "video")
        self.source_kind.addItem("Webcam", "webcam")
        self.source_kind.addItem("RTSP", "rtsp")
        self.source_kind.currentIndexChanged.connect(self._toggle_source)
        layout.addWidget(self.source_kind)

        self.video_page = QWidget()
        video_layout = QHBoxLayout(self.video_page)
        video_layout.setContentsMargins(0, 0, 0, 0)
        open_btn = QPushButton("OPEN VIDEO")
        open_btn.setObjectName("Primary")
        open_btn.clicked.connect(self.open_video.emit)
        video_layout.addWidget(open_btn)
        layout.addWidget(self.video_page)

        self.webcam_page = QWidget()
        webcam_layout = QGridLayout(self.webcam_page)
        webcam_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_index = QComboBox()
        self.camera_index.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.camera_resolution = QComboBox()
        self.camera_resolution.addItems(["640x480", "1280x720", "1920x1080"])
        self.camera_resolution.setCurrentText("1280x720")
        webcam_btn = QPushButton("CONNECT")
        webcam_btn.clicked.connect(self.connect_webcam.emit)
        webcam_layout.addWidget(QLabel("Camera"), 0, 0)
        webcam_layout.addWidget(self.camera_index, 0, 1)
        webcam_layout.addWidget(QLabel("Resolution"), 1, 0)
        webcam_layout.addWidget(self.camera_resolution, 1, 1)
        webcam_layout.addWidget(webcam_btn, 2, 0, 1, 2)
        self.webcam_page.hide()
        layout.addWidget(self.webcam_page)

        self.rtsp_page = QWidget()
        rtsp_layout = QVBoxLayout(self.rtsp_page)
        rtsp_layout.setContentsMargins(0, 0, 0, 0)
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("rtsp://host:554/stream")
        rtsp_buttons = QHBoxLayout()
        connect_btn = QPushButton("CONNECT")
        disconnect_btn = QPushButton("DISCONNECT")
        connect_btn.clicked.connect(self.connect_rtsp.emit)
        disconnect_btn.clicked.connect(self.disconnect_source.emit)
        rtsp_buttons.addWidget(connect_btn)
        rtsp_buttons.addWidget(disconnect_btn)
        rtsp_layout.addWidget(QLabel("RTSP URL"))
        rtsp_layout.addWidget(self.rtsp_url)
        rtsp_layout.addLayout(rtsp_buttons)
        self.rtsp_page.hide()
        layout.addWidget(self.rtsp_page)

        self.file_info = QLabel("No source")
        self.file_info.setWordWrap(True)
        self.file_info.setObjectName("Subtitle")
        layout.addWidget(self.file_info)
        return box

    def _playback_box(self) -> QGroupBox:
        box = QGroupBox("PLAYBACK")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        for text, signal in (
            ("PLAY", self.play_clicked),
            ("PAUSE", self.pause_clicked),
            ("STOP", self.stop_clicked),
            ("RESTART", self.restart_clicked),
        ):
            button = QPushButton(text)
            button.clicked.connect(signal.emit)
            row.addWidget(button)
        layout.addLayout(row)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderReleased.connect(self._emit_seek)
        layout.addWidget(self.timeline)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("ReadoutVal")
        layout.addWidget(self.time_label)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Processing"))
        self.processing_mode = QComboBox()
        self.processing_mode.addItem("REALTIME", "REALTIME")
        self.processing_mode.addItem("EVERY FRAME", "EVERY_FRAME")
        self.processing_mode.currentIndexChanged.connect(self._notify)
        mode_row.addWidget(self.processing_mode)
        layout.addLayout(mode_row)
        return box

    def _detect_box(self) -> QGroupBox:
        box = QGroupBox("DETECTION")
        form = QFormLayout(box)
        self.detected = QLabel("0")
        self.detected.setObjectName("ReadoutVal")
        self.proc_fps = QLabel("0.0")
        self.proc_fps.setObjectName("ReadoutVal")
        self.detector_name = QLabel("YOLOv4-tiny")
        self.backend = QLabel("—")
        self.backend.setObjectName("ReadoutVal")
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        self.confidence.setValue(0.20)
        self.nms = QDoubleSpinBox()
        self.nms.setRange(0.05, 0.95)
        self.nms.setSingleStep(0.05)
        self.nms.setDecimals(2)
        self.nms.setValue(0.40)
        self.device = QComboBox()
        self.device.addItems(["AUTO", "CUDA", "CPU"])
        self.detection_scale = QComboBox()
        self.detection_scale.addItem("100%", 1.0)
        self.detection_scale.addItem("75%", 0.75)
        self.detection_scale.addItem("50%", 0.50)
        self.confidence.valueChanged.connect(self._notify)
        self.nms.valueChanged.connect(self._notify)
        self.device.currentIndexChanged.connect(self._notify)
        self.detection_scale.currentIndexChanged.connect(self._notify)
        form.addRow("Detected Cars", self.detected)
        form.addRow("Confidence Threshold", self.confidence)
        form.addRow("NMS Threshold", self.nms)
        form.addRow("Processing FPS", self.proc_fps)
        form.addRow("Detector", self.detector_name)
        form.addRow("Backend", self.backend)
        form.addRow("Device", self.device)
        form.addRow("Detection Scale", self.detection_scale)
        return box

    def _notify(self, *_args) -> None:
        self.changed.emit()

    def _toggle_source(self) -> None:
        kind = self.source_kind.currentData()
        self.video_page.setVisible(kind == "video")
        self.webcam_page.setVisible(kind == "webcam")
        self.rtsp_page.setVisible(kind == "rtsp")
        self._notify()

    def _emit_seek(self) -> None:
        self.seek_released.emit(self.timeline.value() / 1000.0)

    def set_source_info(self, info: dict) -> None:
        fps = info.get("fps") or 0
        duration = info.get("duration") or 0
        name = info.get("filename") or "Source"
        width = info.get("width") or 0
        height = info.get("height") or 0
        if duration:
            extra = f"\nDuration {_clock(duration)}"
        else:
            extra = ""
        self.file_info.setText(f"{name}\n{width}×{height}   {fps:.2f} FPS{extra}")

    def set_position(self, position: dict, user_holding: bool) -> None:
        duration = float(position.get("duration") or 0.0)
        current = float(position.get("time") or 0.0)
        self.time_label.setText(f"{_clock(current)} / {_clock(duration)}")
        if not user_holding and duration > 0:
            ratio = float(position.get("ratio") or 0.0)
            self.timeline.blockSignals(True)
            self.timeline.setValue(int(max(0.0, min(1.0, ratio)) * 1000))
            self.timeline.blockSignals(False)


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
