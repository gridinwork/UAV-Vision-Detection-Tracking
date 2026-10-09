"""Ground-control window for the vision studio."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.control_panel import ControlPanel
from app.performance_panel import PerformancePanel
from app.settings_dialog import SettingsDialog
from app.tracking_panel import TrackingPanel
from app.video_widget import VideoWidget
from capture.source_manager import CaptureWorker, FrameHub
from detection.detector_worker import DetectionWorker
from recording.recorder import RecordingWorker
from tracking.engine import PipelineConfig
from utils.logger import get_logger
from utils.paths import CONFIG_PATH, RECORDINGS_DIR, ensure_dirs


log = get_logger()

OVERLAYS = (
    ("boxes", "Car Boxes", True),
    ("confidence", "Confidence", True),
    ("ids", "IDs", True),
    ("trail", "Target Trail", True),
    ("crosshair", "Center Crosshair", True),
    ("dead_zone", "Dead Zone", True),
    ("flight_director", "Flight Director", True),
    ("command_text", "Command Text", True),
    ("fps", "FPS", True),
    ("target_vector", "Target Vector", True),
    ("debug", "Debug", False),
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("Autonomous VTOL Vehicle Tracking Studio")
        self.resize(1480, 920)
        self.hub = FrameHub()
        self.capture = CaptureWorker(self.hub)
        self.recorder = RecordingWorker()
        self.detector = DetectionWorker(self.hub, self.recorder)
        self._slider_held = False
        self._have_frame = False
        self._source_fps = 30.0
        self._frame_shape = (0, 0)
        self._video_path = ""
        self._last_status = ""
        self._build()
        self._wire()
        self._load_settings()
        self._push_config()
        self.capture.start()
        self.recorder.start()
        self.detector.start()
        self._rec_clock = QTimer(self)
        self._rec_clock.setInterval(100)
        self._rec_clock.timeout.connect(self._refresh_rec_label)

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self._header())

        self.video = VideoWidget()
        video_frame = QFrame()
        video_frame.setObjectName("VideoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(6, 6, 6, 6)
        video_layout.addWidget(self.video)

        self.control = ControlPanel()
        self.tracking = TrackingPanel()
        self.performance = PerformancePanel()
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.control)
        side_layout.addWidget(self.tracking)
        side_layout.addWidget(self.performance)
        side_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(460)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(video_frame)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._bottom())
        self.statusBar().showMessage("Flight backend: VIRTUAL  ·  no vehicle link")

    def _header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Header")
        layout = QHBoxLayout(frame)
        title_box = QVBoxLayout()
        title = QLabel("AUTONOMOUS VTOL VEHICLE TRACKING STUDIO")
        title.setObjectName("Title")
        subtitle = QLabel("Bird's-eye car tracking  ·  simulation only  ·  commands are not sent to a vehicle")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, 1)
        self.backend_badge = QLabel("FLIGHT BACKEND: VIRTUAL")
        self.backend_badge.setObjectName("BadgeVirtual")
        self.model_badge = QLabel("MODEL: LOADING")
        self.model_badge.setObjectName("BadgeModel")
        layout.addWidget(self.backend_badge)
        layout.addWidget(self.model_badge)
        return frame

    def _bottom(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("BottomBar")
        layout = QHBoxLayout(frame)
        self.overlay_boxes: dict[str, QCheckBox] = {}
        for key, title, default in OVERLAYS:
            box = QCheckBox(title)
            box.setChecked(default)
            box.toggled.connect(self._on_overlay)
            self.overlay_boxes[key] = box
            layout.addWidget(box)
        layout.addStretch(1)
        self.rec_label = QLabel("")
        self.rec_label.setObjectName("RecBadge")
        self.record_btn = QPushButton("RECORD")
        self.record_btn.setObjectName("Record")
        self.record_btn.setEnabled(False)
        shot_btn = QPushButton("SAVE SCREENSHOT")
        csv_btn = QPushButton("EXPORT CONTROL CSV")
        export_btn = QPushButton("EXPORT FULL VIDEO")
        settings_btn = QPushButton("SETTINGS")
        self.record_btn.clicked.connect(self._record)
        shot_btn.clicked.connect(self.detector.request_screenshot)
        csv_btn.clicked.connect(self.detector.request_csv)
        export_btn.clicked.connect(self._export_full)
        settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(self.rec_label)
        layout.addWidget(self.record_btn)
        layout.addWidget(shot_btn)
        layout.addWidget(csv_btn)
        layout.addWidget(export_btn)
        layout.addWidget(settings_btn)
        return frame

    def _wire(self) -> None:
        self.control.open_video.connect(self._open_video)
        self.control.connect_webcam.connect(self._open_webcam)
        self.control.connect_rtsp.connect(self._open_rtsp)
        self.control.disconnect_source.connect(self._disconnect)
        self.control.play_clicked.connect(self._play)
        self.control.pause_clicked.connect(self._pause)
        self.control.stop_clicked.connect(self._stop)
        self.control.restart_clicked.connect(self._restart)
        self.control.seek_released.connect(self._seek)
        self.control.timeline.sliderPressed.connect(self._hold_slider)
        self.control.timeline.sliderReleased.connect(self._release_slider)
        self.control.changed.connect(self._push_config)
        self.control.processing_mode.currentIndexChanged.connect(self._push_mode)
        self.tracking.changed.connect(self._push_config)
        self.tracking.algorithm.currentIndexChanged.connect(self._on_algorithm)
        self.tracking.next_target.connect(lambda: self.detector.request_cycle(1))
        self.tracking.prev_target.connect(lambda: self.detector.request_cycle(-1))
        self.tracking.clear_target.connect(self.detector.request_clear_target)
        self.video.clicked.connect(self._on_video_click)
        self.capture.info_ready.connect(self._on_source_info)
        self.capture.error.connect(self._on_error)
        self.capture.ended.connect(self._on_ended)
        self.capture.position.connect(self._on_position)
        self.detector.frame_ready.connect(self._on_frame)
        self.detector.state_ready.connect(self._on_state)
        self.detector.model_ready.connect(self._on_model_ready)
        self.detector.model_failed.connect(self._on_model_failed)
        self.detector.screenshot_saved.connect(self._on_screenshot)
        self.detector.csv_saved.connect(self._on_csv)
        self.detector.export_progress.connect(self._on_export_progress)
        self.detector.export_finished.connect(self._on_export_finished)
        self.detector.export_failed.connect(self._on_error)
        self.recorder.tick.connect(self._on_rec_tick)
        self.recorder.finished_ok.connect(self._on_record_finished)
        self.recorder.failed.connect(self._on_record_failed)
        self._export_dialog = None

    def _gather(self) -> PipelineConfig:
        overlays = {key: box.isChecked() for key, box in self.overlay_boxes.items()}
        return PipelineConfig(
            confidence=float(self.control.confidence.value()),
            nms=float(self.control.nms.value()),
            device=self.control.device.currentText(),
            target_mode=self.tracking.target_mode.currentData(),
            algorithm=self.tracking.algorithm.currentData(),
            dead_zone=float(self.tracking.dead_zone.value()) / 100.0,
            smoothing=self.tracking.smoothing.currentData(),
            pixel_scale=float(self.tracking.pixel_scale.value()),
            trail_length=int(self.tracking.trail_length.value()),
            lost_timeout=float(self.tracking.lost_timeout.value()),
            auto_reacquire=bool(self.tracking.auto_reacquire.currentData()),
            detection_scale=float(self.control.detection_scale.currentData()),
            control_display=self.tracking.control_display.currentData(),
            overlays=overlays,
            show_debug=overlays.get("debug", False),
        )

    def _push_config(self) -> None:
        self.detector.set_config(self._gather())
        self.performance.set_debug_visible(self.overlay_boxes["debug"].isChecked())

    def _push_mode(self) -> None:
        self.capture.post("mode", mode=self.control.processing_mode.currentData())
        self._push_config()

    def _on_overlay(self) -> None:
        self._push_config()

    def _on_algorithm(self) -> None:
        if self.tracking.algorithm.currentData() == "ORIGINAL":
            index = self.tracking.smoothing.findData("ORIGINAL_20")
            if index >= 0:
                self.tracking.smoothing.setCurrentIndex(index)
        self.detector.request_reset()
        self._push_config()

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open drone video",
            str(Path(self._video_path).parent if self._video_path else Path.home()),
            "Video (*.mp4 *.avi *.mov *.mkv)",
        )
        if not path:
            return
        self._video_path = path
        self.detector.request_reset()
        self.capture.post("open_video", path=path)
        self.statusBar().showMessage(f"Opening {Path(path).name}")

    def _open_webcam(self) -> None:
        text = self.control.camera_resolution.currentText()
        width, height = (int(part) for part in text.split("x"))
        index = self.control.camera_index.currentIndex()
        self.detector.request_reset()
        self.capture.post("open_webcam", index=index, width=width, height=height)
        self.statusBar().showMessage(f"Connecting camera {index}…")

    def _open_rtsp(self) -> None:
        url = self.control.rtsp_url.text().strip()
        if not url:
            QMessageBox.warning(self, "RTSP", "Enter an RTSP URL first.")
            return
        self.detector.request_reset()
        self.statusBar().showMessage("Connecting to RTSP…")
        self.capture.post("open_rtsp", url=url)

    def _disconnect(self) -> None:
        self.capture.post("close")
        self.control.file_info.setText("Disconnected")
        self.statusBar().showMessage("Source disconnected")

    def _play(self) -> None:
        self.capture.post("play")
        self.detector.set_playing(True)
        self.statusBar().showMessage("Playing")

    def _pause(self) -> None:
        self.capture.post("pause")
        self.detector.set_playing(False)
        self.statusBar().showMessage("Paused")

    def _stop(self) -> None:
        self.capture.post("stop")
        self.detector.set_playing(False)

    def _restart(self) -> None:
        self.detector.request_reset()
        self.capture.post("restart")
        self.detector.set_playing(True)

    def _seek(self, ratio: float) -> None:
        self.capture.post("seek", ratio=ratio)

    def _hold_slider(self) -> None:
        self._slider_held = True

    def _release_slider(self) -> None:
        self._slider_held = False

    def _on_video_click(self, x: float, y: float) -> None:
        if self.tracking.algorithm.currentData() != "ENHANCED":
            self.statusBar().showMessage("Click target is part of Enhanced tracking")
            return
        if self.tracking.target_mode.currentData() != "CLICK":
            index = self.tracking.target_mode.findData("CLICK")
            self.tracking.target_mode.setCurrentIndex(index)
        self.detector.request_click(x, y)
        log.info("Click target at %.1f, %.1f", x, y)

    def _on_source_info(self, info: dict) -> None:
        self.control.set_source_info(info)
        self._source_fps = float(info.get("fps") or 30.0)
        self.statusBar().showMessage(f"Source ready: {info.get('filename')}")

    def _on_position(self, position: dict) -> None:
        self.control.set_position(position, self._slider_held)

    def _on_frame(self, image) -> None:
        self.video.set_frame(image)
        self._have_frame = True
        self._frame_shape = (image.height(), image.width())
        if not self.recorder.recorder.active:
            self.record_btn.setEnabled(True)

    def _on_state(self, info: dict) -> None:
        self.control.detected.setText(str(info.get("detected_count", 0)))
        self.control.proc_fps.setText(f"{info.get('proc_fps', 0):.1f}")
        self.control.backend.setText(str(info.get("backend") or "—"))
        self.tracking.apply_state(info)
        self.performance.apply_state(info)
        status = info.get("status") or ""
        if status != self._last_status and status:
            self.statusBar().showMessage(f"{info.get('target_label') or 'Target'}  ·  {status}  ·  {info.get('command')}")
            self._last_status = status

    def _on_model_ready(self, status: str, backend: str) -> None:
        self.model_badge.setText(f"MODEL: {status}")
        self.control.backend.setText(backend)
        log.info("Backend | %s", backend)

    def _on_model_failed(self, message: str) -> None:
        self.model_badge.setText("MODEL: ERROR")
        QMessageBox.critical(self, "Model", message)

    def _on_error(self, message: str) -> None:
        log.info("Error | %s", message)
        QMessageBox.warning(self, "Source", message)
        self.statusBar().showMessage(message.splitlines()[0] if message else "Error")

    def _on_ended(self) -> None:
        self.statusBar().showMessage("End of video")

    def _on_screenshot(self, path: str) -> None:
        self.statusBar().showMessage(f"Screenshot saved: {path}")

    def _on_csv(self, path: str, count: int) -> None:
        self.statusBar().showMessage(f"Control CSV saved ({count} rows): {path}")

    def _record(self) -> None:
        if self.recorder.recorder.active or not self._have_frame:
            return
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RECORDINGS_DIR / f"vtol_tracking_{stamp}.mp4"
        height, width = self._frame_shape
        if width < 2 or height < 2:
            QMessageBox.warning(self, "Record", "There is no processed frame to record yet.")
            return
        self.record_btn.setEnabled(False)
        self.rec_label.setText("● REC 00:15")
        log.info("Recording started | %s", path.name)
        self.recorder.request_start(path, self._source_fps or 30.0, (height, width))
        self.statusBar().showMessage("Recording 15 seconds of the processed view")

    def _on_rec_tick(self, remaining: float) -> None:
        self._paint_rec(remaining)

    def _refresh_rec_label(self) -> None:
        if self.recorder.recorder.active:
            self._paint_rec(self.recorder.recorder.remaining)

    def _paint_rec(self, remaining: float) -> None:
        shown = int(round(max(0.0, min(15.0, remaining))))
        self.rec_label.setText(f"● REC 00:{shown:02d}")

    def _on_record_finished(self, path: str) -> None:
        self.rec_label.setText("● REC 00:00")
        self.record_btn.setEnabled(self._have_frame)
        log.info("Recording completed | %s", path)
        self.statusBar().showMessage(f"Recording saved: {path}")
        QTimer.singleShot(1200, lambda: self.rec_label.setText("") if not self.recorder.recorder.active else None)

    def _on_record_failed(self, message: str) -> None:
        self.record_btn.setEnabled(self._have_frame)
        self.rec_label.setText("")
        log.info("Recording failed | %s", message)
        QMessageBox.warning(self, "Record", message)

    def _export_full(self) -> None:
        if not self._video_path:
            QMessageBox.information(self, "Export", "Open a video file first. Export runs that file offline.")
            return
        self._pause()
        self._export_dialog = QProgressDialog("Exporting annotated video…", "Hide", 0, 100, self)
        self._export_dialog.setWindowTitle("Export full video")
        self._export_dialog.setMinimumDuration(0)
        self._export_dialog.setValue(0)
        self.detector.request_export(self._video_path)

    def _on_export_progress(self, index: int, total: int) -> None:
        if self._export_dialog is None:
            return
        if total > 0:
            self._export_dialog.setMaximum(total)
            self._export_dialog.setValue(min(index, total))
        else:
            self._export_dialog.setMaximum(0)

    def _on_export_finished(self, path: str) -> None:
        if self._export_dialog is not None:
            self._export_dialog.close()
            self._export_dialog = None
        self.statusBar().showMessage(f"Full video exported: {path}")
        QMessageBox.information(self, "Export", f"Annotated video saved:\n{path}")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.load_from(self)
        if dialog.exec():
            dialog.apply_to(self)
            self._push_mode()

    def _load_settings(self) -> None:
        if not CONFIG_PATH.is_file():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.info("Settings load failed: %s", exc)
            return
        self._set_combo_data(self.control.source_kind, data.get("source_kind", "video"))
        self._video_path = data.get("video_path") or ""
        self.control.camera_index.setCurrentIndex(int(data.get("camera_index") or 0))
        resolution = data.get("camera_resolution") or "1280x720"
        if self.control.camera_resolution.findText(resolution) >= 0:
            self.control.camera_resolution.setCurrentText(resolution)
        self.control.rtsp_url.setText(data.get("rtsp_url") or "")
        device = data.get("device") or "AUTO"
        if self.control.device.findText(device) >= 0:
            self.control.device.setCurrentText(device)
        self.control.confidence.setValue(float(data.get("confidence", 0.2)))
        self.control.nms.setValue(float(data.get("nms", 0.4)))
        self._set_combo_data(self.tracking.target_mode, data.get("target_mode", "AUTO_CENTER"))
        self._set_combo_data(self.tracking.algorithm, data.get("algorithm", "ENHANCED"))
        self.tracking.dead_zone.setValue(int(round(float(data.get("dead_zone", 0.2)) * 100)))
        self._set_combo_data(self.tracking.smoothing, data.get("smoothing", "ORIGINAL_20"))
        self._set_combo_data(self.tracking.control_display, data.get("control_display", "DIRECTIONAL"))
        self.tracking.pixel_scale.setValue(float(data.get("pixel_scale", 0.1)))
        self.tracking.trail_length.setValue(int(data.get("trail_length", 60)))
        self.tracking.lost_timeout.setValue(float(data.get("lost_timeout", 1.5)))
        self._set_combo_data(self.tracking.auto_reacquire, bool(data.get("auto_reacquire", True)))
        self._set_combo_data(self.control.detection_scale, float(data.get("detection_scale", 1.0)))
        self._set_combo_data(self.control.processing_mode, data.get("processing_mode", "REALTIME"))
        overlays = data.get("overlays") or {}
        for key, box in self.overlay_boxes.items():
            if key in overlays:
                box.setChecked(bool(overlays[key]))
        geometry = data.get("geometry") or ""
        if geometry:
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
        self.capture.post("mode", mode=self.control.processing_mode.currentData())

    def _save_settings(self) -> None:
        ensure_dirs()
        data = {
            "source_kind": self.control.source_kind.currentData(),
            "video_path": self._video_path,
            "camera_index": self.control.camera_index.currentIndex(),
            "camera_resolution": self.control.camera_resolution.currentText(),
            "rtsp_url": self.control.rtsp_url.text().strip(),
            "device": self.control.device.currentText(),
            "confidence": self.control.confidence.value(),
            "nms": self.control.nms.value(),
            "target_mode": self.tracking.target_mode.currentData(),
            "algorithm": self.tracking.algorithm.currentData(),
            "dead_zone": self.tracking.dead_zone.value() / 100.0,
            "smoothing": self.tracking.smoothing.currentData(),
            "control_display": self.tracking.control_display.currentData(),
            "pixel_scale": self.tracking.pixel_scale.value(),
            "trail_length": self.tracking.trail_length.value(),
            "lost_timeout": self.tracking.lost_timeout.value(),
            "auto_reacquire": bool(self.tracking.auto_reacquire.currentData()),
            "detection_scale": float(self.control.detection_scale.currentData()),
            "processing_mode": self.control.processing_mode.currentData(),
            "overlays": {key: box.isChecked() for key, box in self.overlay_boxes.items()},
            "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _set_combo_data(combo, value) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        log.info("Application Shutdown")
        self.capture.stop()
        self.detector.stop()
        self.recorder.stop()
        self.hub.ack(10**12)
        self.capture.wait(2000)
        self.detector.wait(2000)
        self.recorder.wait(2000)
        event.accept()
