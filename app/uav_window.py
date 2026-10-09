"""Main window for UAV Vision Detection & Tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.i18n import class_text, tr
from app.video_widget import VideoWidget
from capture.source_manager import CaptureWorker, FrameHub
from detection.detector_worker import DetectionWorker
from integration.command_center import CommandCenterClient, events_from_tracks
from recording.recorder import RecordingWorker
from tracking.engine import PipelineConfig
from utils.logger import get_logger
from utils.paths import CONFIG_PATH, DATASET_DIR, DEMO_SOURCE_DIR, EVENTS_DIR, OUTPUT_DIR, PREVIEW_DIR, RECORDINGS_DIR, ensure_dirs


log = get_logger()

STANDBY_FRAMES = (
    PREVIEW_DIR / "preview_drone.jpg",
    PREVIEW_DIR / "preview_aircraft.jpg",
    PREVIEW_DIR / "preview_helicopter.jpg",
)
STANDBY_INTERVAL_MS = 3000


DEMO_SCENES = (
    ("uav_tracking", "Сопровождение БПЛА", "UAV tracking"),
    ("fpv_drone", "FPV-дрон", "FPV drone"),
    ("multiple_drones", "Несколько БПЛА", "Multiple drones"),
    ("bird_vs_drone", "Птица и БПЛА", "Bird vs drone"),
)


class _Job(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._fn())
        except Exception as exc:
            self.failed.emit(str(exc))


class UavWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        self.lang = "ru"
        self.mission = "uav"
        self.last_info: dict = {}
        self._video_path = ""
        self._slider_held = False
        self._have_frame = False
        self._standby = True
        self._standby_images: list[QImage] = []
        self._standby_index = 0
        self._standby_elapsed = 0
        self._source_fps = 25.0
        self._frame_shape = (0, 0)
        self._job: _Job | None = None
        self.hub = FrameHub()
        self.capture = CaptureWorker(self.hub)
        self.recorder = RecordingWorker()
        self.detector = DetectionWorker(self.hub, self.recorder)
        self._build()
        self._wire()
        self._load_settings()
        self._retranslate()
        self._load_standby()
        self._push()
        self.capture.start()
        self.recorder.start()
        self.detector.start()
        self._clock = QTimer(self)
        self._clock.setInterval(100)
        self._clock.timeout.connect(self._refresh_rec)
        self._clock.start()

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self._header())
        self.video = VideoWidget()
        frame = QFrame()
        frame.setObjectName("VideoFrame")
        video_layout = QVBoxLayout(frame)
        video_layout.setContentsMargins(6, 6, 6, 6)
        video_layout.addWidget(self.video)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._live_tab(), "")
        self.tabs.addTab(self._uav_tab(), "")
        self.tabs.addTab(self._vehicle_tab(), "")
        self.tabs.addTab(self._targets_tab(), "")
        self.tabs.addTab(self._track_tab(), "")
        self.tabs.addTab(self._model_tab(), "")
        self.tabs.addTab(self._perf_tab(), "")
        self.tabs.addTab(self._record_tab(), "")
        self.tabs.addTab(self._export_tab(), "")
        self.tabs.addTab(self._settings_tab(), "")
        self.tabs.setMinimumWidth(420)
        self.tabs.setMaximumWidth(520)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(frame)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._overlays())
        self.resize(1480, 920)
        self.statusBar().showMessage("PTZ virtual")

    def _header(self) -> QFrame:
        box = QFrame()
        box.setObjectName("Header")
        row = QHBoxLayout(box)
        text = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("Title")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("Subtitle")
        self.subtitle.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.subtitle)
        row.addLayout(text, 1)
        self.mode_badge = QLabel()
        self.mode_badge.setObjectName("BadgeVirtual")
        self.demo_badge = QLabel("DEMO / SIMULATED DATA")
        self.demo_badge.setObjectName("BadgeDemo")
        self.demo_badge.hide()
        self.ptz_badge = QLabel()
        self.ptz_badge.setObjectName("BadgeVirtual")
        self.model_badge = QLabel()
        self.model_badge.setObjectName("BadgeModel")
        for badge in (self.mode_badge, self.demo_badge, self.ptz_badge, self.model_badge):
            row.addWidget(badge)
        return box

    def _wrap(self, inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _live_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        self.source_kind = QComboBox()
        self.video_btn = QPushButton()
        self.video_btn.setObjectName("Primary")
        self.camera_index = QComboBox()
        self.camera_index.addItems(["0", "1", "2"])
        self.camera_resolution = QComboBox()
        self.camera_resolution.addItems(["640x480", "1280x720", "1920x1080"])
        self.camera_resolution.setCurrentText("1280x720")
        self.webcam_btn = QPushButton()
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("rtsp://host:554/stream")
        self.rtsp_btn = QPushButton()
        self.disconnect_btn = QPushButton()
        self.play_btn = QPushButton()
        self.pause_btn = QPushButton()
        self.stop_btn = QPushButton()
        self.restart_btn = QPushButton()
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        self.time_label = QLabel("00:00 / 00:00")
        self.file_info = QLabel()
        self.file_info.setWordWrap(True)
        self.processing_mode = QComboBox()
        buttons = QHBoxLayout()
        for button in (self.play_btn, self.pause_btn, self.stop_btn, self.restart_btn):
            buttons.addWidget(button)
        root.addWidget(self.source_kind)
        root.addWidget(self.video_btn)
        cam = QFormLayout()
        self.camera_label = QLabel()
        self.resolution_label = QLabel()
        cam.addRow(self.camera_label, self.camera_index)
        cam.addRow(self.resolution_label, self.camera_resolution)
        root.addLayout(cam)
        root.addWidget(self.webcam_btn)
        root.addWidget(self.rtsp_url)
        pair = QHBoxLayout()
        pair.addWidget(self.rtsp_btn)
        pair.addWidget(self.disconnect_btn)
        root.addLayout(pair)
        root.addLayout(buttons)
        root.addWidget(self.timeline)
        root.addWidget(self.time_label)
        self.processing_label = QLabel()
        root.addWidget(self.processing_label)
        root.addWidget(self.processing_mode)
        root.addWidget(self.file_info)
        root.addStretch(1)
        return self._wrap(page)

    def _uav_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.uav_hint = QLabel()
        self.uav_hint.setWordWrap(True)
        self.weights = QLineEdit()
        self.weights_btn = QPushButton()
        weight_row = QHBoxLayout()
        weight_row.addWidget(self.weights)
        weight_row.addWidget(self.weights_btn)
        self.uav_conf = self._double(0.05, 0.95, 0.35, 0.05)
        self.uav_nms = self._double(0.05, 0.95, 0.45, 0.05)
        self.uav_device = QComboBox()
        self.uav_device.addItems(["AUTO", "CUDA", "CPU"])
        self.uav_imgsz = QComboBox()
        for size in (320, 416, 640, 960, 1280):
            self.uav_imgsz.addItem(str(size), size)
        self.uav_imgsz.setCurrentIndex(2)
        self.demo_check = QCheckBox()
        self.demo_hint = QLabel()
        self.demo_hint.setWordWrap(True)
        self.demo_scene = QComboBox()
        self.demo_btn = QPushButton()
        self.demo_btn.setObjectName("Primary")
        self.uav_mode_btn = QPushButton()
        self.uav_mode_btn.setObjectName("Primary")
        form.addRow(self.uav_hint)
        self.weights_label = QLabel()
        form.addRow(self.weights_label, weight_row)
        self.download_weights_btn = QPushButton()
        self.download_weights_btn.setObjectName("Primary")
        self.library_status = QLabel()
        self.library_status.setWordWrap(True)
        form.addRow(self.download_weights_btn)
        form.addRow(self.library_status)
        self.conf_label_uav = QLabel()
        self.nms_label_uav = QLabel()
        self.device_label_uav = QLabel()
        self.imgsz_label = QLabel()
        form.addRow(self.conf_label_uav, self.uav_conf)
        form.addRow(self.nms_label_uav, self.uav_nms)
        form.addRow(self.device_label_uav, self.uav_device)
        form.addRow(self.imgsz_label, self.uav_imgsz)
        form.addRow(self.demo_check)
        form.addRow(self.demo_hint)
        self.scene_label = QLabel()
        form.addRow(self.scene_label, self.demo_scene)
        form.addRow(self.demo_btn)
        form.addRow(self.uav_mode_btn)
        return self._wrap(page)

    def _vehicle_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.vehicle_hint = QLabel()
        self.vehicle_hint.setWordWrap(True)
        self.virtual_note = QLabel()
        self.virtual_note.setWordWrap(True)
        self.veh_conf = self._double(0.05, 0.95, 0.20, 0.05)
        self.veh_nms = self._double(0.05, 0.95, 0.40, 0.05)
        self.veh_device = QComboBox()
        self.veh_device.addItems(["AUTO", "CUDA", "CPU"])
        self.veh_scale = QComboBox()
        self.veh_scale.addItem("100%", 1.0)
        self.veh_scale.addItem("75%", 0.75)
        self.veh_scale.addItem("50%", 0.5)
        self.veh_algo = QComboBox()
        self.veh_mode_btn = QPushButton()
        self.veh_mode_btn.setObjectName("Primary")
        form.addRow(self.vehicle_hint)
        form.addRow(self.virtual_note)
        self.conf_label_veh = QLabel()
        self.nms_label_veh = QLabel()
        self.device_label_veh = QLabel()
        self.scale_label = QLabel()
        self.algo_label = QLabel()
        form.addRow(self.conf_label_veh, self.veh_conf)
        form.addRow(self.nms_label_veh, self.veh_nms)
        form.addRow(self.device_label_veh, self.veh_device)
        form.addRow(self.scale_label, self.veh_scale)
        form.addRow(self.algo_label, self.veh_algo)
        form.addRow(self.veh_mode_btn)
        return self._wrap(page)

    def _targets_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.table = QTableWidget(0, 6)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.target_readout = QLabel()
        self.target_readout.setWordWrap(True)
        self.target_readout.setObjectName("ReadoutVal")
        layout.addWidget(self.target_readout)
        layout.addWidget(self.table, 1)
        return page

    def _track_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.target_mode = QComboBox()
        self.uav_tracker = QComboBox()
        self.lost_timeout = self._double(0.3, 5.0, 1.5, 0.1)
        self.trail_length = QSpinBox()
        self.trail_length.setRange(10, 120)
        self.trail_length.setValue(60)
        self.low_conf = self._double(0.01, 0.5, 0.10, 0.01)
        self.match_iou = self._double(0.05, 0.9, 0.30, 0.05)
        self.meters = self._double(0.0, 5.0, 0.0, 0.001, 4)
        self.mpp_hint = QLabel()
        self.mpp_hint.setWordWrap(True)
        self.next_btn = QPushButton()
        self.prev_btn = QPushButton()
        self.clear_btn = QPushButton()
        row = QHBoxLayout()
        row.addWidget(self.prev_btn)
        row.addWidget(self.next_btn)
        row.addWidget(self.clear_btn)
        self.ptz_slew = QCheckBox()
        self.ptz_note = QLabel()
        self.ptz_note.setWordWrap(True)
        self.ptz_stop_btn = QPushButton()
        self.ptz_home_btn = QPushButton()
        self.pan_min = self._double(-180, 0, -90, 1, 1)
        self.pan_max = self._double(0, 180, 90, 1, 1)
        self.tilt_min = self._double(-90, 0, -30, 1, 1)
        self.tilt_max = self._double(0, 90, 45, 1, 1)
        self.ptz_deg = self._double(1, 90, 35, 1, 1)
        self.ptz_readout = QLabel("pan 0.0   tilt 0.0")
        self.ptz_readout.setObjectName("ReadoutVal")
        self.mode_label = QLabel()
        self.tracker_label = QLabel()
        self.lost_label = QLabel()
        self.trail_label = QLabel()
        self.low_label = QLabel()
        self.iou_label = QLabel()
        self.mpp_label = QLabel()
        self.pan_min_label = QLabel()
        self.pan_max_label = QLabel()
        self.tilt_min_label = QLabel()
        self.tilt_max_label = QLabel()
        self.deg_label = QLabel()
        form.addRow(self.mode_label, self.target_mode)
        form.addRow(self.tracker_label, self.uav_tracker)
        form.addRow(self.lost_label, self.lost_timeout)
        form.addRow(self.trail_label, self.trail_length)
        form.addRow(self.low_label, self.low_conf)
        form.addRow(self.iou_label, self.match_iou)
        form.addRow(self.mpp_label, self.meters)
        form.addRow(self.mpp_hint)
        form.addRow(row)
        form.addRow(self.ptz_slew)
        form.addRow(self.ptz_note)
        form.addRow(self.ptz_stop_btn)
        form.addRow(self.ptz_home_btn)
        form.addRow(self.pan_min_label, self.pan_min)
        form.addRow(self.pan_max_label, self.pan_max)
        form.addRow(self.tilt_min_label, self.tilt_min)
        form.addRow(self.tilt_max_label, self.tilt_max)
        form.addRow(self.deg_label, self.ptz_deg)
        form.addRow(self.ptz_readout)
        return self._wrap(page)

    def _model_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.download_weights_model_btn = QPushButton()
        self.download_weights_model_btn.setObjectName("Primary")
        layout.addWidget(self.download_weights_model_btn)
        self.import_images_btn = QPushButton()
        self.import_video_btn = QPushButton()
        self.import_labels_btn = QPushButton()
        self.every_n = QSpinBox()
        self.every_n.setRange(1, 120)
        self.every_n.setValue(15)
        self.val_ratio = self._double(0.05, 0.5, 0.2, 0.05)
        self.split_btn = QPushButton()
        self.base_weights = QLineEdit()
        self.base_btn = QPushButton()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 300)
        self.epochs.setValue(20)
        self.train_btn = QPushButton()
        self.train_btn.setObjectName("Primary")
        self.results_dir = QLineEdit()
        self.metrics_btn = QPushButton()
        self.metric_text = QLabel()
        self.metric_text.setWordWrap(True)
        self.metric_text.setObjectName("ReadoutVal")
        self.matrix = QLabel()
        self.matrix.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conditions = QLabel()
        self.conditions.setWordWrap(True)
        self.build_scenes_btn = QPushButton()
        self.test_video_btn = QPushButton()
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMinimumHeight(120)
        for widget in (
            self.import_images_btn, self.import_video_btn, self.import_labels_btn,
            self.split_btn, self.train_btn, self.metrics_btn, self.build_scenes_btn, self.test_video_btn,
        ):
            layout.addWidget(widget)
        form = QFormLayout()
        self.every_label = QLabel()
        self.val_label = QLabel()
        self.base_label = QLabel()
        self.epoch_label = QLabel()
        self.results_label = QLabel()
        base_row = QHBoxLayout()
        base_row.addWidget(self.base_weights)
        base_row.addWidget(self.base_btn)
        form.addRow(self.every_label, self.every_n)
        form.addRow(self.val_label, self.val_ratio)
        form.addRow(self.base_label, base_row)
        form.addRow(self.epoch_label, self.epochs)
        form.addRow(self.results_label, self.results_dir)
        layout.addLayout(form)
        layout.addWidget(self.metric_text)
        layout.addWidget(self.matrix)
        layout.addWidget(self.conditions)
        layout.addWidget(self.train_log, 1)
        return self._wrap(page)

    def _perf_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.perf: dict[str, QLabel] = {}
        self.perf_labels: dict[str, QLabel] = {}
        for key in ("source_fps", "detected", "det_ms", "track_ms", "render_ms", "backend", "status", "dropped"):
            label = QLabel("—")
            label.setObjectName("ReadoutVal")
            title = QLabel()
            self.perf[key] = label
            self.perf_labels[key] = title
            form.addRow(title, label)
        return self._wrap(page)

    def _record_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.record_seconds = QSpinBox()
        self.record_seconds.setRange(15, 30)
        self.record_seconds.setValue(15)
        self.record_btn = QPushButton()
        self.record_btn.setObjectName("Record")
        self.record_btn.setEnabled(False)
        self.rec_label = QLabel("")
        self.rec_label.setObjectName("RecBadge")
        self.shot_btn = QPushButton()
        self.record_caption = QLabel()
        layout.addWidget(self.record_caption)
        layout.addWidget(self.record_seconds)
        layout.addWidget(self.rec_label)
        layout.addWidget(self.record_btn)
        layout.addWidget(self.shot_btn)
        layout.addStretch(1)
        return self._wrap(page)

    def _export_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.no_gps = QLabel()
        self.no_gps.setWordWrap(True)
        self.sensor_id = QLineEdit("uav-vision-01")
        self.event_host = QLineEdit("127.0.0.1")
        self.event_port = QSpinBox()
        self.event_port.setRange(1, 65535)
        self.event_port.setValue(5001)
        self.live_events = QCheckBox()
        self.csv_btn = QPushButton()
        self.json_btn = QPushButton()
        self.test_event_btn = QPushButton()
        self.export_btn = QPushButton()
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        form = QFormLayout()
        self.sensor_label = QLabel()
        self.host_label = QLabel()
        self.port_label = QLabel()
        form.addRow(self.sensor_label, self.sensor_id)
        form.addRow(self.host_label, self.event_host)
        form.addRow(self.port_label, self.event_port)
        for widget in (self.no_gps,):
            layout.addWidget(widget)
        layout.addLayout(form)
        for widget in (self.live_events, self.csv_btn, self.json_btn, self.test_event_btn, self.export_btn, self.event_log):
            layout.addWidget(widget)
        return self._wrap(page)

    def _settings_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.language = QComboBox()
        self.language.addItem("Русский", "ru")
        self.language.addItem("English", "en")
        self.dead_zone = QSpinBox()
        self.dead_zone.setRange(5, 40)
        self.dead_zone.setValue(20)
        self.smoothing = QComboBox()
        self.pixel_scale = self._double(0.01, 2.0, 0.1, 0.05, 3)
        self.auto_reacquire = QComboBox()
        self.lang_label = QLabel()
        self.dead_label = QLabel()
        self.smooth_label = QLabel()
        self.pixel_label = QLabel()
        self.reacquire_label = QLabel()
        form.addRow(self.lang_label, self.language)
        form.addRow(self.dead_label, self.dead_zone)
        form.addRow(self.smooth_label, self.smoothing)
        form.addRow(self.pixel_label, self.pixel_scale)
        form.addRow(self.reacquire_label, self.auto_reacquire)
        return self._wrap(page)

    def _overlays(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("BottomBar")
        row = QHBoxLayout(bar)
        self.overlay_boxes: dict[str, QCheckBox] = {}
        for key in ("boxes", "confidence", "ids", "trail", "status", "speed", "fps", "crosshair"):
            box = QCheckBox()
            box.setChecked(key != "crosshair")
            box.toggled.connect(self._push)
            self.overlay_boxes[key] = box
            row.addWidget(box)
        return bar

    def _wire(self) -> None:
        self.video_btn.clicked.connect(self._open_video)
        self.webcam_btn.clicked.connect(self._open_webcam)
        self.rtsp_btn.clicked.connect(self._open_rtsp)
        self.disconnect_btn.clicked.connect(self._disconnect)
        self.play_btn.clicked.connect(self._play)
        self.pause_btn.clicked.connect(self._pause)
        self.stop_btn.clicked.connect(lambda: self.capture.post("stop"))
        self.restart_btn.clicked.connect(self._restart)
        self.timeline.sliderPressed.connect(lambda: setattr(self, "_slider_held", True))
        self.timeline.sliderReleased.connect(self._seek)
        self.processing_mode.currentIndexChanged.connect(self._push_mode)
        self.video.clicked.connect(self._click)
        self.capture.info_ready.connect(self._on_source)
        self.capture.error.connect(self._on_error)
        self.capture.ended.connect(lambda: self.statusBar().showMessage("End"))
        self.capture.position.connect(self._on_position)
        self.detector.frame_ready.connect(self._on_frame)
        self.detector.state_ready.connect(self._on_state)
        self.detector.model_ready.connect(self._on_model)
        self.detector.model_failed.connect(lambda message: self.statusBar().showMessage(message))
        self.detector.screenshot_saved.connect(lambda path: self.statusBar().showMessage(path))
        self.detector.csv_saved.connect(lambda path, count: self.statusBar().showMessage(f"{path} ({count})"))
        self.detector.export_progress.connect(self._on_export_progress)
        self.detector.export_finished.connect(self._on_export_finished)
        self.detector.export_failed.connect(self._on_error)
        self.recorder.finished_ok.connect(self._on_record_finished)
        self.recorder.failed.connect(self._on_error)
        self.weights_btn.clicked.connect(lambda: self._browse(self.weights, "Weights (*.pt *.onnx)"))
        self.download_weights_btn.clicked.connect(self._download_weights)
        self.download_weights_model_btn.clicked.connect(self._download_weights)
        self.base_btn.clicked.connect(lambda: self._browse(self.base_weights, "Weights (*.pt)"))
        self.demo_btn.clicked.connect(self._open_demo)
        self.uav_mode_btn.clicked.connect(lambda: self._set_mission("uav"))
        self.veh_mode_btn.clicked.connect(lambda: self._set_mission("vehicle"))
        self.next_btn.clicked.connect(lambda: self.detector.request_cycle(1))
        self.prev_btn.clicked.connect(lambda: self.detector.request_cycle(-1))
        self.clear_btn.clicked.connect(self.detector.request_clear_target)
        self.ptz_stop_btn.clicked.connect(lambda: self.detector.request_ptz("stop"))
        self.ptz_home_btn.clicked.connect(lambda: self.detector.request_ptz("home"))
        self.ptz_slew.toggled.connect(self._slew_toggled)
        self.record_btn.clicked.connect(self._record)
        self.shot_btn.clicked.connect(self.detector.request_screenshot)
        self.csv_btn.clicked.connect(self.detector.request_csv)
        self.json_btn.clicked.connect(self._save_json)
        self.test_event_btn.clicked.connect(self._send_test)
        self.export_btn.clicked.connect(self._export_full)
        self.import_images_btn.clicked.connect(self._import_images)
        self.import_video_btn.clicked.connect(self._import_video)
        self.import_labels_btn.clicked.connect(self._import_labels)
        self.split_btn.clicked.connect(self._split)
        self.train_btn.clicked.connect(self._train)
        self.metrics_btn.clicked.connect(self._load_metrics)
        self.build_scenes_btn.clicked.connect(self._build_scenes)
        self.test_video_btn.clicked.connect(self._test_video)
        self.language.currentIndexChanged.connect(self._on_language)
        for widget in (
            self.uav_conf, self.uav_nms, self.veh_conf, self.veh_nms, self.lost_timeout,
            self.low_conf, self.match_iou, self.meters, self.trail_length, self.dead_zone, self.pixel_scale,
            self.pan_min, self.pan_max, self.tilt_min, self.tilt_max, self.ptz_deg, self.record_seconds,
        ):
            widget.valueChanged.connect(self._push)
        for widget in (
            self.uav_device, self.uav_imgsz, self.veh_device, self.veh_scale, self.veh_algo,
            self.target_mode, self.uav_tracker, self.demo_scene, self.smoothing, self.auto_reacquire, self.demo_check,
        ):
            widget.currentIndexChanged.connect(self._push) if isinstance(widget, QComboBox) else widget.toggled.connect(self._push)
        self.live_events.toggled.connect(self._push)
        self.sensor_id.editingFinished.connect(self._push)
        self.event_host.editingFinished.connect(self._push)
        self.event_port.valueChanged.connect(self._push)

    def _on_language(self) -> None:
        self.lang = self.language.currentData() or "ru"
        self._retranslate()
        self._push()

    def _set_mission(self, mission: str) -> None:
        self.mission = mission
        if mission == "vehicle":
            self.demo_check.setChecked(False)
        self.detector.request_reset()
        self._push()
        self._retranslate()

    def _slew_toggled(self, checked: bool) -> None:
        if checked:
            self.detector.request_ptz("release")
        self._push()

    def _gather(self) -> PipelineConfig:
        uav = self.mission == "uav"
        overlays = {
            "boxes": self.overlay_boxes["boxes"].isChecked(),
            "confidence": self.overlay_boxes["confidence"].isChecked(),
            "ids": self.overlay_boxes["ids"].isChecked(),
            "trail": self.overlay_boxes["trail"].isChecked(),
            "status": self.overlay_boxes["status"].isChecked(),
            "speed": self.overlay_boxes["speed"].isChecked(),
            "fps": self.overlay_boxes["fps"].isChecked(),
            "crosshair": self.overlay_boxes["crosshair"].isChecked(),
            "dead_zone": True,
            "flight_director": not uav,
            "command_text": not uav,
            "target_vector": not uav,
            "debug": False,
        }
        return PipelineConfig(
            confidence=float(self.uav_conf.value() if uav else self.veh_conf.value()),
            nms=float(self.uav_nms.value() if uav else self.veh_nms.value()),
            device=self.uav_device.currentText() if uav else self.veh_device.currentText(),
            target_mode=self.target_mode.currentData() or "AUTO_CENTER",
            algorithm=(self.uav_tracker.currentData() or "BYTETRACK") if uav else (self.veh_algo.currentData() or "ENHANCED"),
            dead_zone=float(self.dead_zone.value()) / 100.0,
            smoothing=self.smoothing.currentData() or "MEDIUM",
            pixel_scale=float(self.pixel_scale.value()),
            trail_length=int(self.trail_length.value()),
            lost_timeout=float(self.lost_timeout.value()),
            auto_reacquire=bool(self.auto_reacquire.currentData()),
            detection_scale=float(self.veh_scale.currentData() or 1.0),
            control_display="DIRECTIONAL",
            overlays=overlays,
            mission=self.mission,
            imgsz=int(self.uav_imgsz.currentData() or 640),
            demo=bool(self.demo_check.isChecked()) and uav,
            demo_scene=self.demo_scene.currentData() or "",
            weights_path=self.weights.text().strip(),
            meters_per_pixel=float(self.meters.value()),
            sensor_id=self.sensor_id.text().strip() or "uav-vision-01",
            emit_events=self.live_events.isChecked(),
            event_host=self.event_host.text().strip() or "127.0.0.1",
            event_port=int(self.event_port.value()),
            event_log=str(EVENTS_DIR / "live.jsonl"),
            low_conf=float(self.low_conf.value()),
            match_iou=float(self.match_iou.value()),
            ptz_slew=self.ptz_slew.isChecked(),
            record_seconds=float(self.record_seconds.value()),
            ptz_pan_min=float(self.pan_min.value()),
            ptz_pan_max=float(self.pan_max.value()),
            ptz_tilt_min=float(self.tilt_min.value()),
            ptz_tilt_max=float(self.tilt_max.value()),
            ptz_deg=float(self.ptz_deg.value()),
            language=self.lang or "ru",
        )

    def _push(self, *_args) -> None:
        if not hasattr(self, "detector"):
            return
        self.detector.set_config(self._gather())

    def _push_mode(self) -> None:
        self.capture.post("mode", mode=self.processing_mode.currentData())
        self._push()

    def _retranslate(self) -> None:
        t = lambda key: tr(self.lang, key)
        self.setWindowTitle("UAV Vision Detection & Tracking")
        self.title.setText(t("title"))
        self.subtitle.setText(t("subtitle"))
        names = (
            "tab_live", "tab_uav", "tab_vehicle", "tab_targets", "tab_track",
            "tab_model", "tab_perf", "tab_record", "tab_export", "tab_settings",
        )
        for index, key in enumerate(names):
            self.tabs.setTabText(index, t(key))
        self._fill_combo(self.source_kind, (("video_file", "video"), ("webcam", "webcam"), ("rtsp", "rtsp")))
        self._fill_combo(self.processing_mode, (("realtime", "REALTIME"), ("every_frame", "EVERY_FRAME")))
        self._fill_combo(self.target_mode, (("auto_center", "AUTO_CENTER"), ("click_target", "CLICK"), ("first_detected", "FIRST")))
        self._fill_combo(self.uav_tracker, (("ByteTrack", "BYTETRACK"), ("BoT-SORT", "BOTSORT")), literal=True)
        self._fill_combo(self.veh_algo, (("ENHANCED", "ENHANCED"), ("ORIGINAL", "ORIGINAL")), literal=True)
        self._fill_combo(self.smoothing, (("OFF", "OFF"), ("LOW", "LOW"), ("MEDIUM", "MEDIUM"), ("HIGH", "HIGH")), literal=True)
        self._fill_combo(self.auto_reacquire, (("yes", True), ("no", False)))
        current_scene = self.demo_scene.currentData()
        self.demo_scene.blockSignals(True)
        self.demo_scene.clear()
        for key, ru, en in DEMO_SCENES:
            self.demo_scene.addItem(ru if self.lang == "ru" else en, key)
        found = self.demo_scene.findData(current_scene)
        if found >= 0:
            self.demo_scene.setCurrentIndex(found)
        self.demo_scene.blockSignals(False)
        self.video_btn.setText(t("open_video"))
        self.webcam_btn.setText(t("connect"))
        self.rtsp_btn.setText(t("connect"))
        self.disconnect_btn.setText(t("disconnect"))
        self.play_btn.setText(t("play"))
        self.pause_btn.setText(t("pause"))
        self.stop_btn.setText(t("stop"))
        self.restart_btn.setText(t("restart"))
        self.camera_label.setText(t("camera"))
        self.resolution_label.setText(t("resolution"))
        self.processing_label.setText(t("processing"))
        self.uav_hint.setText(t("uav_hint"))
        self.vehicle_hint.setText(t("vehicle_hint"))
        self.virtual_note.setText(t("virtual_only"))
        self.weights_label.setText(t("weights"))
        self.weights_btn.setText(t("browse"))
        self.download_weights_btn.setText(t("download_all"))
        self.download_weights_model_btn.setText(t("download_all"))
        self.base_btn.setText(t("browse"))
        self.conf_label_uav.setText(t("confidence"))
        self.nms_label_uav.setText(t("nms"))
        self.device_label_uav.setText(t("device"))
        self.imgsz_label.setText(t("imgsz"))
        self.conf_label_veh.setText(t("confidence"))
        self.nms_label_veh.setText(t("nms"))
        self.device_label_veh.setText(t("device"))
        self.scale_label.setText(t("detection_scale"))
        self.algo_label.setText(t("vehicle_algo"))
        self.demo_check.setText(t("demo"))
        self.demo_hint.setText(t("demo_hint"))
        self.scene_label.setText(t("open_demo_scene"))
        self.demo_btn.setText(t("open_demo"))
        self.uav_mode_btn.setText(t("activate_uav"))
        self.veh_mode_btn.setText(t("activate_vehicle"))
        self.mode_label.setText(t("target_mode"))
        self.tracker_label.setText(t("tracker"))
        self.lost_label.setText(t("lost_timeout"))
        self.trail_label.setText(t("trail"))
        self.low_label.setText(t("low_conf"))
        self.iou_label.setText(t("match_iou"))
        self.mpp_label.setText(t("meters_per_pixel"))
        self.mpp_hint.setText(t("mpp_hint"))
        self.next_btn.setText(t("next"))
        self.prev_btn.setText(t("prev"))
        self.clear_btn.setText(t("clear"))
        self.ptz_slew.setText(t("ptz_slew"))
        self.ptz_note.setText(t("ptz_note"))
        self.ptz_stop_btn.setText(t("ptz_stop"))
        self.ptz_home_btn.setText(t("ptz_home"))
        self.pan_min_label.setText(t("pan_min"))
        self.pan_max_label.setText(t("pan_max"))
        self.tilt_min_label.setText(t("tilt_min"))
        self.tilt_max_label.setText(t("tilt_max"))
        self.deg_label.setText(t("deg_per_norm"))
        self.import_images_btn.setText(t("import_images"))
        self.import_video_btn.setText(t("import_video"))
        self.import_labels_btn.setText(t("import_labels"))
        self.split_btn.setText(t("split"))
        self.train_btn.setText(t("start_train"))
        self.metrics_btn.setText(t("load_metrics"))
        self.build_scenes_btn.setText(t("build_scenes"))
        self.test_video_btn.setText(t("test_video"))
        self.every_label.setText(t("every_n"))
        self.val_label.setText(t("val_ratio"))
        self.base_label.setText(t("base_weights"))
        self.epoch_label.setText(t("epochs"))
        self.results_label.setText(t("results_dir"))
        self.conditions.setText(t("conditions"))
        if not self.metric_text.text() or self.metric_text.text() in (tr("ru", "metrics_empty"), tr("en", "metrics_empty")):
            self.metric_text.setText(t("metrics_empty"))
        self.record_caption.setText(t("record_seconds"))
        self.record_btn.setText(t("record"))
        self.shot_btn.setText(t("screenshot"))
        self.no_gps.setText(t("no_gps"))
        self.sensor_label.setText(t("sensor_id"))
        self.host_label.setText(t("host"))
        self.port_label.setText(t("port"))
        self.live_events.setText(t("live_events"))
        self.csv_btn.setText(t("export_csv"))
        self.json_btn.setText(t("export_json"))
        self.test_event_btn.setText(t("send_test"))
        self.export_btn.setText(t("export_video"))
        self.lang_label.setText(t("language"))
        self.dead_label.setText(t("dead_zone"))
        self.smooth_label.setText(t("smoothing"))
        self.pixel_label.setText(t("pixel_scale"))
        self.reacquire_label.setText(t("auto_reacquire"))
        self.mode_badge.setText(t("mode_uav") if self.mission == "uav" else t("mode_vehicle"))
        self.ptz_badge.setText(t("virtual_badge"))
        self.demo_badge.setText(t("demo_badge"))
        headers = [t("col_id"), t("col_class"), t("col_conf"), t("col_status"), t("col_px"), t("col_kmh")]
        self.table.setHorizontalHeaderLabels(headers)
        overlay_keys = {
            "boxes": "overlay_boxes", "confidence": "overlay_conf", "ids": "overlay_ids",
            "trail": "overlay_trail", "status": "overlay_status", "speed": "overlay_speed",
            "fps": "overlay_fps", "crosshair": "overlay_cross",
        }
        for key, name in overlay_keys.items():
            self.overlay_boxes[key].setText(t(name))
        for key, name in (
            ("source_fps", "source_fps"), ("detected", "detected"), ("det_ms", "det_ms"),
            ("track_ms", "track_ms"), ("render_ms", "render_ms"), ("backend", "backend"),
            ("status", "status"), ("dropped", "dropped"),
        ):
            self.perf_labels[key].setText(t(name))
        if self.video._image is None:
            self.video.setText(t("no_source"))

    def _fill_combo(self, combo: QComboBox, pairs, literal: bool = False) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, data in pairs:
            combo.addItem(label if literal else tr(self.lang, label), data)
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr(self.lang, "open_video"), str(Path(self._video_path).parent if self._video_path else Path.home()), "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._start_video(path)

    def _start_video(self, path: str) -> None:
        self._video_path = path
        self.detector.request_reset()
        self._push()
        self.capture.post("open_video", path=path)
        self.statusBar().showMessage(Path(path).name)

    def _open_demo(self) -> None:
        key = self.demo_scene.currentData() or "uav_tracking"
        path = DEMO_SOURCE_DIR / f"demo_{key}.mp4"
        if not path.is_file():
            self.statusBar().showMessage(tr(self.lang, "generating"))
            from PySide6.QtWidgets import QApplication
            from simulation.scenarios import write_scene

            QApplication.processEvents()
            write_scene(path, key, seconds=20, fps=25)
        self.mission = "uav"
        self.demo_check.setChecked(True)
        self._retranslate()
        self._start_video(str(path))
        self.capture.post("play")
        self.detector.set_playing(True)

    def _load_standby(self) -> None:
        images = []
        for path in STANDBY_FRAMES:
            image = QImage(str(path))
            if not image.isNull():
                images.append(image)
        self._standby_images = images
        self._standby_index = 0
        self._standby_elapsed = 0
        if images:
            self._standby = True
            self.video.set_frame(images[0])

    def _disconnect(self) -> None:
        self.capture.post("close")
        self.detector.set_playing(False)
        self._have_frame = False
        self.record_btn.setEnabled(False)
        self._standby = True
        self._standby_elapsed = 0
        if self._standby_images:
            self.video.set_frame(self._standby_images[self._standby_index % len(self._standby_images)])

    def _open_webcam(self) -> None:
        width, height = (int(part) for part in self.camera_resolution.currentText().split("x"))
        self.detector.request_reset()
        self.capture.post("open_webcam", index=self.camera_index.currentIndex(), width=width, height=height)

    def _open_rtsp(self) -> None:
        url = self.rtsp_url.text().strip()
        if not url:
            QMessageBox.warning(self, "RTSP", "RTSP URL")
            return
        self.detector.request_reset()
        self.capture.post("open_rtsp", url=url)

    def _play(self) -> None:
        self.capture.post("play")
        self.detector.set_playing(True)

    def _pause(self) -> None:
        self.capture.post("pause")
        self.detector.set_playing(False)

    def _restart(self) -> None:
        self.detector.request_reset()
        self.capture.post("restart")
        self.detector.set_playing(True)

    def _seek(self) -> None:
        self._slider_held = False
        self.capture.post("seek", ratio=self.timeline.value() / 1000.0)

    def _click(self, x: float, y: float) -> None:
        if self.target_mode.currentData() != "CLICK":
            index = self.target_mode.findData("CLICK")
            if index >= 0:
                self.target_mode.setCurrentIndex(index)
        self.detector.request_click(x, y)

    def _on_source(self, info: dict) -> None:
        self._source_fps = float(info.get("fps") or 25.0)
        name = info.get("filename") or ""
        self.file_info.setText(f"{name}\n{info.get('width')}×{info.get('height')}  {self._source_fps:.1f} FPS")

    def _on_position(self, position: dict) -> None:
        duration = float(position.get("duration") or 0.0)
        current = float(position.get("time") or 0.0)
        self.time_label.setText(f"{_clock(current)} / {_clock(duration)}")
        if not self._slider_held and duration > 0:
            self.timeline.blockSignals(True)
            self.timeline.setValue(int(max(0.0, min(1.0, float(position.get("ratio") or 0.0))) * 1000))
            self.timeline.blockSignals(False)

    def _on_frame(self, image) -> None:
        self._standby = False
        self.video.set_frame(image)
        self._have_frame = True
        self._frame_shape = (image.height(), image.width())
        if not self.recorder.recorder.active:
            self.record_btn.setEnabled(True)

    def _on_state(self, info: dict) -> None:
        self.last_info = info
        demo = bool(info.get("demo"))
        self.demo_badge.setVisible(demo)
        kind = info.get("model_kind") or ""
        name = info.get("model_name") or info.get("detector") or ""
        self.model_badge.setText(f"{tr(self.lang, 'model_badge')}: {name}")
        tracks = info.get("tracks") or []
        self.table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            kmh = track.get("speed_kmh")
            status_code = str(track.get("status") or "")
            values = [
                str(track.get("id") or ""),
                class_text(self.lang, str(track.get("class_name") or "")),
                f"{float(track.get('confidence') or 0):.2f}",
                tr(self.lang, "st_" + status_code) if status_code else "",
                f"{float(track.get('speed_px_s') or 0):.1f}",
                tr(self.lang, "kmh_na") if kmh is None else f"{float(kmh):.1f}",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        label = info.get("target_label") or tr(self.lang, "targets_empty")
        speed = info.get("speed_px_s")
        speed_text = "" if speed is None else f"{float(speed):.1f} {tr(self.lang, 'unit_px')}"
        if info.get("speed_kmh") is not None:
            speed_text += f"  {float(info['speed_kmh']):.1f} {tr(self.lang, 'unit_kmh')}"
        elif speed is not None:
            speed_text += f"  {tr(self.lang, 'kmh_na')}"
        status_code = str(info.get("status") or "")
        status_text = tr(self.lang, "st_" + status_code) if status_code else ""
        self.target_readout.setText(f"{label}   {status_text}   {speed_text}".strip())
        ptz = info.get("ptz") or {}
        if ptz:
            stop = f"  {tr(self.lang, 'ptz_stop')}" if ptz.get("stopped") else ""
            self.ptz_readout.setText(
                f"{tr(self.lang, 'ptz_pan')} {float(ptz.get('pan') or 0):+.1f}   {tr(self.lang, 'ptz_tilt')} {float(ptz.get('tilt') or 0):+.1f}{stop}"
            )
        self.perf["source_fps"].setText(f"{float(info.get('source_fps') or 0):.1f}")
        self.perf["detected"].setText(str(info.get("detected_count", 0)))
        self.perf["det_ms"].setText(f"{float(info.get('det_ms') or 0):.1f}")
        self.perf["track_ms"].setText(f"{float(info.get('track_ms') or 0):.1f}")
        self.perf["render_ms"].setText(f"{float(info.get('render_ms') or 0):.1f}")
        self.perf["backend"].setText(str(info.get("backend") or "—"))
        status_code = str(info.get("status") or "")
        self.perf["status"].setText(tr(self.lang, "st_" + status_code) if status_code else "—")
        self.perf["dropped"].setText(str(info.get("dropped", 0)))
        warning = info.get("warning_text") or ""
        if warning and warning != getattr(self, "_last_warning", ""):
            self.statusBar().showMessage(warning)
            self._last_warning = warning
        if info.get("event_error"):
            self.event_log.appendPlainText(str(info["event_error"]))
        self.table.resizeColumnsToContents()

    def _on_model(self, status: str, backend: str) -> None:
        if status and status != "READY":
            self.statusBar().showMessage(status)
        log.info("Vehicle model | %s | %s", status, backend)

    def _on_error(self, message: str) -> None:
        log.info("Error | %s", message)
        self.statusBar().showMessage(message.splitlines()[0] if message else "Error")
        QMessageBox.warning(self, "UAV Vision", message)

    def _record(self) -> None:
        if self.recorder.recorder.active or not self._have_frame:
            return
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RECORDINGS_DIR / f"uav_vision_{stamp}.mp4"
        height, width = self._frame_shape
        if width < 2 or height < 2:
            return
        seconds = int(self.record_seconds.value())
        self.record_btn.setEnabled(False)
        self.rec_label.setText(f"● REC 00:{seconds:02d}")
        self.recorder.request_start(path, self._source_fps or 25.0, (height, width), float(seconds))

    def _refresh_rec(self) -> None:
        if self._standby and self._standby_images:
            self._standby_elapsed += self._clock.interval()
            if self._standby_elapsed >= STANDBY_INTERVAL_MS:
                self._standby_elapsed = 0
                self._standby_index = (self._standby_index + 1) % len(self._standby_images)
                self.video.set_frame(self._standby_images[self._standby_index])
        if self.recorder.recorder.active:
            shown = int(round(max(0.0, self.recorder.recorder.remaining)))
            self.rec_label.setText(f"● REC 00:{shown:02d}")

    def _on_record_finished(self, path: str) -> None:
        self.rec_label.setText("")
        self.record_btn.setEnabled(self._have_frame)
        self.statusBar().showMessage(path)

    def _save_json(self) -> None:
        events = self._current_events()
        if not events:
            self.statusBar().showMessage(tr(self.lang, "targets_empty"))
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
        self.event_log.appendPlainText(str(path))
        self.statusBar().showMessage(str(path))

    def _send_test(self) -> None:
        events = self._current_events()
        if not events:
            self.statusBar().showMessage(tr(self.lang, "targets_empty"))
            return
        try:
            count = CommandCenterClient().send_events(
                events,
                self.event_host.text().strip(),
                int(self.event_port.value()),
                EVENTS_DIR / "live.jsonl",
                min_interval=0,
            )
            self.event_log.appendPlainText(f"UDP {count}")
        except Exception as exc:
            self.event_log.appendPlainText(str(exc))
            self.statusBar().showMessage(str(exc))

    def _current_events(self) -> list[dict]:
        info = self.last_info or {}
        tracks = info.get("tracks") or []
        if not tracks:
            return []
        return events_from_tracks(
            tracks,
            int(info.get("frame_w") or 1),
            int(info.get("frame_h") or 1),
            self.sensor_id.text().strip() or "uav-vision-01",
            self._video_path,
        )

    def _export_full(self) -> None:
        if not self._video_path:
            return
        self._pause()
        self._export_dialog = QProgressDialog("…", "", 0, 100, self)
        self._export_dialog.setWindowTitle("Export")
        self._export_dialog.setMinimumDuration(0)
        self.detector.request_export(self._video_path)

    def _on_export_progress(self, index: int, total: int) -> None:
        dialog = getattr(self, "_export_dialog", None)
        if dialog is not None and total > 0:
            dialog.setMaximum(total)
            dialog.setValue(min(index, total))

    def _on_export_finished(self, path: str) -> None:
        dialog = getattr(self, "_export_dialog", None)
        if dialog is not None:
            dialog.close()
        self.statusBar().showMessage(path)

    def _import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, tr(self.lang, "import_images"), "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        from training.dataset import import_images

        count = import_images(files, DATASET_DIR / "incoming")
        self._log(f"images {count}")

    def _import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr(self.lang, "import_video"), "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        every = int(self.every_n.value())

        def work():
            from training.dataset import import_video

            return import_video(path, DATASET_DIR / "incoming", every)

        self._run_job(work, lambda count: self._log(f"frames {count}"))

    def _import_labels(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr(self.lang, "import_labels"))
        if not folder:
            return
        from training.dataset import import_labels

        count = import_labels([folder], DATASET_DIR / "incoming")
        self._log(f"labels {count}")

    def _split(self) -> None:
        try:
            from training.dataset import prepare_split

            result = prepare_split(DATASET_DIR / "incoming", DATASET_DIR / "uav", float(self.val_ratio.value()))
            self._log(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            self._log(str(exc))

    def _train(self) -> None:
        yaml_path = DATASET_DIR / "uav" / "data.yaml"
        weights = self.base_weights.text().strip()
        epochs = int(self.epochs.value())
        imgsz = int(self.uav_imgsz.currentData() or 640)
        device = self.uav_device.currentText()

        def work():
            from training.dataset import run_training

            return run_training(str(yaml_path), weights, epochs, imgsz, device, str(DATASET_DIR / "runs"), "uav")

        self._run_job(work, self._show_metrics)

    def _load_metrics(self) -> None:
        folder = self.results_dir.text().strip()
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, tr(self.lang, "results_dir"))
        if not folder:
            self.metric_text.setText(tr(self.lang, "metrics_empty"))
            return
        self.results_dir.setText(folder)
        from training.dataset import load_measured_metrics

        self._show_metrics(load_measured_metrics(folder))

    def _show_metrics(self, metrics) -> None:
        if not metrics:
            self.metric_text.setText(tr(self.lang, "metrics_empty"))
            self.matrix.clear()
            return

        def fmt(value):
            return "—" if value is None else f"{float(value):.4f}"

        text = "\n".join(
            [
                f"precision {fmt(metrics.get('precision'))}",
                f"recall {fmt(metrics.get('recall'))}",
                f"mAP50 {fmt(metrics.get('map50'))}",
                f"mAP50-95 {fmt(metrics.get('map50_95'))}",
                str(metrics.get("source") or ""),
            ]
        )
        self.metric_text.setText(text)
        matrix = metrics.get("confusion_matrix")
        if matrix and Path(matrix).is_file():
            self.matrix.setPixmap(QPixmap(matrix).scaledToWidth(360, Qt.TransformationMode.SmoothTransformation))
        else:
            self.matrix.clear()
        self._log(text)

    def _build_scenes(self) -> None:
        def work():
            from simulation.scenarios import write_demo_set

            return [str(path) for path in write_demo_set(DEMO_SOURCE_DIR, seconds=20)]

        self._run_job(work, lambda paths: self._log("\n".join(paths)))

    def _test_video(self) -> None:
        weights = self.weights.text().strip()
        if not Path(weights).is_file():
            self._log(tr(self.lang, "metrics_empty"))
            self.statusBar().showMessage(tr(self.lang, "metrics_empty"))
            return
        path, _ = QFileDialog.getOpenFileName(self, tr(self.lang, "test_video"), "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        conf = float(self.uav_conf.value())
        nms = float(self.uav_nms.value())
        imgsz = int(self.uav_imgsz.currentData() or 640)
        device = self.uav_device.currentText()

        def work():
            from detection.yolo_v8_detector import YoloV8Detector
            from training.video_eval import evaluate_video

            detector = YoloV8Detector()
            detector.load(weights, device)

            def detect(frame):
                return detector.detect(frame, conf, nms, imgsz)

            return evaluate_video(path, detect)

        self._run_job(work, lambda report: self._log(json.dumps(report, ensure_ascii=False, indent=2)))

    def _download_weights(self) -> None:
        if self._job is not None and self._job.isRunning():
            return
        self.download_weights_btn.setEnabled(False)
        self.download_weights_model_btn.setEnabled(False)
        self.library_status.setText(tr(self.lang, "download_running"))
        self.statusBar().showMessage(tr(self.lang, "download_running"))

        def work():
            from library.weights import download_all

            return download_all()

        def done(rows) -> None:
            self._apply_library(rows)
            self.download_weights_btn.setEnabled(True)
            self.download_weights_model_btn.setEnabled(True)

        def fail(message: str) -> None:
            self._log(message)
            self.library_status.setText(str(message))
            self.statusBar().showMessage(str(message).splitlines()[0] if message else "")
            self.download_weights_btn.setEnabled(True)
            self.download_weights_model_btn.setEnabled(True)

        job = _Job(work)
        self._job = job
        job.finished_ok.connect(done)
        job.failed.connect(fail)
        job.start()

    def _apply_library(self, rows: list[dict]) -> None:
        lines = []
        chosen = ""
        for row in rows:
            status = str(row.get("status") or "failed")
            detail = str(row.get("detail") or "")
            title = str(row.get("title") or row.get("id") or "")
            lines.append(f"{title}: {tr(self.lang, 'lib_' + status)}. {detail}".strip())
            if row.get("usable") and row.get("local_path") and not chosen:
                chosen = str(row["local_path"])
        text = "\n".join(lines)
        self.library_status.setText(text)
        self._log(text)
        if chosen:
            self.weights.setText(chosen)
            self._push()
        self.statusBar().showMessage(tr(self.lang, "download_done"))

    def _run_job(self, fn, done) -> None:
        if self._job is not None and self._job.isRunning():
            return
        job = _Job(fn)
        self._job = job
        job.finished_ok.connect(done)
        job.failed.connect(self._log)
        job.start()

    def _log(self, text) -> None:
        self.train_log.appendPlainText(str(text))

    def _browse(self, line: QLineEdit, pattern: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr(self.lang, "browse"), "", pattern)
        if path:
            line.setText(path)
            self._push()

    def _load_settings(self) -> None:
        if not CONFIG_PATH.is_file():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        self.lang = data.get("lang") or "ru"
        index = self.language.findData(self.lang)
        if index >= 0:
            self.language.setCurrentIndex(index)
        self.mission = data.get("mission") or "uav"
        self.weights.setText(data.get("weights_path") or "")
        self.base_weights.setText(data.get("base_weights") or "")
        self.rtsp_url.setText(data.get("rtsp_url") or "")
        self.sensor_id.setText(data.get("sensor_id") or "uav-vision-01")
        self.event_host.setText(data.get("event_host") or "127.0.0.1")
        self.event_port.setValue(int(data.get("event_port") or 5001))
        self._set_value(self.uav_conf, data.get("uav_conf", 0.35))
        self._set_value(self.uav_nms, data.get("uav_nms", 0.45))
        self._set_value(self.meters, data.get("meters_per_pixel", 0.0))
        geometry = data.get("geometry") or ""
        if geometry:
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))

    def _save_settings(self) -> None:
        ensure_dirs()
        data = {
            "lang": self.lang,
            "mission": self.mission,
            "weights_path": self.weights.text().strip(),
            "base_weights": self.base_weights.text().strip(),
            "rtsp_url": self.rtsp_url.text().strip(),
            "sensor_id": self.sensor_id.text().strip(),
            "event_host": self.event_host.text().strip(),
            "event_port": int(self.event_port.value()),
            "uav_conf": float(self.uav_conf.value()),
            "uav_nms": float(self.uav_nms.value()),
            "meters_per_pixel": float(self.meters.value()),
            "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

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

    @staticmethod
    def _double(low, high, value, step, decimals=2) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        return box

    @staticmethod
    def _set_value(box, value) -> None:
        box.setValue(float(value))


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
