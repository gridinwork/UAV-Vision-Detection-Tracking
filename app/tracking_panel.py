"""Target, smoothing, dead zone, and virtual command readouts."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrackingPanel(QWidget):
    changed = Signal()
    next_target = Signal()
    prev_target = Signal()
    clear_target = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self._mode_box())
        root.addWidget(self._readout_box())
        root.addWidget(self._command_box())
        self.target_mode.currentIndexChanged.connect(self._notify)
        self.algorithm.currentIndexChanged.connect(self._notify)
        self.smoothing.currentIndexChanged.connect(self._notify)
        self.control_display.currentIndexChanged.connect(self._notify)
        self.auto_reacquire.currentIndexChanged.connect(self._notify)
        self.dead_zone.valueChanged.connect(self._notify)
        self.pixel_scale.valueChanged.connect(self._notify)
        self.trail_length.valueChanged.connect(self._notify)
        self.lost_timeout.valueChanged.connect(self._notify)

    def _notify(self, *_args) -> None:
        self.changed.emit()

    def _mode_box(self) -> QGroupBox:
        box = QGroupBox("TARGET")
        form = QFormLayout(box)
        self.target_mode = QComboBox()
        self.target_mode.addItem("AUTO CENTER", "AUTO_CENTER")
        self.target_mode.addItem("CLICK TARGET", "CLICK")
        self.target_mode.addItem("FIRST DETECTED", "FIRST")
        self.algorithm = QComboBox()
        self.algorithm.addItem("ENHANCED", "ENHANCED")
        self.algorithm.addItem("ORIGINAL", "ORIGINAL")
        self.smoothing = QComboBox()
        self.smoothing.addItem("OFF", "OFF")
        self.smoothing.addItem("LOW", "LOW")
        self.smoothing.addItem("MEDIUM", "MEDIUM")
        self.smoothing.addItem("HIGH", "HIGH")
        self.smoothing.addItem("ORIGINAL 20", "ORIGINAL_20")
        self.smoothing.setCurrentIndex(4)
        self.dead_zone = QSpinBox()
        self.dead_zone.setRange(10, 40)
        self.dead_zone.setValue(20)
        self.dead_zone.setSuffix(" %")
        self.control_display = QComboBox()
        self.control_display.addItem("DIRECTIONAL", "DIRECTIONAL")
        self.control_display.addItem("NED SIMULATION", "NED")
        self.pixel_scale = QDoubleSpinBox()
        self.pixel_scale.setRange(0.01, 2.0)
        self.pixel_scale.setDecimals(3)
        self.pixel_scale.setSingleStep(0.05)
        self.pixel_scale.setValue(0.1)
        self.trail_length = QSpinBox()
        self.trail_length.setRange(30, 100)
        self.trail_length.setValue(60)
        self.lost_timeout = QDoubleSpinBox()
        self.lost_timeout.setRange(0.5, 3.0)
        self.lost_timeout.setSingleStep(0.1)
        self.lost_timeout.setDecimals(1)
        self.lost_timeout.setValue(1.5)
        self.lost_timeout.setSuffix(" s")
        self.auto_reacquire = QComboBox()
        self.auto_reacquire.addItem("ON", True)
        self.auto_reacquire.addItem("OFF", False)
        prev_btn = QPushButton("PREVIOUS TARGET")
        next_btn = QPushButton("NEXT TARGET")
        clear_btn = QPushButton("CLEAR TARGET")
        prev_btn.clicked.connect(self.prev_target.emit)
        next_btn.clicked.connect(self.next_target.emit)
        clear_btn.clicked.connect(self.clear_target.emit)
        buttons = QHBoxLayout()
        buttons.addWidget(prev_btn)
        buttons.addWidget(next_btn)
        form.addRow("Target Mode", self.target_mode)
        form.addRow("Tracking Algorithm", self.algorithm)
        form.addRow("Target Smoothing", self.smoothing)
        form.addRow("Dead Zone Size", self.dead_zone)
        form.addRow("Control Display", self.control_display)
        form.addRow("Pixel Control Scale", self.pixel_scale)
        form.addRow("Scale note", QLabel("Simulation scale only"))
        form.addRow("Trail Length", self.trail_length)
        form.addRow("Lost Timeout", self.lost_timeout)
        form.addRow("Auto Reacquire", self.auto_reacquire)
        form.addRow(buttons)
        form.addRow(clear_btn)
        return box

    def _readout_box(self) -> QGroupBox:
        box = QGroupBox("TRACKING")
        form = QFormLayout(box)
        self.values: dict[str, QLabel] = {}
        rows = (
            ("id", "ID"),
            ("status", "Status"),
            ("confidence", "Confidence"),
            ("x_offset", "X Offset"),
            ("y_offset", "Y Offset"),
            ("filtered_x", "Filtered X"),
            ("filtered_y", "Filtered Y"),
            ("size", "Target Size"),
            ("trend", "Scale Trend"),
            ("direction", "Direction"),
            ("speed", "Virtual Speed"),
            ("zone", "Dead Zone"),
            ("control", "Control State"),
        )
        for key, title in rows:
            label = QLabel("—")
            label.setObjectName("ReadoutVal")
            self.values[key] = label
            form.addRow(title, label)
        return box

    def _command_box(self) -> QGroupBox:
        box = QGroupBox("VIRTUAL COMMAND")
        form = QFormLayout(box)
        self.command = QLabel("HOLD")
        self.command.setObjectName("ReadoutVal")
        self.north = QLabel("+0.0")
        self.north.setObjectName("ReadoutVal")
        self.east = QLabel("+0.0")
        self.east.setObjectName("ReadoutVal")
        self.error = QLabel("0%")
        self.error.setObjectName("ReadoutVal")
        self.norm = QLabel("0.00 / 0.00")
        self.norm.setObjectName("ReadoutVal")
        note = QLabel("Computed guidance only. No vehicle link.")
        note.setWordWrap(True)
        note.setObjectName("Subtitle")
        form.addRow("Command", self.command)
        form.addRow("N Command", self.north)
        form.addRow("E Command", self.east)
        form.addRow("Centering Error", self.error)
        form.addRow("Normalized X / Y", self.norm)
        form.addRow(note)
        return box

    def apply_state(self, info: dict) -> None:
        conf = info.get("target_conf")
        self.values["id"].setText(info.get("target_label") or "—")
        self.values["status"].setText(info.get("status") or "SEARCHING")
        self.values["confidence"].setText("—" if conf is None else f"{conf * 100:.1f}%")
        self.values["x_offset"].setText(f"{info.get('x_offset', 0):+.0f} px")
        self.values["y_offset"].setText(f"{info.get('y_offset', 0):+.0f} px")
        self.values["filtered_x"].setText(f"{info.get('filtered_x', 0):+.0f} px")
        self.values["filtered_y"].setText(f"{info.get('filtered_y', 0):+.0f} px")
        self.values["size"].setText(f"{info.get('target_size', 0):.1f}%")
        self.values["trend"].setText(str(info.get("scale_trend") or "STABLE"))
        self.values["direction"].setText(str(info.get("command") or "HOLD"))
        self.values["speed"].setText(str(info.get("speed") or "HOLD SPEED"))
        self.values["zone"].setText(str(info.get("dead_zone") or "—"))
        self.values["control"].setText(str(info.get("control_state") or "—"))
        self.command.setText(str(info.get("command") or "HOLD"))
        self.north.setText(f"{info.get('virtual_n', 0):+.1f}")
        self.east.setText(f"{info.get('virtual_e', 0):+.1f}")
        self.error.setText(f"{info.get('centering_error', 0):.0f}%")
        self.norm.setText(f"{info.get('x_norm', 0):+.2f} / {info.get('y_norm', 0):+.2f}")
