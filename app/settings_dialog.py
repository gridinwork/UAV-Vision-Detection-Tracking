"""Extra session options. The live panels already expose the same values."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Session Settings")
        self.setModal(True)
        layout = QVBoxLayout(self)
        note = QLabel(
            "These values are also on the main panels. "
            "Pixel scale is a simulation coefficient, not a calibrated metres-per-pixel value. "
            "The flight backend stays VIRTUAL."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.lost_timeout = QDoubleSpinBox()
        self.lost_timeout.setRange(0.5, 3.0)
        self.lost_timeout.setSingleStep(0.1)
        self.lost_timeout.setDecimals(1)
        self.pixel_scale = QDoubleSpinBox()
        self.pixel_scale.setRange(0.01, 2.0)
        self.pixel_scale.setDecimals(3)
        self.pixel_scale.setSingleStep(0.05)
        self.trail = QSpinBox()
        self.trail.setRange(30, 100)
        self.scale = QComboBox()
        self.scale.addItem("100%", 1.0)
        self.scale.addItem("75%", 0.75)
        self.scale.addItem("50%", 0.50)
        self.processing = QComboBox()
        self.processing.addItem("REALTIME", "REALTIME")
        self.processing.addItem("EVERY FRAME", "EVERY_FRAME")
        self.reacquire = QComboBox()
        self.reacquire.addItem("ON", True)
        self.reacquire.addItem("OFF", False)
        form.addRow("Lost Timeout (s)", self.lost_timeout)
        form.addRow("Pixel Control Scale", self.pixel_scale)
        form.addRow("Simulation scale only", QLabel("do not treat N/E as real metres"))
        form.addRow("Trail Length", self.trail)
        form.addRow("Detection Scale", self.scale)
        form.addRow("Processing Mode", self.processing)
        form.addRow("Auto Reacquire", self.reacquire)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def load_from(self, window) -> None:
        tracking = window.tracking
        control = window.control
        self.lost_timeout.setValue(tracking.lost_timeout.value())
        self.pixel_scale.setValue(tracking.pixel_scale.value())
        self.trail.setValue(tracking.trail_length.value())
        self._set_data(self.scale, control.detection_scale.currentData())
        self._set_data(self.processing, control.processing_mode.currentData())
        self._set_data(self.reacquire, tracking.auto_reacquire.currentData())

    def apply_to(self, window) -> None:
        tracking = window.tracking
        control = window.control
        tracking.lost_timeout.setValue(self.lost_timeout.value())
        tracking.pixel_scale.setValue(self.pixel_scale.value())
        tracking.trail_length.setValue(self.trail.value())
        self._set_data(control.detection_scale, self.scale.currentData())
        self._set_data(control.processing_mode, self.processing.currentData())
        self._set_data(tracking.auto_reacquire, self.reacquire.currentData())

    @staticmethod
    def _set_data(combo: QComboBox, value) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
