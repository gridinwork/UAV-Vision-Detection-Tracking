"""Timing, device, and debug readout."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class PerformancePanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("PERFORMANCE")
        form = QFormLayout(box)
        self.values: dict[str, QLabel] = {}
        for key, title in (
            ("source_fps", "Source FPS"),
            ("det_fps", "Detection FPS"),
            ("display_fps", "Display FPS"),
            ("det_ms", "Detection ms"),
            ("track_ms", "Tracking ms"),
            ("render_ms", "Render ms"),
            ("device", "Device"),
            ("target", "Target"),
            ("dropped", "Dropped Frames"),
        ):
            label = QLabel("—")
            label.setObjectName("ReadoutVal")
            self.values[key] = label
            form.addRow(title, label)
        root.addWidget(box)
        self.debug = QPlainTextEdit()
        self.debug.setReadOnly(True)
        self.debug.setPlaceholderText("Debug")
        self.debug.setMinimumHeight(140)
        self.debug.hide()
        root.addWidget(self.debug)

    def apply_state(self, info: dict) -> None:
        self.values["source_fps"].setText(f"{info.get('source_fps', 0):.1f}")
        self.values["det_fps"].setText(f"{info.get('proc_fps', 0):.1f}")
        self.values["display_fps"].setText(f"{info.get('display_fps', 0):.1f}")
        self.values["det_ms"].setText(f"{info.get('det_ms', 0):.1f}")
        self.values["track_ms"].setText(f"{info.get('track_ms', 0):.1f}")
        self.values["render_ms"].setText(f"{info.get('render_ms', 0):.1f}")
        self.values["device"].setText(str(info.get("backend") or "—"))
        self.values["target"].setText(info.get("target_label") or "—")
        self.values["dropped"].setText(str(info.get("dropped", 0)))
        raw = info.get("raw_xy")
        filt = info.get("filtered_xy")
        raw_s = "—" if not raw else f"{raw[0]:.1f}, {raw[1]:.1f}"
        filt_s = "—" if not filt else f"{filt[0]:.1f}, {filt[1]:.1f}"
        self.debug.setPlainText(
            "\n".join(
                [
                    f"Frame ID: {info.get('frame_id')}",
                    f"Frame Size: {info.get('frame_w')} x {info.get('frame_h')}",
                    f"YOLO Input: {info.get('yolo_input')}",
                    f"Raw Detections: {info.get('raw_detections')}",
                    f"NMS Detections: {info.get('nms_detections')}",
                    f"Confidence: {info.get('target_conf')}",
                    f"Target ID: {info.get('target_label')}",
                    f"Target Raw X/Y: {raw_s}",
                    f"Target Filtered X/Y: {filt_s}",
                    f"X Offset: {info.get('x_offset', 0):+.1f}",
                    f"Y Offset: {info.get('y_offset', 0):+.1f}",
                    f"Normalized Offset: {info.get('x_norm', 0):+.3f}, {info.get('y_norm', 0):+.3f}",
                    f"Dead-zone: {info.get('dead_zone')}",
                    f"N/E simulation: {info.get('virtual_n', 0):+.3f}, {info.get('virtual_e', 0):+.3f}",
                    f"BBox area %: {info.get('target_size', 0):.2f}",
                    f"Area trend: {info.get('scale_trend')}",
                    f"Tracking time: {info.get('track_ms', 0):.2f} ms",
                    f"Inference time: {info.get('det_ms', 0):.2f} ms",
                    f"Render time: {info.get('render_ms', 0):.2f} ms",
                    f"Total frame time: {info.get('total_ms', 0):.2f} ms",
                    f"Algorithm: {info.get('algorithm')}",
                    f"Flight backend: {info.get('backend_name')}",
                    f"Pixel scale: {info.get('pixel_scale')} (simulation only)",
                ]
            )
        )

    def set_debug_visible(self, visible: bool) -> None:
        self.debug.setVisible(visible)
