"""One-frame vision pipeline used by the live worker and the full-video export."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2

from control.speed_estimator import SpeedEstimator
from control.virtual_flight_controller import CommandBuilder
from detection.yolo_v4_detector import YoloV4Detector
from tracking.target_selector import TargetSelector
from utils.logger import get_logger
from utils.timers import LapTimer
from visualization.renderer import render_debug, render_frame


log = get_logger()


@dataclass
class PipelineConfig:
    confidence: float = 0.2
    nms: float = 0.4
    device: str = "AUTO"
    target_mode: str = "AUTO_CENTER"
    algorithm: str = "ENHANCED"
    dead_zone: float = 0.20
    smoothing: str = "ORIGINAL_20"
    pixel_scale: float = 0.1
    trail_length: int = 60
    lost_timeout: float = 1.5
    auto_reacquire: bool = True
    detection_scale: float = 1.0
    control_display: str = "DIRECTIONAL"
    overlays: dict = field(default_factory=dict)
    source_fps: float = 0.0
    proc_fps: float = 0.0
    show_debug: bool = False
    mission: str = "vehicle"
    imgsz: int = 640
    demo: bool = False
    demo_scene: str = ""
    weights_path: str = ""
    meters_per_pixel: float = 0.0
    sensor_id: str = "uav-vision-01"
    source_uri: str = ""
    frame_index: int = 0
    emit_events: bool = False
    event_host: str = "127.0.0.1"
    event_port: int = 5001
    event_log: str = ""
    low_conf: float = 0.10
    match_iou: float = 0.30
    ptz_slew: bool = False
    record_seconds: float = 15.0
    ptz_pan_min: float = -90.0
    ptz_pan_max: float = 90.0
    ptz_tilt_min: float = -30.0
    ptz_tilt_max: float = 45.0
    ptz_deg: float = 35.0
    ptz_pan_offset: float = 0.0
    ptz_tilt_offset: float = 0.0
    language: str = "ru"


class VisionEngine:
    def __init__(self) -> None:
        self.detector = YoloV4Detector()
        self.selector = TargetSelector()
        self.commands = CommandBuilder()
        self.speed = SpeedEstimator()
        self.frame_id = 0
        self._device = ""
        self._last_target_key: tuple | None = None
        self.vehicle_status = "NOT LOADED"
        self._last_mission = "vehicle"
        self._last_tracks = []

    def load(self, device: str = "AUTO") -> str:
        self._device = device
        try:
            label = self.detector.load(device)
            self.vehicle_status = self.detector.manager.status_text
            return label
        except Exception as exc:
            self.vehicle_status = str(exc)
            log.info("YOLOv4-tiny is not available: %s", exc)
            return "CPU"

    def reset(self) -> None:
        self.selector.reset()
        self.speed.reset()
        self.frame_id = 0
        self._last_target_key = None
        if getattr(self, "byte", None) is not None:
            self.byte.reset()
            self.bot.reset()
            self.lock.reset()
            self._uav_tracker_name = ""

    def cycle_target(self, step: int) -> None:
        if self._last_mission == "uav" and getattr(self, "lock", None) is not None:
            self.lock.cycle(step, self._last_tracks)
            return
        self.selector.cycle(step)

    def clear_target(self) -> None:
        if getattr(self, "lock", None) is not None:
            self.lock.clear()
        self.selector.clear_target()

    def step(
        self,
        frame,
        config: PipelineConfig,
        click_xy: tuple[float, float] | None = None,
        recording: bool = False,
        rec_label: str = "",
    ) -> tuple[object, dict]:
        if getattr(config, "mission", "vehicle") == "uav":
            from tracking.uav_pipeline import run_uav_step

            return run_uav_step(self, frame, config, click_xy, recording, rec_label)
        if not self.detector.ready:
            return self._unavailable(frame, config, self.vehicle_status or "YOLOv4-tiny weights are not loaded")
        timer = LapTimer()
        if config.device != self._device:
            self.detector.set_device(config.device)
            self._device = config.device
        detections, raw_count = self.detector.detect(
            frame,
            confidence=config.confidence,
            nms=config.nms,
            detection_scale=config.detection_scale,
            count_raw=config.show_debug,
        )
        det_ms = timer.lap_ms()
        self.frame_id += 1
        height, width = frame.shape[:2]
        state = self.selector.update(
            detections,
            width,
            height,
            self.frame_id,
            config.dead_zone,
            click_xy,
            config.target_mode,
            config.algorithm,
            config.smoothing,
            config.lost_timeout,
            config.auto_reacquire,
            config.trail_length,
        )
        state.raw_detection_count = raw_count
        active = state.target_id is not None and state.status not in ("LOST", "SEARCHING", "DETECTED")
        if state.status == "REACQUIRED":
            active = True
        packet = self.commands.build(state, config.pixel_scale, active)
        now = time.perf_counter()
        speed_cmd, trend = self.speed.update(
            now,
            state.raw_xy[1] if active and state.raw_xy else None,
            state.area_ratio if active else None,
            active,
        )
        state.scale_trend = trend
        packet["speed"] = speed_cmd
        track_ms = timer.lap_ms()
        view = render_frame(
            frame,
            state,
            packet,
            config,
            self.detector.backend_label,
            recording=recording,
            rec_label=rec_label,
        )
        render_ms = timer.lap_ms()
        info = self._info(state, packet, config, det_ms, track_ms, render_ms, raw_count, len(detections))
        if config.show_debug or (config.overlays or {}).get("debug", False):
            render_debug(view, info)
        self._last_target_key = (state.status, state.target_id)
        return view, info

    def _info(self, state, packet, config: PipelineConfig, det_ms, track_ms, render_ms, raw_count, nms_count) -> dict:
        return {
            "frame_id": state.frame_id,
            "frame_w": state.frame_w,
            "frame_h": state.frame_h,
            "detected_count": nms_count,
            "raw_detections": raw_count,
            "nms_detections": nms_count,
            "confidence_threshold": config.confidence,
            "nms_threshold": config.nms,
            "backend": self.detector.backend_label,
            "detector": "YOLOv4-tiny",
            "yolo_input": "416x416",
            "model_name": self.detector.manager.display_name,
            "model_status": self.detector.manager.status_text,
            "target_id": state.target_id,
            "target_label": state.target_label,
            "target_conf": state.confidence,
            "status": state.status,
            "command": packet["command"],
            "speed": packet["speed"],
            "x_offset": packet["raw_offset_x"],
            "y_offset": packet["raw_offset_y"],
            "filtered_x": packet["filtered_offset_x"],
            "filtered_y": packet["filtered_offset_y"],
            "x_norm": packet["x_norm"],
            "y_norm": packet["y_norm"],
            "virtual_n": packet["virtual_n"],
            "virtual_e": packet["virtual_e"],
            "target_size": state.area_ratio * 100.0,
            "scale_trend": state.scale_trend,
            "dead_zone": packet["dead_zone"],
            "control_state": packet["control_state"],
            "centering_error": packet["centering_error"],
            "magnitude": packet["magnitude"],
            "dx": packet["dx"],
            "dy": packet["dy"],
            "raw_xy": state.raw_xy,
            "filtered_xy": state.filtered_xy,
            "det_ms": det_ms,
            "track_ms": track_ms,
            "render_ms": render_ms,
            "total_ms": det_ms + track_ms + render_ms,
            "backend_name": "VIRTUAL",
            "pixel_scale": config.pixel_scale,
            "algorithm": config.algorithm,
            "target_mode": config.target_mode,
            "tracks": [
                {
                    "id": track.get("id"),
                    "class_name": track.get("class_name") or "car",
                    "confidence": track.get("confidence"),
                    "box": track.get("box"),
                    "status": "TRACKING" if track.get("selected") else "DETECTED",
                    "speed_px_s": 0.0,
                    "speed_kmh": None,
                    "selected": track.get("selected"),
                    "trail": [],
                }
                for track in state.tracks
            ],
            "demo": False,
            "mission": "vehicle",
            "speed_px_s": None,
            "speed_kmh": None,
            "warning_code": "",
            "warning_text": "",
            "model_kind": "yolov4",
            "ptz": {},
        }

    def _unavailable(self, frame, config: PipelineConfig, message: str):
        view = frame.copy()
        cv2.putText(view, message[:180], (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 220), 2, cv2.LINE_AA)
        height, width = frame.shape[:2]
        info = {
            "frame_id": self.frame_id,
            "frame_w": width,
            "frame_h": height,
            "detected_count": 0,
            "raw_detections": 0,
            "nms_detections": 0,
            "backend": "CPU",
            "detector": "YOLOv4-tiny",
            "yolo_input": "416x416",
            "model_name": "YOLOv4-tiny",
            "model_status": message,
            "target_id": None,
            "target_label": "",
            "target_conf": None,
            "status": "SEARCHING",
            "command": "HOLD",
            "speed": "HOLD SPEED",
            "x_offset": 0.0,
            "y_offset": 0.0,
            "filtered_x": 0.0,
            "filtered_y": 0.0,
            "x_norm": 0.0,
            "y_norm": 0.0,
            "virtual_n": 0.0,
            "virtual_e": 0.0,
            "target_size": 0.0,
            "scale_trend": "STABLE",
            "dead_zone": "",
            "raw_xy": None,
            "filtered_xy": None,
            "det_ms": 0.0,
            "track_ms": 0.0,
            "render_ms": 0.0,
            "total_ms": 0.0,
            "backend_name": "VIRTUAL",
            "pixel_scale": config.pixel_scale,
            "algorithm": config.algorithm,
            "target_mode": config.target_mode,
            "tracks": [],
            "demo": False,
            "mission": "vehicle",
            "speed_px_s": None,
            "speed_kmh": None,
            "warning_code": "weights_missing",
            "warning_text": message,
            "model_kind": "yolov4",
            "ptz": {},
            "proc_fps": config.proc_fps,
            "source_fps": config.source_fps,
        }
        return view, info
