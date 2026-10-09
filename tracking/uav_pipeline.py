"""UAV frame step: YOLOv8 or a marked demo scene, then multi-object tracking."""

from __future__ import annotations

import time

import cv2

from app.i18n import class_text, tr
from detection.yolo_v8_detector import YoloV8Detector
from ptz.virtual_ptz import VirtualPtz
from simulation.scenarios import scene_by_key, scene_key_from_path
from tracking.multi_tracker import BotSortTracker, ByteTracker, TargetLock
from utils.logger import get_logger
from visualization.uav_draw import render_uav


log = get_logger()


def _lang(config) -> str:
    return "en" if getattr(config, "language", "ru") == "en" else "ru"


def _model_title(engine, lang: str) -> str:
    filename = str(engine.v8.path or "").replace("\\", "/").rsplit("/", 1)[-1]
    if engine.v8.kind == "coco_general" and filename:
        return tr(lang, "coco_title").format(file=filename)
    if engine.v8.kind == "unknown" and filename:
        return tr(lang, "unknown_title").format(file=filename)
    if engine.v8.kind == "uav_custom" and filename:
        return tr(lang, "uav_title").format(file=filename)
    return engine.v8.display_name or "YOLOv8"


def ensure_uav(engine) -> None:
    if getattr(engine, "v8", None) is None:
        engine.v8 = YoloV8Detector()
        engine.byte = ByteTracker()
        engine.bot = BotSortTracker()
        engine.lock = TargetLock()
        engine.ptz = VirtualPtz()
        engine._uav_weights = ""
        engine._uav_tracker_name = ""
        engine._last_tracks = []


def run_uav_step(engine, frame, config, click_xy, recording: bool, rec_label: str):
    ensure_uav(engine)
    started = time.perf_counter()
    height, width = frame.shape[:2]
    fps = float(config.source_fps or 25.0)
    if fps < 1.0 or fps > 240.0:
        fps = 25.0
    warning = ""
    detections = []
    detector_name = "YOLOv8"
    backend = engine.v8.backend_label
    try:
        if config.demo:
            detections, warning = _demo_detections(config, width, height, fps)
            detector_name = tr(_lang(config), "demo_banner")
            backend = "SIM"
        else:
            _load_if_needed(engine, config)
            if not engine.v8.ready:
                warning = "weights_missing"
            else:
                engine.v8.set_device(config.device)
                backend = engine.v8.backend_label
                detector_name = _model_title(engine, _lang(config))
                warning = engine.v8.warning_code
                infer_conf = min(float(config.confidence), float(config.low_conf))
                detections = engine.v8.detect(frame, infer_conf, float(config.nms), int(config.imgsz))
                backend = engine.v8.backend_label
    except Exception as exc:
        warning = "weights_missing"
        log.info("UAV detector error: %s", exc)
        detections = []
    det_ms = (time.perf_counter() - started) * 1000.0
    tracker = _tracker(engine, config)
    now = time.perf_counter()
    tracks = tracker.update(
        detections,
        now,
        fps,
        float(config.lost_timeout),
        float(config.meters_per_pixel or 0.0),
        high_conf=float(config.confidence),
        low_conf=float(config.low_conf),
        match_iou=float(config.match_iou),
    )
    selected = engine.lock.choose(tracks, width, height, click_xy, config.target_mode)
    engine.ptz.set_limits(config.ptz_pan_min, config.ptz_pan_max, config.ptz_tilt_min, config.ptz_tilt_max)
    engine.ptz.pan_offset = float(config.ptz_pan_offset)
    engine.ptz.tilt_offset = float(config.ptz_tilt_offset)
    engine.ptz.deg_per_norm = max(1.0, float(config.ptz_deg))
    track_ms = (time.perf_counter() - started) * 1000.0 - det_ms
    ptz_state = engine.ptz.snapshot()
    if config.ptz_slew and selected is not None and selected.status in ("DETECTED", "TRACKING", "REACQUIRED"):
        cx = selected.box[0] + selected.box[2] / 2.0
        cy = selected.box[1] + selected.box[3] / 2.0
        ptz_state = engine.ptz.slew_toward((cx - width / 2.0) / max(1.0, width / 2.0), (height / 2.0 - cy) / max(1.0, height / 2.0), 1.0 / fps)
    public = [track.as_dict() for track in tracks]
    lang = _lang(config)
    for item in public:
        item["class_label"] = class_text(lang, str(item.get("class_name") or ""))
        item["status_label"] = tr(lang, "st_" + str(item.get("status") or "DETECTED"))
    engine._last_tracks = tracks
    engine._last_mission = "uav"
    kind = "demo" if config.demo else engine.v8.kind
    info = _info(config, public, selected, width, height, detector_name, backend, warning, det_ms, track_ms, ptz_state, kind)
    view = render_uav(frame, public, info, config.overlays or {})
    if recording and rec_label:
        cv2.putText(view, rec_label, (12, view.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 230), 2, cv2.LINE_AA)
    info["render_ms"] = (time.perf_counter() - started) * 1000.0 - det_ms - track_ms
    info["total_ms"] = det_ms + track_ms + info["render_ms"]
    return view, info


def _demo_detections(config, width, height, fps):
    key = config.demo_scene or scene_key_from_path(config.source_uri)
    scene = scene_by_key(key)
    if scene is None:
        return [], "demo_scene_missing"
    return scene.detections(max(0, int(config.frame_index)), width, height, fps), ""


def _load_if_needed(engine, config) -> None:
    path = (config.weights_path or "").strip()
    if not path:
        return
    if path == engine._uav_weights and engine.v8.ready:
        return
    engine.v8.load(path, config.device, vehicle_only=False)
    engine._uav_weights = path


def _tracker(engine, config):
    name = (config.algorithm or "BYTETRACK").upper()
    if name not in ("BYTETRACK", "BOTSORT"):
        name = "BYTETRACK"
    if name != engine._uav_tracker_name:
        engine.byte.reset()
        engine.bot.reset()
        engine.lock.reset()
        engine._uav_tracker_name = name
    return engine.bot if name == "BOTSORT" else engine.byte


def _info(config, tracks, selected, width, height, detector_name, backend, warning, det_ms, track_ms, ptz_state, kind: str) -> dict:
    lang = _lang(config)
    status = "SEARCHING"
    label = ""
    confidence = None
    raw_xy = None
    box = None
    speed_px = None
    speed_kmh = None
    target_id = None
    if selected is not None:
        status = selected.status
        target_id = selected.track_id
        label = f"{class_text(lang, selected.class_name)} #{selected.track_id}"
        confidence = selected.confidence
        cx = selected.box[0] + selected.box[2] / 2.0
        cy = selected.box[1] + selected.box[3] / 2.0
        raw_xy = (cx, cy)
        box = selected.box
        speed_px = selected.speed_px_s
        speed_kmh = selected.speed_kmh
    elif tracks:
        status = "DETECTED"
    warning_text = tr(lang, "warn_" + warning) if warning else ""
    if warning_text.startswith("warn_"):
        warning_text = ""
    if config.demo and warning != "demo_scene_missing":
        warning_text = ""
    pan = float(getattr(ptz_state, "pan_deg", 0.0) or 0.0)
    tilt = float(getattr(ptz_state, "tilt_deg", 0.0) or 0.0)
    ptz_text = tr(lang, "ptz_line").format(pan=pan, tilt=tilt)
    if getattr(ptz_state, "stopped", False):
        ptz_text += "  " + tr(lang, "ptz_stop")
    return {
        "frame_id": int(config.frame_index),
        "frame_w": width,
        "frame_h": height,
        "detected_count": len(tracks),
        "raw_detections": len(tracks),
        "nms_detections": len(tracks),
        "confidence_threshold": config.confidence,
        "nms_threshold": config.nms,
        "backend": backend,
        "proc_fps": float(config.proc_fps or 0.0),
        "source_fps": float(config.source_fps or 0.0),
        "detector": detector_name,
        "yolo_input": f"{int(config.imgsz)}x{int(config.imgsz)}",
        "model_name": detector_name,
        "model_status": tr(lang, "demo_banner") if config.demo else detector_name,
        "banner": tr(lang, "demo_banner"),
        "status_label": tr(lang, "st_" + status),
        "unit_px": tr(lang, "unit_px"),
        "unit_kmh": tr(lang, "unit_kmh"),
        "kmh_na": tr(lang, "kmh_na"),
        "ptz_text": ptz_text,
        "model_kind": kind,
        "warning_code": warning,
        "warning_text": warning_text,
        "target_id": target_id,
        "target_label": label,
        "target_conf": confidence,
        "status": status,
        "command": "VIRTUAL PTZ" if config.ptz_slew else "HOLD",
        "speed": "px/s",
        "speed_px_s": speed_px,
        "speed_kmh": speed_kmh,
        "x_offset": (raw_xy[0] - width / 2.0) if raw_xy else 0.0,
        "y_offset": (raw_xy[1] - height / 2.0) if raw_xy else 0.0,
        "filtered_x": 0.0,
        "filtered_y": 0.0,
        "x_norm": ((raw_xy[0] / width) - 0.5) if raw_xy else 0.0,
        "y_norm": ((raw_xy[1] / height) - 0.5) if raw_xy else 0.0,
        "virtual_n": 0.0,
        "virtual_e": 0.0,
        "target_size": ((box[2] * box[3]) / float(width * height) * 100.0) if box else 0.0,
        "scale_trend": "STABLE",
        "dead_zone": "",
        "control_state": status,
        "centering_error": 0.0,
        "magnitude": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "raw_xy": raw_xy,
        "filtered_xy": raw_xy,
        "det_ms": det_ms,
        "track_ms": track_ms,
        "render_ms": 0.0,
        "total_ms": det_ms + track_ms,
        "backend_name": "VIRTUAL",
        "pixel_scale": config.pixel_scale,
        "algorithm": config.algorithm,
        "target_mode": config.target_mode,
        "tracks": tracks,
        "demo": bool(config.demo),
        "mission": "uav",
        "ptz": {
            "pan": ptz_state.pan_deg,
            "tilt": ptz_state.tilt_deg,
            "stopped": ptz_state.stopped,
            "backend": ptz_state.backend,
        },
        "box": box,
    }

