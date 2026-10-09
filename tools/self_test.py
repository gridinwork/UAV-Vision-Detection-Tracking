"""Headless checks for detection, tracking, virtual control, and the 15s recorder."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from control.control_logic import directional_command
from control.ned_simulation import virtual_ned
from control.speed_estimator import SpeedEstimator
from detection.yolo_v4_detector import Detection, YoloV4Detector
from recording.recorder import RECORD_SECONDS, SessionRecorder
from tracking.enhanced_tracker import EnhancedTracker
from tracking.engine import PipelineConfig, VisionEngine
from tracking.moving_average import MovingAverageFilter
from tracking.original_tracker import OriginalTracker
from utils.paths import RECORDINGS_DIR, SAMPLES_DIR, SCREENSHOTS_DIR, ensure_dirs


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print("OK", message)


def test_average() -> None:
    filt = MovingAverageFilter(20)
    values = list(range(1, 21))
    result = None
    for value in values:
        result = filt.push(value, 100 - value)
    expect(result == (10, 90), f"20-point average {result}")
    result = filt.push(21, 0)
    expect(result == (12, 85), f"buffer dropped oldest {result}")


def test_commands() -> None:
    half = 40.0
    expect(directional_command(10, -10, half, half, True) == "HOLD", "inside dead zone holds")
    expect(directional_command(80, 5, half, half, True) == "RIGHT", "right command")
    expect(directional_command(-80, -90, half, half, True) == "FORWARD + LEFT", "forward left")
    expect(directional_command(80, 90, half, half, True) == "BACK + RIGHT", "back right")
    expect(directional_command(0, 90, half, half, True) == "BACK", "back")
    north, east = virtual_ned(300, 100, 200, 200, 0.1, False)
    expect(abs(east - 10.0) < 1e-6 and abs(north - 10.0) < 1e-6, f"N/E {north},{east}")
    north, east = virtual_ned(300, 100, 200, 200, 0.1, True)
    expect(north == 0.0 and east == 0.0, "dead zone zeros N/E")


def test_original_nearest() -> None:
    tracker = OriginalTracker(20)
    near = Detection(0, "car", 0.9, (180, 180, 40, 40))
    far = Detection(0, "car", 0.99, (0, 0, 30, 30))
    state = tracker.update([far, near], 400, 400, 1, (40.0, 40.0))
    expect(state.box == near.box, "original tracker picks the car nearest the centre")
    expect(state.status in ("CENTERED", "CORRECTING"), state.status)


def test_enhanced_click_lost_reacquire() -> None:
    tracker = EnhancedTracker(1)
    car_a = Detection(0, "car", 0.91, (20, 20, 40, 30))
    car_b = Detection(0, "car", 0.88, (300, 200, 50, 40))
    now = 1000.0
    state = tracker.update([car_a, car_b], 640, 480, 1, (64, 48), (40, 30), "CLICK", 1.0, False, 40, now)
    expect(state.target_label == "CAR #1", f"click locked {state.target_label} {state.status}")
    lost = tracker.update([], 640, 480, 2, (64, 48), None, "CLICK", 1.0, False, 40, now + 0.2)
    expect(lost.status == "LOST", f"temporary loss {lost.status}")
    expect(lost.target_id == state.target_id, "click mode did not switch cars")
    back = Detection(0, "car", 0.93, (22, 22, 40, 30))
    again = tracker.update([back, car_b], 640, 480, 3, (64, 48), None, "CLICK", 1.0, False, 40, now + 0.4)
    expect(again.status == "REACQUIRED", f"reacquire {again.status} id {again.target_id}")
    expect(again.target_id == state.target_id, "same identity after the gap")
    expect(len(again.trail) >= 1, "trail is stored")


def test_speed() -> None:
    est = SpeedEstimator()
    now = 0.0
    speed, trend = est.update(now, 200, 0.08, True)
    for index in range(8):
        now += 0.15
        speed, trend = est.update(now, 200 - index * 8, 0.08 - index * 0.004, True)
    expect(speed == "SPEED UP", f"receding heuristic {speed}")
    expect(trend == "RECEDING", trend)
    est.reset()
    now = 0.0
    est.update(now, 100, 0.05, True)
    for index in range(8):
        now += 0.15
        speed, trend = est.update(now, 100 + index * 12, 0.05 + index * 0.01, True)
    expect(speed == "SLOW DOWN", f"closing heuristic {speed}")
    expect(trend == "APPROACHING", trend)


def test_detector_and_engine() -> None:
    detector = YoloV4Detector()
    backend = detector.load("CPU")
    expect(backend == "CPU", backend)
    image_path = next((SAMPLES_DIR / "images").glob("*.jpg"))
    frame = cv2.imread(str(image_path))
    expect(frame is not None, "sample image opened")
    found, _raw = detector.detect(frame, 0.2, 0.4)
    print(f"sample {image_path.name} detections {len(found)}")
    engine = VisionEngine()
    engine.load("CPU")
    best = None
    best_count = -1
    for path in (SAMPLES_DIR / "images").glob("*.jpg"):
        image = cv2.imread(str(path))
        if image is None:
            continue
        dets, _ = engine.detector.detect(image, 0.2, 0.4)
        if len(dets) > best_count:
            best_count = len(dets)
            best = image
    expect(best is not None and best_count > 0, "model produced car detections")
    config = PipelineConfig(algorithm="ENHANCED", smoothing="ORIGINAL_20", dead_zone=0.2, pixel_scale=0.1)
    config.overlays = {
        "boxes": True,
        "confidence": True,
        "ids": True,
        "trail": True,
        "crosshair": True,
        "dead_zone": True,
        "flight_director": True,
        "command_text": True,
        "fps": True,
        "target_vector": True,
        "debug": True,
    }
    view, info = engine.step(best, config)
    expect(view.shape == best.shape, "overlay keeps the frame size")
    expect(info["detected_count"] > 0, "engine reports cars")
    expect(info["target_label"].startswith("CAR"), info["target_label"])
    expect(info["backend_name"] == "VIRTUAL", "virtual backend")
    expect(info["command"] in {
        "HOLD", "MOVE LEFT", "MOVE RIGHT", "MOVE FORWARD", "MOVE BACK",
        "LEFT", "RIGHT", "FORWARD", "BACK",
        "FORWARD + LEFT", "FORWARD + RIGHT", "BACK + LEFT", "BACK + RIGHT",
    }, info["command"])
    print(
        "target", info["target_label"],
        "status", info["status"],
        "cmd", info["command"],
        "off", round(info["x_offset"], 1), round(info["y_offset"], 1),
        "NE", round(info["virtual_n"], 2), round(info["virtual_e"], 2),
    )
    shot = SCREENSHOTS_DIR / "self_test_frame.png"
    cv2.imwrite(str(shot), view)
    expect(shot.is_file() and shot.stat().st_size > 1000, "screenshot written")


def test_record(frame: np.ndarray) -> None:
    ensure_dirs()
    path = RECORDINGS_DIR / "self_test_record.mp4"
    if path.exists():
        path.unlink()
    recorder = SessionRecorder()
    recorder.start(path, 20.0, frame.shape[:2])
    t0 = time.perf_counter()
    while recorder.active and time.perf_counter() - t0 < RECORD_SECONDS + 2:
        painted = frame.copy()
        cv2.putText(
            painted,
            f"{time.perf_counter() - t0:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        recorder.submit(painted)
        time.sleep(0.05)
    recorder.stop()
    elapsed = time.perf_counter() - t0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    ok, sample = cap.read()
    cap.release()
    duration = count / fps if fps else 0
    print(f"record wall {elapsed:.2f}s file {duration:.2f}s frames {count:.0f} fps {fps:.2f}")
    expect(ok and sample is not None, "recorded mp4 opens")
    expect(abs(duration - 15.0) < 0.75, f"duration {duration:.2f}s")
    expect(12.0 <= elapsed <= 18.0, f"wall clock {elapsed:.2f}")


def main() -> None:
    ensure_dirs()
    test_average()
    test_commands()
    test_original_nearest()
    test_enhanced_click_lost_reacquire()
    test_speed()
    test_detector_and_engine()
    image = cv2.imread(str(next((SAMPLES_DIR / "images").glob("ezgif-frame-002*.jpg"))))
    test_record(image)
    print("SELF TEST PASSED")


if __name__ == "__main__":
    main()
