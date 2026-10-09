"""Open the studio, play a sample video, and record 15 seconds."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import DARK_QSS
from utils.paths import RECORDINGS_DIR, SAMPLES_DIR


def main() -> int:
    video = SAMPLES_DIR / "sample_drone.mp4"
    if not video.is_file():
        print("Missing sample video. Run tools/verify_install.py first.")
        return 1
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    window = MainWindow()
    window.show()
    phase = {
        "step": "wait_model",
        "info": None,
        "record_path": "",
        "started": time.perf_counter(),
        "saw_original": False,
    }

    def on_state(info: dict) -> None:
        phase["info"] = info

    def on_record(path: str) -> None:
        phase["record_path"] = path
        phase["step"] = "done"

    window.detector.state_ready.connect(on_state)
    window.recorder.finished_ok.connect(on_record)
    def poll() -> None:
        step = phase["step"]
        if step == "wait_model":
            if "READY" in window.model_badge.text():
                window.capture.post("open_video", path=str(video))
                window.capture.post("play")
                phase["step"] = "wait_detect"
                phase["started"] = time.perf_counter()
            elif "ERROR" in window.model_badge.text() or time.perf_counter() - phase["started"] > 30:
                print("Model did not become ready:", window.model_badge.text())
                phase["step"] = "fail"
            return
        info = phase["info"]
        if step == "wait_detect":
            if info and info.get("detected_count", 0) > 0 and info.get("target_label"):
                print(
                    "LIVE",
                    info.get("target_label"),
                    info.get("status"),
                    info.get("command"),
                    "off",
                    round(info.get("x_offset", 0), 1),
                    round(info.get("y_offset", 0), 1),
                    "NE",
                    round(info.get("virtual_n", 0), 2),
                    round(info.get("virtual_e", 0), 2),
                    "backend",
                    info.get("backend"),
                )
                window.detector.request_screenshot()
                index = window.tracking.algorithm.findData("ORIGINAL")
                window.tracking.algorithm.setCurrentIndex(index)
                phase["step"] = "wait_original"
            elif time.perf_counter() - phase["started"] > 20:
                print("Timed out waiting for a detection")
                phase["step"] = "fail"
            return
        if step == "wait_original":
            if info and info.get("algorithm") == "ORIGINAL" and info.get("detected_count", 0) > 0:
                print("ORIGINAL", info.get("status"), info.get("command"), "smooth window locked to 20")
                phase["saw_original"] = True
                index = window.tracking.algorithm.findData("ENHANCED")
                window.tracking.algorithm.setCurrentIndex(index)
                phase["step"] = "wait_enhanced"
            elif time.perf_counter() - phase["started"] > 25:
                print("ORIGINAL mode did not report a target")
                phase["step"] = "fail"
            return
        if step == "wait_enhanced":
            if info and info.get("algorithm") == "ENHANCED":
                window._record()
                phase["step"] = "recording"
                phase["started"] = time.perf_counter()
                print("RECORD started")
            elif time.perf_counter() - phase["started"] > 25:
                print("Did not return to enhanced mode")
                phase["step"] = "fail"
            return
        if step == "recording":
            if phase["record_path"]:
                phase["step"] = "done"
            elif time.perf_counter() - phase["started"] > 22:
                print("Recording did not finish")
                phase["step"] = "fail"
            return
        if step in ("done", "fail"):
            app.quit()

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(200)
    app.exec()
    window.close()

    if phase["step"] == "fail" or not phase["saw_original"]:
        print("GUI smoke failed", phase["step"], "original", phase["saw_original"])
        return 1
    path = Path(phase["record_path"])
    if not path.is_file():
        print("Recording file missing")
        return 1
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    ok, frame = cap.read()
    cap.release()
    duration = count / fps if fps else 0
    print(f"GUI recording {path.name} duration {duration:.2f}s frames {count:.0f} opened {ok}")
    if not ok or abs(duration - 15.0) > 0.75:
        return 1
    shots = list((ROOT / "screenshots").glob("vtol_tracking_*.png"))
    print("screenshots", len(shots))
    if not shots:
        return 1
    print("GUI SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
