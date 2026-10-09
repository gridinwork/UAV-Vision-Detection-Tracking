"""Project paths and directory setup."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
WEIGHTS_PATH = MODEL_DIR / "custom-yolov4-tiny-detector_best.weights"
CFG_PATH = MODEL_DIR / "custom-yolov4-tiny-detector.cfg"
CLASSES_PATH = MODEL_DIR / "classes.txt"
CONFIG_PATH = ROOT / "config" / "settings.json"
LOG_PATH = ROOT / "logs" / "app.log"
CONTROL_LOG_DIR = ROOT / "logs" / "control"
RECORDINGS_DIR = ROOT / "recordings"
SCREENSHOTS_DIR = ROOT / "screenshots"
OUTPUT_DIR = ROOT / "output"
SAMPLES_DIR = ROOT / "samples"
DATASET_DIR = ROOT / "datasets"
DEMO_DIR = ROOT / "demo"
DEMO_SOURCE_DIR = DEMO_DIR / "source"
PREVIEW_DIR = DEMO_DIR / "preview"
EVENTS_DIR = ROOT / "logs" / "events"
RUNS_DIR = ROOT / "runs"

MODEL_DISPLAY_NAME = "YOLOv4-tiny Bird's-eye Car"


def ensure_dirs() -> None:
    for path in (
        MODEL_DIR,
        ROOT / "config",
        ROOT / "logs" / "control",
        RECORDINGS_DIR,
        SCREENSHOTS_DIR,
        OUTPUT_DIR,
        SAMPLES_DIR,
        DATASET_DIR / "incoming" / "images",
        DATASET_DIR / "incoming" / "labels",
        DATASET_DIR / "uav",
        DEMO_DIR / "screenshots",
        DEMO_DIR / "video",
        DEMO_SOURCE_DIR,
        PREVIEW_DIR,
        EVENTS_DIR,
        RUNS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def model_status() -> tuple[bool, str]:
    missing = []
    if not WEIGHTS_PATH.is_file():
        missing.append(WEIGHTS_PATH.name)
    if not CFG_PATH.is_file():
        missing.append(CFG_PATH.name)
    if not CLASSES_PATH.is_file():
        missing.append(CLASSES_PATH.name)
    if missing:
        return False, "Missing model file(s): " + ", ".join(missing)
    return True, "READY"
