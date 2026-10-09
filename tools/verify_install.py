"""Core import check. Missing model weights are reported and do not fail install."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from utils.paths import ensure_dirs, model_status


def main() -> int:
    ensure_dirs()
    print("Python", sys.version.split()[0])
    print("OpenCV", cv2.__version__)
    if int(cv2.__version__.split(".")[0]) != 4:
        print("OpenCV 4.x is required.")
        return 1
    try:
        import PySide6

        print("PySide6", PySide6.__version__)
    except Exception as exc:
        print("PySide6 import failed:", exc)
        return 1
    try:
        import ultralytics

        print("ultralytics", getattr(ultralytics, "__version__", "installed"))
    except Exception:
        print("ultralytics is not installed. UAV weights cannot be loaded until it is.")
    ok, message = model_status()
    print("YOLOv4-tiny:", message if ok else message + " (vehicle mode waits for these files)")
    print("Core install check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
