"""Autonomous VTOL Vehicle Tracking Studio."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from app.uav_window import UavWindow
from app.theme import DARK_QSS
from utils.logger import get_logger
from utils.paths import ensure_dirs


def main() -> int:
    ensure_dirs()
    log = get_logger()
    log.info("Application Start")
    log.info("Python version %s", sys.version.replace("\n", " "))
    try:
        import cv2

        log.info("OpenCV version %s", cv2.__version__)
        if int(cv2.__version__.split(".")[0]) >= 5:
            log.info("OpenCV 5 cannot load Darknet weights. Install opencv-python 4.x.")
    except Exception as exc:
        log.info("OpenCV import failed: %s", exc)
    app = QApplication(sys.argv)
    app.setApplicationName("UAV Vision Detection & Tracking")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    window = UavWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
