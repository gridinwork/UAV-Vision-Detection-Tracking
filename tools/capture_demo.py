"""Build marked demo videos and grab screenshots from the live window."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from recording.video_encoder import open_writer
from simulation.scenarios import scene_by_key, write_scene
from tracking.engine import PipelineConfig, VisionEngine
from utils.paths import DEMO_DIR, DEMO_SOURCE_DIR, ensure_dirs


SHOWCASE = (
    ("uav_tracking", "uav_tracking.png", "uav_tracking_20s.mp4"),
    ("fpv_drone", "fpv_drone_tracking.png", "fpv_drone_20s.mp4"),
    ("multiple_drones", "multiple_drones.png", "multiple_drones_20s.mp4"),
    ("bird_vs_drone", "bird_vs_drone.png", "bird_vs_drone_20s.mp4"),
)


def render_annotated(source: Path, destination: Path, scene_key: str) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    engine = VisionEngine()
    config = PipelineConfig(
        mission="uav",
        demo=True,
        demo_scene=scene_key,
        algorithm="BYTETRACK",
        confidence=0.35,
        nms=0.45,
        lost_timeout=1.5,
        source_fps=fps,
        proc_fps=fps,
        source_uri=str(source),
        overlays={"boxes": True, "confidence": True, "ids": True, "trail": True, "status": True, "speed": True, "fps": True},
    )
    writer = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            config.frame_index = index
            view, _info = engine.step(frame, config)
            if writer is None:
                writer, _codec, size, _fps = open_writer(str(destination), fps, (view.shape[1], view.shape[0]))
            if view.shape[1] != size[0] or view.shape[0] != size[1]:
                view = cv2.resize(view, size)
            writer.write(view)
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    print(f"annotated {destination.name} frames {index}")


def build_media() -> None:
    ensure_dirs()
    video_dir = DEMO_DIR / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    for key, _shot, video_name in SHOWCASE:
        source = DEMO_SOURCE_DIR / f"demo_{key}.mp4"
        if not source.is_file() or source.stat().st_size < 1000:
            print("writing", source.name)
            write_scene(source, key, seconds=20, fps=25, size=(1280, 720))
        render_annotated(source, video_dir / video_name, key)


def grab_gui() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.theme import DARK_QSS
    from app.uav_window import UavWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    window = UavWindow()
    window.resize(1480, 920)
    window.show()
    app.processEvents()
    shot_dir = DEMO_DIR / "screenshots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    for key, shot_name, _video in SHOWCASE:
        index = window.demo_scene.findData(key)
        if index >= 0:
            window.demo_scene.setCurrentIndex(index)
        window._open_demo()
        deadline = time.time() + 30
        ready = False
        saw_restart = False
        while time.time() < deadline:
            app.processEvents()
            info = window.last_info or {}
            name = window.file_info.text()
            frame_id = int(info.get("frame_id") or 0)
            if f"demo_{key}" in name and frame_id < 8:
                saw_restart = True
            if (
                saw_restart
                and f"demo_{key}" in name
                and frame_id > 40
                and info.get("status") in ("TRACKING", "REACQUIRED")
                and float(info.get("proc_fps") or 0) > 1.0
            ):
                ready = True
                break
            time.sleep(0.03)
        app.processEvents()
        image = window.grab()
        target = shot_dir / shot_name
        image.save(str(target))
        print("screenshot", shot_name, "ready" if ready else "TIMEOUT", "bytes", target.stat().st_size if target.is_file() else 0)
    window.close()
    app.processEvents()


def main() -> int:
    build_media()
    grab_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
