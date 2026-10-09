"""Synthetic scenes for the marked DEMO mode.

Detections returned here are scripted. They are not the output of a neural
network and must be displayed with the DEMO / SIMULATED DATA banner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detection.yolo_v4_detector import Detection


@dataclass
class Actor:
    class_name: str
    kind: str
    cx: int
    cy: int
    scale: float
    emit: bool = True


class Scene:
    key = "scene"
    seconds = 20

    def actors(self, index: int, width: int, height: int, fps: float) -> list[Actor]:
        return []

    def frame(self, index: int, width: int, height: int, fps: float) -> np.ndarray:
        image = sky(width, height, index / max(fps, 1.0))
        self._decorate(image, index, fps)
        pending = []
        for actor in self.actors(index, width, height, fps):
            if actor.kind == "occluder":
                pending.append(actor)
                continue
            draw_actor(image, actor)
        for actor in pending:
            draw_actor(image, actor)
        cv2.putText(
            image,
            "SIMULATED SCENE",
            (16, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        return image

    def detections(self, index: int, width: int, height: int, fps: float) -> list[Detection]:
        found = []
        for actor in self.actors(index, width, height, fps):
            if not actor.emit or actor.kind == "occluder":
                continue
            found.append(
                Detection(0, actor.class_name, 0.93, box_of(actor.cx, actor.cy, actor.scale, actor.kind))
            )
        return found

    def _decorate(self, image, index: int, fps: float) -> None:
        return None


class UavTracking(Scene):
    key = "uav_tracking"

    def actors(self, index, width, height, fps):
        t = index / fps
        cx = int(90 + (width - 180) * ((t % 16.0) / 16.0))
        cy = int(height * 0.42 + math.sin(t * 1.4) * 36)
        return [Actor("drone", "quad", cx, cy, 1.15)]


class FpvDrone(Scene):
    key = "fpv_drone"

    def actors(self, index, width, height, fps):
        t = index / fps
        cx = int(width * 0.5 + math.sin(t * 0.7) * width * 0.22)
        cy = int(height * 0.58 + math.cos(t * 0.9) * 30)
        return [Actor("FPV drone", "quad", cx, cy, 2.4)]


class MultipleDrones(Scene):
    key = "multiple_drones"

    def actors(self, index, width, height, fps):
        t = index / fps
        lanes = (0.28, 0.48, 0.68)
        names = ("drone", "quadcopter", "fixed-wing UAV")
        kinds = ("quad", "quad", "wing")
        actors = []
        for lane, name, kind, phase in zip(lanes, names, kinds, (0.0, 2.0, 4.0)):
            progress = ((t * 0.08) + phase / 16.0) % 1.0
            cx = int(70 + (width - 140) * progress)
            cy = int(height * lane + math.sin(t * 1.1 + phase) * 16)
            actors.append(Actor(name, kind, cx, cy, 1.0))
        return actors


class BirdVsDrone(Scene):
    key = "bird_vs_drone"

    def actors(self, index, width, height, fps):
        t = index / fps
        bird_x = int(width * 0.2 + (t % 14.0) / 14.0 * width * 0.6)
        bird_y = int(height * 0.36 + math.sin(t * 3.0) * 28)
        drone_x = int(width * 0.75 - (t % 12.0) / 12.0 * width * 0.4)
        drone_y = int(height * 0.62 + math.sin(t * 1.2) * 18)
        return [
            Actor("bird", "bird", bird_x, bird_y, 1.2),
            Actor("drone", "quad", drone_x, drone_y, 1.15),
        ]


class SmallObject(Scene):
    key = "small_object"

    def actors(self, index, width, height, fps):
        t = index / fps
        cx = int(width * 0.3 + (t % 18.0) / 18.0 * width * 0.4)
        cy = int(height * 0.22 + math.sin(t) * 10)
        return [Actor("drone", "quad", cx, cy, 0.42)]


class Glare(Scene):
    key = "glare"

    def actors(self, index, width, height, fps):
        t = index / fps
        cx = int(width * 0.62 + math.sin(t * 0.8) * 40)
        cy = int(height * 0.34 + math.cos(t) * 16)
        return [Actor("drone", "quad", cx, cy, 1.0)]

    def _decorate(self, image, index, fps):
        overlay = image.copy()
        h, w = image.shape[:2]
        cv2.circle(overlay, (int(w * 0.68), int(h * 0.28)), int(min(w, h) * 0.16), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)


class Clouds(Scene):
    key = "clouds"

    def actors(self, index, width, height, fps):
        t = index / fps
        cx = int(80 + (width - 160) * ((t % 15.0) / 15.0))
        cy = int(height * 0.4)
        return [Actor("drone", "quad", cx, cy, 1.0)]

    def _decorate(self, image, index, fps):
        h, w = image.shape[:2]
        t = index / max(fps, 1.0)
        for offset, cy, rx, ry in ((0, 0.3, 180, 46), (240, 0.48, 220, 54), (480, 0.22, 140, 36)):
            cx = int((offset + t * 40) % (w + 300) - 150)
            cv2.ellipse(image, (cx, int(h * cy)), (rx, ry), 0, 0, 360, (190, 185, 180), -1)


class Industrial(Scene):
    key = "industrial"

    def actors(self, index, width, height, fps):
        t = index / fps
        cx = int(100 + (width - 200) * ((t % 12.0) / 12.0))
        cy = int(height * 0.46)
        return [Actor("drone", "quad", cx, cy, 1.05)]

    def _decorate(self, image, index, fps):
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, int(h * 0.62)), (w, h), (70, 72, 74), -1)
        for x in range(40, w, 180):
            cv2.rectangle(image, (x, int(h * 0.42)), (x + 70, int(h * 0.62)), (50, 52, 58), -1)
            cv2.rectangle(image, (x + 24, int(h * 0.28)), (x + 46, int(h * 0.42)), (40, 42, 48), -1)


class Occlusion(Scene):
    key = "occlusion"

    def actors(self, index, width, height, fps):
        t = index / fps
        cx = int(40 + (width - 80) * ((t % 8.0) / 8.0))
        cy = int(height * 0.48)
        bx, by, bw, bh = self._building(width, height)
        hidden = bx <= cx <= bx + bw and by <= cy <= by + bh
        return [
            Actor("drone", "quad", cx, cy, 1.1, emit=not hidden),
            Actor("building", "occluder", bx + bw // 2, by + bh // 2, 1.0, emit=False),
        ]

    def _building(self, width, height):
        bw, bh = 150, 230
        return width // 2 - bw // 2, height // 2 - bh // 2, bw, bh


class Birds(Scene):
    key = "birds"

    def actors(self, index, width, height, fps):
        t = index / fps
        actors = []
        for number, phase in enumerate((0.0, 1.4, 2.6)):
            cx = int(60 + ((t * 0.09 + phase / 10) % 1.0) * (width - 120))
            cy = int(height * (0.3 + 0.08 * number) + math.sin(t * 2.5 + phase) * 20)
            actors.append(Actor("bird", "bird", cx, cy, 1.0))
        dx = int(width * 0.5 + math.sin(t * 0.6) * 80)
        actors.append(Actor("drone", "quad", dx, int(height * 0.7), 1.1))
        return actors


SCENES: dict[str, Scene] = {
    scene.key: scene
    for scene in (
        UavTracking(),
        FpvDrone(),
        MultipleDrones(),
        BirdVsDrone(),
        SmallObject(),
        Glare(),
        Clouds(),
        Industrial(),
        Occlusion(),
        Birds(),
    )
}


def scene_by_key(key: str) -> Scene | None:
    return SCENES.get(key)


def scene_key_from_path(path: str) -> str:
    stem = Path(path).stem.lower()
    for key in sorted(SCENES, key=len, reverse=True):
        if key in stem:
            return key
    return ""


def sky(width: int, height: int, t: float) -> np.ndarray:
    column = np.linspace(0.0, 1.0, height)[:, None]
    blue = (28 + 55 * column).astype(np.uint8)
    green = (36 + 28 * column).astype(np.uint8)
    red = (22 + 12 * column).astype(np.uint8)
    image = np.dstack([np.repeat(blue, width, 1), np.repeat(green, width, 1), np.repeat(red, width, 1)])
    for offset, cy, rx, ry in ((80, 0.18, 90, 24), (360, 0.3, 120, 28)):
        cx = int((offset + t * 25) % (width + 200) - 80)
        cv2.ellipse(image, (cx, int(height * cy)), (rx, ry), 0, 0, 360, (160, 150, 145), -1)
    return image


def box_of(cx: int, cy: int, scale: float, kind: str) -> tuple[int, int, int, int]:
    if kind == "wing":
        w, h = int(78 * scale), int(28 * scale)
    elif kind == "bird":
        w, h = int(54 * scale), int(24 * scale)
    elif kind == "car":
        w, h = int(64 * scale), int(28 * scale)
    else:
        w = h = int(58 * scale)
    w = max(8, w)
    h = max(8, h)
    return int(cx - w / 2), int(cy - h / 2), w, h


def draw_actor(image, actor: Actor) -> None:
    if actor.kind == "occluder":
        h, w = image.shape[:2]
        bw, bh = 150, 230
        x, y = w // 2 - bw // 2, h // 2 - bh // 2
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (42, 46, 52), -1)
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (90, 96, 104), 2)
        return
    x, y, bw, bh = box_of(actor.cx, actor.cy, actor.scale, actor.kind)
    cx, cy = actor.cx, actor.cy
    color = {
        "drone": (60, 220, 255),
        "quadcopter": (80, 180, 255),
        "FPV drone": (40, 140, 255),
        "fixed-wing UAV": (255, 180, 80),
        "bird": (80, 200, 120),
        "car": (200, 200, 210),
    }.get(actor.class_name, (60, 220, 255))
    if actor.kind == "quad":
        arm = max(8, int(min(bw, bh) * 0.38))
        cv2.line(image, (cx - arm, cy - arm), (cx + arm, cy + arm), color, 2, cv2.LINE_AA)
        cv2.line(image, (cx - arm, cy + arm), (cx + arm, cy - arm), color, 2, cv2.LINE_AA)
        for dx, dy in ((-arm, -arm), (arm, -arm), (-arm, arm), (arm, arm)):
            cv2.circle(image, (cx + dx, cy + dy), max(3, arm // 4), color, 2, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), max(3, arm // 5), color, -1, cv2.LINE_AA)
    elif actor.kind == "wing":
        pts = np.array([[cx, cy - bh // 3], [cx + bw // 2, cy], [cx, cy + bh // 5], [cx - bw // 2, cy]], np.int32)
        cv2.fillPoly(image, [pts], color)
    elif actor.kind == "bird":
        cv2.ellipse(image, (cx - bw // 5, cy), (bw // 3, max(4, bh // 3)), 20, 200, 340, color, 2, cv2.LINE_AA)
        cv2.ellipse(image, (cx + bw // 5, cy), (bw // 3, max(4, bh // 3)), -20, 200, 340, color, 2, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (x, y), (x + bw, y + bh), color, -1)


def write_scene(path: str | Path, key: str, seconds: float = 20.0, fps: float = 25.0, size: tuple[int, int] = (1280, 720)) -> Path:
    scene = SCENES.get(key)
    if scene is None:
        raise KeyError(key)
    from recording.video_encoder import open_writer

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    count = max(1, int(round(seconds * fps)))
    writer, _codec, (width, height), fps = open_writer(str(destination), fps, (width, height))
    try:
        for index in range(count):
            frame = scene.frame(index, width, height, fps)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return destination


def write_demo_set(directory: str | Path, seconds: float = 20.0) -> list[Path]:
    root = Path(directory)
    keys = ("uav_tracking", "fpv_drone", "multiple_drones", "bird_vs_drone", "small_object", "glare", "clouds", "industrial", "occlusion", "birds")
    return [write_scene(root / f"demo_{key}.mp4", key, seconds=seconds) for key in keys]
