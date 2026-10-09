"""Class policy for UAV and vehicle detectors.

Ordinary COCO checkpoints are never treated as a UAV detector. A detection is
kept only when its class name maps to one of the names the program knows.
"""

from __future__ import annotations

import re


CANONICAL = {
    "drone": "drone",
    "uav": "drone",
    "quadcopter": "quadcopter",
    "quad copter": "quadcopter",
    "fpv": "FPV drone",
    "fpv drone": "FPV drone",
    "fixed wing": "fixed-wing UAV",
    "fixed wing uav": "fixed-wing UAV",
    "fixedwing": "fixed-wing UAV",
    "bird": "bird",
    "aircraft": "aircraft",
    "airplane": "aircraft",
    "aeroplane": "aircraft",
    "plane": "aircraft",
    "car": "car",
}

UAV_NAMES = {"drone", "quadcopter", "FPV drone", "fixed-wing UAV"}

COCO_MARKERS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "chair",
    "couch", "bed", "toilet", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "toothbrush",
}

COCO_FILE = re.compile(
    r"(^|[\\/])(yolov5|yolov8|yolov9|yolov10|yolo11|yolo12)[nsmxl](-cls|-seg|-pose|-obb)?\.pt$",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    text = name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def canonical_class(name: str) -> str | None:
    return CANONICAL.get(normalize_name(name))


def class_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    return []


def classify_weights(names, filename: str = "") -> tuple[str, str]:
    """Return (kind, warning_code).

    kind is ``uav_custom``, ``coco_general``, ``custom_filtered`` or ``unknown``.
    warning_code is empty when there is nothing to warn about.
    """
    labels = class_names(names)
    normalized = [normalize_name(item) for item in labels]
    has_uav = any(canonical_class(item) in UAV_NAMES for item in labels)
    coco_hits = sum(1 for item in normalized if item in COCO_MARKERS)
    coco_file = bool(COCO_FILE.search(filename.replace("\\", "/")))
    if has_uav:
        return "uav_custom", ""
    if not labels:
        if coco_file:
            return "coco_general", "coco_not_uav"
        return "unknown", "no_class_names"
    if coco_hits >= 8 or coco_file:
        return "coco_general", "coco_not_uav"
    return "custom_filtered", "filtered_classes"


def allowed_indices(names, vehicle_only: bool = False) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if isinstance(names, dict):
        pairs = [(int(key), str(value)) for key, value in names.items()]
    else:
        pairs = list(enumerate(class_names(names)))
    for index, name in pairs:
        canonical = canonical_class(name)
        if canonical is None:
            continue
        if vehicle_only and canonical != "car":
            continue
        mapping[index] = canonical
    return mapping
