"""Dataset import, split and measured-metric loading.

Training metrics are read from an Ultralytics ``results.csv`` after a real
run. Missing values stay empty; this module does not invent precision or mAP.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def import_images(sources: list[str], incoming: Path) -> int:
    folder = incoming / "images"
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sources:
        path = Path(source)
        files = [path] if path.is_file() else list(path.glob("*"))
        for file_path in files:
            if file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            shutil.copy2(file_path, folder / file_path.name)
            count += 1
    return count


def import_labels(sources: list[str], incoming: Path) -> int:
    folder = incoming / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sources:
        path = Path(source)
        files = [path] if path.is_file() else list(path.glob("*.txt"))
        for file_path in files:
            if file_path.suffix.lower() != ".txt" or file_path.name == "classes.txt":
                if file_path.name == "classes.txt":
                    shutil.copy2(file_path, incoming / "classes.txt")
                continue
            shutil.copy2(file_path, folder / file_path.name)
            count += 1
    return count


def import_video(video_path: str, incoming: Path, every_n: int = 15) -> int:
    folder = incoming / "images"
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    stem = Path(video_path).stem
    saved = 0
    index = 0
    step = max(1, int(every_n))
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if index % step == 0:
            cv2.imwrite(str(folder / f"{stem}_{index:06d}.jpg"), frame)
            saved += 1
        index += 1
    capture.release()
    return saved


def prepare_split(incoming: Path, dataset: Path, val_ratio: float = 0.2, seed: int = 1) -> dict:
    images = sorted((incoming / "images").glob("*"))
    images = [path for path in images if path.suffix.lower() in IMAGE_SUFFIXES]
    if len(images) < 2:
        raise RuntimeError("Для разбиения нужны хотя бы два изображения.")
    labels = incoming / "labels"
    ratio = min(0.5, max(0.05, float(val_ratio)))
    order = images[:]
    random.Random(seed).shuffle(order)
    val_count = max(1, int(round(len(order) * ratio)))
    if val_count >= len(order):
        val_count = len(order) - 1
    val_set = set(order[:val_count])
    names = _class_names(incoming, labels)
    groups = (
        ("train", [path for path in order if path not in val_set]),
        ("val", [path for path in order if path in val_set]),
    )
    for split, chosen in groups:
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in chosen:
            shutil.copy2(image, image_dir / image.name)
            label = labels / f"{image.stem}.txt"
            target = label_dir / f"{image.stem}.txt"
            if label.is_file():
                shutil.copy2(label, target)
            else:
                target.write_text("", encoding="utf-8")
    yaml_path = dataset / "data.yaml"
    lines = [
        f"path: {dataset.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "images": len(images),
        "train": len(order) - val_count,
        "val": val_count,
        "yaml": str(yaml_path),
        "classes": names,
    }


def _class_names(incoming: Path, labels: Path) -> list[str]:
    classes = incoming / "classes.txt"
    if classes.is_file():
        names = [line.strip() for line in classes.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    highest = -1
    if labels.is_dir():
        for label in labels.glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    highest = max(highest, int(float(parts[0])))
                except ValueError:
                    continue
    if highest < 0:
        return ["object"]
    return [f"class_{index}" for index in range(highest + 1)]


def load_measured_metrics(run_dir: str | Path) -> dict | None:
    """Read the last row of a real Ultralytics results.csv. Returns None if absent."""
    directory = Path(run_dir)
    csv_path = directory / "results.csv"
    if not csv_path.is_file():
        return None
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    last = {(key or "").strip(): (value or "").strip() for key, value in rows[-1].items()}

    def pick(*needles: str) -> float | None:
        for key, value in last.items():
            compact = key.replace(" ", "")
            if any(needle in compact for needle in needles):
                try:
                    number = float(value)
                except ValueError:
                    continue
                if number == number:
                    return number
        return None

    matrix = None
    for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        candidate = directory / name
        if candidate.is_file():
            matrix = str(candidate)
            break
    return {
        "precision": pick("metrics/precision(B)", "precision"),
        "recall": pick("metrics/recall(B)", "recall"),
        "map50": pick("metrics/mAP50(B)", "mAP50(B)"),
        "map50_95": pick("metrics/mAP50-95(B)", "mAP50-95(B)"),
        "confusion_matrix": matrix,
        "source": str(csv_path),
        "epoch": last.get("epoch") or None,
    }


def run_training(data_yaml: str, base_weights: str, epochs: int, imgsz: int, device: str, project: str, name: str = "uav") -> dict | None:
    weights = Path(base_weights)
    if not weights.is_file():
        raise RuntimeError(
            "Нужен локальный файл весов .pt. Приложение не скачивает COCO-чекпоинт "
            "и не считает его специализированной моделью БПЛА."
        )
    if weights.suffix.lower() != ".pt":
        raise RuntimeError("Обучение Ultralytics запускается из файла .pt.")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Пакет ultralytics не установлен. Обучение недоступно.") from exc
    from detection.device_select import resolve_device

    device_arg, _label = resolve_device(device)
    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=max(1, int(epochs)),
        imgsz=int(imgsz),
        device=device_arg,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    return load_measured_metrics(Path(project) / name)
