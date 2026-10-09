"""Count detections on a video. Scores are computed only from real label files."""

from __future__ import annotations

from pathlib import Path

import cv2


def evaluate_video(path: str, detect_fn, class_names: list[str] | None = None, labels_dir: str | None = None, max_frames: int | None = None) -> dict:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(path)
    counts: dict[str, int] = {}
    frames = 0
    frames_with = 0
    matched = 0
    pred_total = 0
    gt_total = 0
    confusion: dict[str, int] = {}
    labels = Path(labels_dir) if labels_dir else None
    names = list(class_names or [])
    while True:
        if max_frames is not None and frames >= max_frames:
            break
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        detections = list(detect_fn(frame) or [])
        frames += 1
        if detections:
            frames_with += 1
        for detection in detections:
            name = str(getattr(detection, "class_name", None) or detection.get("class_name"))
            counts[name] = counts.get(name, 0) + 1
        pred_total += len(detections)
        label_path = None
        if labels is not None:
            label_path = labels / f"frame_{frames - 1:06d}.txt"
            if not label_path.is_file():
                label_path = None
        if label_path is not None:
            gt_boxes = _read_label(label_path, frame.shape[1], frame.shape[0], names)
            gt_total += len(gt_boxes)
            used = set()
            for detection in detections:
                box = getattr(detection, "box", None) or detection.get("box")
                pred_name = str(getattr(detection, "class_name", None) or detection.get("class_name"))
                best_iou = 0.0
                best_index = None
                for index, (gt_name, gt_box) in enumerate(gt_boxes):
                    if index in used:
                        continue
                    score = _iou(box, gt_box)
                    if score > best_iou:
                        best_iou = score
                        best_index = index
                if best_index is not None and best_iou >= 0.5:
                    used.add(best_index)
                    matched += 1
                    gt_name = gt_boxes[best_index][0]
                    key = f"{gt_name}->{pred_name}"
                    confusion[key] = confusion.get(key, 0) + 1
    capture.release()
    report = {
        "video": str(path),
        "frames": frames,
        "frames_with_detections": frames_with,
        "detections": pred_total,
        "counts_by_class": counts,
        "note": "Это количества детекций. Проценты точности не подставляются.",
    }
    if labels is not None and gt_total > 0:
        precision = matched / pred_total if pred_total else 0.0
        recall = matched / gt_total if gt_total else 0.0
        report["measured"] = {
            "precision_iou50": precision,
            "recall_iou50": recall,
            "matched": matched,
            "predictions": pred_total,
            "ground_truth": gt_total,
            "confusion_pairs": confusion,
            "source": str(labels),
        }
    return report


def _read_label(path: Path, width: int, height: int, names: list[str]):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
        x = int((cx - bw / 2.0) * width)
        y = int((cy - bh / 2.0) * height)
        boxes.append((name, (x, y, int(bw * width), int(bh * height))))
    return boxes


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union
