"""Ultralytics YOLOv8 / ONNX backend.

Weights are loaded only from a local ``.pt`` or ``.onnx`` file. The loader
never downloads a checkpoint and never describes COCO weights as a UAV model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from detection.class_catalog import allowed_indices, classify_weights
from detection.device_select import resolve_device
from detection.model_manager import ModelError
from detection.yolo_v4_detector import Detection
from utils.logger import get_logger


log = get_logger()


class YoloV8Detector:
    def __init__(self, loader=None) -> None:
        self._loader = loader
        self.model = None
        self.path = ""
        self.kind = "unknown"
        self.warning_code = ""
        self.device: object = "cpu"
        self.backend_label = "CPU"
        self.ready = False
        self.display_name = "YOLOv8"
        self.class_map: dict[int, str] = {}
        self._vehicle_only = False

    def load(self, path: str, device: str = "AUTO", vehicle_only: bool = False) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            self.ready = False
            self.model = None
            raise ModelError(f"Файл весов не найден: {path}")
        if file_path.suffix.lower() not in {".pt", ".onnx"}:
            self.ready = False
            raise ModelError("Поддерживаются только локальные файлы .pt и .onnx.")
        loader = self._loader
        if loader is None:
            try:
                from ultralytics import YOLO
            except Exception as exc:
                raise ModelError(
                    "Пакет ultralytics не установлен. Детектор YOLOv8 недоступен, "
                    "пока не выполнен install.bat или pip install ultralytics."
                ) from exc
            loader = YOLO
        self.model = loader(str(file_path))
        names = getattr(self.model, "names", {}) or {}
        self.kind, self.warning_code = classify_weights(names, file_path.name)
        self._vehicle_only = vehicle_only
        self.class_map = allowed_indices(names, vehicle_only=vehicle_only)
        self.path = str(file_path)
        self.device, self.backend_label = resolve_device(device)
        self.display_name = self._title(file_path.name)
        self.ready = True
        log.info(
            "YOLOv8 loaded | %s | kind %s | backend %s | classes %s",
            file_path.name,
            self.kind,
            self.backend_label,
            ",".join(sorted(set(self.class_map.values()))) or "-",
        )
        if self.warning_code:
            log.info("YOLOv8 warning | %s", self.warning_code)
        return self.backend_label

    def set_device(self, device: str) -> str:
        self.device, self.backend_label = resolve_device(device)
        return self.backend_label

    def detect(
        self,
        frame: np.ndarray,
        confidence: float,
        nms: float,
        imgsz: int,
        class_filter: set[str] | None = None,
    ) -> list[Detection]:
        if self.model is None:
            raise ModelError("Веса YOLOv8 не загружены.")
        results = self._predict(frame, confidence, nms, imgsz)
        return self._pack(results, class_filter)

    def _predict(self, frame, confidence: float, nms: float, imgsz: int):
        kwargs = {
            "conf": float(confidence),
            "iou": float(nms),
            "imgsz": int(imgsz),
            "device": self.device,
            "verbose": False,
        }
        try:
            return self.model.predict(frame, **kwargs)
        except Exception as exc:
            if self.device == "cpu":
                raise
            log.info("CUDA inference failed (%s). Switching to CPU.", exc)
            self.device = "cpu"
            self.backend_label = "CPU (fallback)"
            kwargs["device"] = "cpu"
            return self.model.predict(frame, **kwargs)

    def _pack(self, results, class_filter: set[str] | None) -> list[Detection]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            return []
        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach().cpu().numpy()
        else:
            xyxy = np.asarray(xyxy)
        confs = getattr(boxes, "conf", None)
        clss = getattr(boxes, "cls", None)
        if confs is None or clss is None or len(xyxy) == 0:
            return []
        if hasattr(confs, "detach"):
            confs = confs.detach().cpu().numpy()
            clss = clss.detach().cpu().numpy()
        else:
            confs = np.asarray(confs)
            clss = np.asarray(clss)
        found: list[Detection] = []
        for box, score, class_id in zip(xyxy, confs, clss):
            index = int(class_id)
            name = self.class_map.get(index)
            if name is None:
                continue
            if class_filter is not None and name not in class_filter:
                continue
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            width = max(1, int(round(x2 - x1)))
            height = max(1, int(round(y2 - y1)))
            found.append(
                Detection(
                    index,
                    name,
                    float(score),
                    (int(round(x1)), int(round(y1)), width, height),
                )
            )
        return found

    def _title(self, filename: str) -> str:
        if self.kind == "uav_custom":
            return f"YOLOv8 UAV ({filename})"
        if self.kind == "coco_general":
            return f"YOLOv8 COCO ({filename}) — не модель БПЛА"
        if self.kind == "unknown":
            return f"YOLOv8 ({filename}) — классы не прочитаны"
        return f"YOLOv8 ({filename})"
