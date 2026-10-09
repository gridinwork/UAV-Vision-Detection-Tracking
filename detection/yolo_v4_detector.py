"""YOLOv4-tiny inference.

The live path uses cv2.dnn_DetectionModel, the same call as detection/detect.py
in the upstream project: 416x416 input, scale 1/255, swapRB, confidence and NMS
applied inside detect(). Boxes come back in the coordinate frame of the image
that was passed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detection.model_manager import ModelManager
from detection.nms import non_max_suppression


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h in full-frame pixels

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


class YoloV4Detector:
    INPUT_SIZE = (416, 416)

    def __init__(self) -> None:
        self.manager = ModelManager()
        self.model: cv2.dnn_DetectionModel | None = None
        self._failed_cuda = False

    @property
    def ready(self) -> bool:
        return self.manager.ready and self.model is not None

    @property
    def backend_label(self) -> str:
        return self.manager.backend_label

    def load(self, device: str = "AUTO") -> str:
        label = self.manager.load(device)
        self._bind_model()
        return label

    def set_device(self, device: str) -> str:
        if not self.manager.ready:
            return self.load(device)
        if device == self.manager.requested_device and self.model is not None and not (
            device == "CUDA" and self.manager.backend_label == "CPU"
        ):
            if device != "CUDA" or self.manager.backend_label.startswith("CUDA"):
                return self.manager.backend_label
        self._failed_cuda = False
        label = self.manager.load(device)
        self._bind_model()
        return label

    def _bind_model(self) -> None:
        self.model = cv2.dnn_DetectionModel(self.manager.net)
        self.model.setInputParams(size=self.INPUT_SIZE, scale=1 / 255, swapRB=True)

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = 0.2,
        nms: float = 0.4,
        detection_scale: float = 1.0,
        count_raw: bool = False,
    ) -> tuple[list[Detection], int]:
        if self.model is None or self.manager.net is None:
            raise RuntimeError("Detector is not loaded.")
        scale = float(detection_scale) if detection_scale else 1.0
        scale = min(1.0, max(0.25, scale))
        if scale < 0.999:
            infer_frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            infer_frame = frame
            scale = 1.0
        try:
            classes, scores, boxes = self.model.detect(
                infer_frame, float(confidence), float(nms)
            )
        except cv2.error:
            if self.manager.backend_label != "CPU" and not self._failed_cuda:
                self._failed_cuda = True
                self.manager.force_cpu()
                self._bind_model()
                classes, scores, boxes = self.model.detect(
                    infer_frame, float(confidence), float(nms)
                )
            else:
                raise
        detections = self._pack(classes, scores, boxes, 1.0 / scale)
        raw_count = len(detections)
        if count_raw:
            raw_count = self._count_raw(infer_frame, confidence)
        return detections, raw_count

    def _pack(self, classes, scores, boxes, inv_scale: float) -> list[Detection]:
        if boxes is None or len(boxes) == 0:
            return []
        boxes_arr = np.array(boxes).reshape(-1, 4)
        scores_arr = np.array(scores).reshape(-1)
        classes_arr = np.array(classes).reshape(-1)
        names = self.manager.class_names or ["car"]
        found: list[Detection] = []
        for class_id, score, box in zip(classes_arr, scores_arr, boxes_arr):
            x, y, w, h = [int(round(float(v) * inv_scale)) for v in box]
            if w < 2 or h < 2:
                continue
            cid = int(class_id)
            name = names[cid] if 0 <= cid < len(names) else "car"
            found.append(Detection(cid, name, float(score), (x, y, w, h)))
        found.sort(key=lambda item: item.confidence, reverse=True)
        return found

    def _count_raw(self, frame: np.ndarray, confidence: float) -> int:
        """Count pre-NMS candidates. Used for the debug panel only."""
        net = self.manager.net
        assert net is not None
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, self.INPUT_SIZE, swapRB=True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(self.manager.output_names)
        raw_boxes: list[list[int]] = []
        raw_confs: list[float] = []
        height, width = frame.shape[:2]
        for output in outputs:
            for det in output:
                scores = det[5:]
                if scores.size == 0:
                    continue
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf < confidence:
                    continue
                w = int(det[2] * width)
                h = int(det[3] * height)
                x = int(det[0] * width - w / 2)
                y = int(det[1] * height - h / 2)
                raw_boxes.append([x, y, w, h])
                raw_confs.append(conf)
        # Returning the pre-NMS candidate count. NMS itself is applied by DetectionModel.
        _ = non_max_suppression(raw_boxes, raw_confs, confidence, 0.4)
        return len(raw_boxes)
