"""Load the shipped Darknet weights and configure the DNN backend."""

from __future__ import annotations

import cv2
import numpy as np

from utils.logger import get_logger
from utils.paths import CFG_PATH, CLASSES_PATH, MODEL_DISPLAY_NAME, WEIGHTS_PATH, model_status


log = get_logger()


class ModelError(RuntimeError):
    pass


class ModelManager:
    def __init__(self) -> None:
        self.net = None
        self.output_names: list[str] = []
        self.class_names: list[str] = []
        self.backend_label = "CPU"
        self.requested_device = "AUTO"
        self.ready = False
        self.status_text = "NOT LOADED"
        self.display_name = MODEL_DISPLAY_NAME

    def load(self, device: str = "AUTO") -> str:
        ok, message = model_status()
        if not ok:
            self.ready = False
            self.status_text = message
            raise ModelError(message)
        self.class_names = [
            line.strip()
            for line in CLASSES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.class_names:
            raise ModelError("Class file is empty: " + CLASSES_PATH.name)
        self.net = cv2.dnn.readNet(str(WEIGHTS_PATH), str(CFG_PATH))
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        indices = np.array(unconnected).reshape(-1)
        self.output_names = [layer_names[int(i) - 1] for i in indices]
        self.requested_device = device
        self.backend_label = self._apply_backend(device)
        self.ready = True
        self.status_text = "READY"
        log.info(
            "Detector loaded | %s | backend %s | classes %s",
            self.display_name,
            self.backend_label,
            ",".join(self.class_names),
        )
        return self.backend_label

    def _apply_backend(self, device: str) -> str:
        assert self.net is not None
        preference = (device or "AUTO").upper()
        if preference in ("AUTO", "CUDA"):
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                self._warmup()
                log.info("CUDA FP16 backend accepted")
                return "CUDA FP16"
            except Exception as exc:
                log.info("CUDA backend is not available. Using CPU. (%s)", type(exc).__name__)
                if preference == "CUDA":
                    log.info("Requested CUDA could not run. Using CPU.")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._warmup()
        return "CPU"

    def _warmup(self) -> None:
        assert self.net is not None
        blank = np.zeros((416, 416, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(blank, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        self.net.forward(self.output_names)

    def force_cpu(self) -> str:
        assert self.net is not None
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._warmup()
        self.backend_label = "CPU"
        log.info("Detector backend forced to CPU")
        return self.backend_label
