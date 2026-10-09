"""Non-maximum suppression matching the original confidence / NMS thresholds."""

from __future__ import annotations

import cv2
import numpy as np


def non_max_suppression(
    boxes: list[list[int]],
    confidences: list[float],
    confidence_threshold: float,
    nms_threshold: float,
) -> list[int]:
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if indices is None or len(indices) == 0:
        return []
    return [int(i) for i in np.array(indices).reshape(-1)]
