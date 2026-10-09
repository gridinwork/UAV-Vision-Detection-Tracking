"""OpenCV DNN backend probe. CUDA is optional and never required."""

from __future__ import annotations

import cv2


def cuda_device_count() -> int:
    try:
        if hasattr(cv2, "cuda"):
            return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0
    return 0


def cuda_build_info() -> str:
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return ""
    for line in info.splitlines():
        if "CUDA" in line or "NVIDIA" in line or "cuDNN" in line:
            return line.strip()
    return ""
