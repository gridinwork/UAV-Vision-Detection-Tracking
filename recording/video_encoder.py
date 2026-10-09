"""Open a VideoWriter, preferring H.264 and falling back to mp4v."""

from __future__ import annotations

import os
from contextlib import contextmanager

import cv2


@contextmanager
def _quiet_stderr():
    """OpenCV prints a missing-OpenH264 warning while probing codecs."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def open_writer(path: str, fps: float, size: tuple[int, int]):
    fps = float(fps) if fps and fps > 1 else 30.0
    width, height = int(size[0]), int(size[1])
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    last_error = "no codec opened"
    for code in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*code)
        with _quiet_stderr():
            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, code, (width, height), fps
        writer.release()
        last_error = f"{code} did not open"
    raise RuntimeError(f"Could not open a video writer for {path}: {last_error}")
