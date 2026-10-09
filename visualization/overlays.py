"""Text blocks: command, offsets, debug, FPS."""

from __future__ import annotations

import cv2


def draw_text_block(frame, lines: list[str], origin: tuple[int, int], accent=(230, 236, 242)) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + 16
    line_h = 18
    height = line_h * len(lines) + 10
    x, y = origin
    x = max(8, min(x, frame.shape[1] - width - 8))
    y = max(8, min(y, frame.shape[0] - height - 8))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (10, 14, 20), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 70, 90), 1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 8, y + 16 + index * line_h),
            font,
            scale,
            accent,
            thickness,
            cv2.LINE_AA,
        )


def command_lines(state, packet, backend: str, show_ned: bool) -> list[str]:
    conf = "—" if state.confidence is None else f"{state.confidence * 100:.0f}%"
    target = state.target_label or "—"
    locked = state.target_id is not None and state.status not in ("LOST", "SEARCHING", "DETECTED")
    lines = [
        "LOCK: TARGET LOCKED" if locked else "LOCK: NONE",
        f"TARGET: {target}",
        f"CONF: {conf}",
        f"STATUS: {state.status}",
        f"COMMAND: {packet['command']}",
        f"SPEED: {packet['speed']}",
        f"X OFFSET: {packet['raw_offset_x']:+.0f} px",
        f"Y OFFSET: {packet['raw_offset_y']:+.0f} px",
        f"X: {packet['x_norm']:+.2f}   Y: {packet['y_norm']:+.2f}",
    ]
    if show_ned or True:
        lines.append(f"VIRTUAL N: {packet['virtual_n']:+.1f}")
        lines.append(f"VIRTUAL E: {packet['virtual_e']:+.1f}")
    lines.append(f"CENTERING: {packet['centering_error']:.0f}%")
    lines.append(f"BACKEND: {backend}")
    return lines


def debug_lines(info: dict) -> list[str]:
    raw = info.get("raw_xy")
    filt = info.get("filtered_xy")
    raw_s = "—" if not raw else f"{raw[0]:.0f},{raw[1]:.0f}"
    filt_s = "—" if not filt else f"{filt[0]:.0f},{filt[1]:.0f}"
    return [
        f"FRAME {info.get('frame_id')}  {info.get('frame_w')}x{info.get('frame_h')}",
        f"YOLO {info.get('yolo_input')}  RAW {info.get('raw_detections')}  NMS {info.get('nms_detections')}",
        f"TARGET RAW {raw_s}  FILT {filt_s}",
        f"DX {info.get('dx', 0):+.1f}  DY {info.get('dy', 0):+.1f}  MAG {info.get('magnitude', 0):.1f}",
        f"NORM {info.get('x_norm', 0):+.2f},{info.get('y_norm', 0):+.2f}",
        f"ZONE {info.get('dead_zone')}  N {info.get('virtual_n', 0):+.2f}  E {info.get('virtual_e', 0):+.2f}",
        f"SIZE {info.get('target_size', 0):.2f}%  {info.get('scale_trend')}",
        f"DET {info.get('det_ms', 0):.1f}  TRK {info.get('track_ms', 0):.1f}  RDR {info.get('render_ms', 0):.1f} ms",
    ]


def draw_fps(frame, text: str) -> None:
    cv2.rectangle(frame, (8, 8), (220, 36), (10, 14, 20), -1)
    cv2.putText(frame, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 220), 1, cv2.LINE_AA)


def draw_rec_badge(frame, label: str) -> None:
    cv2.circle(frame, (28, 58), 7, (40, 40, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, label, (42, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2, cv2.LINE_AA)
