"""Builds one virtual command packet from tracker output. Nothing is transmitted."""

from __future__ import annotations

from control.control_logic import control_state, directional_command, offsets
from control.flight_backend import VirtualFlightController
from control.ned_simulation import virtual_ned
from tracking.target_state import TargetState


class CommandBuilder:
    def __init__(self) -> None:
        self.backend = VirtualFlightController()
        self.backend.connect()

    def build(self, state: TargetState, pixel_scale: float, active_target: bool) -> dict:
        frame_cx = state.frame_w / 2.0
        frame_cy = state.frame_h / 2.0
        half_w, half_h = state.dead_zone_half
        if state.filtered_xy is None or not active_target:
            packet = {
                "command": "HOLD",
                "speed": "HOLD SPEED",
                "virtual_n": 0.0,
                "virtual_e": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "x_norm": 0.0,
                "y_norm": 0.0,
                "magnitude": 0.0,
                "centering_error": 0.0,
                "raw_offset_x": 0.0,
                "raw_offset_y": 0.0,
                "filtered_offset_x": 0.0,
                "filtered_offset_y": 0.0,
                "control_state": control_state(state.status, "HOLD"),
                "dead_zone": "INSIDE" if state.status != "LOST" else "LOST",
                "backend": self.backend.name,
            }
            self.backend.send(packet)
            return packet

        fx, fy = state.filtered_xy
        rx, ry = state.raw_xy or state.filtered_xy
        raw_off = offsets(rx, ry, frame_cx, frame_cy, state.frame_w, state.frame_h)
        filt_off = offsets(fx, fy, frame_cx, frame_cy, state.frame_w, state.frame_h)
        inside = state.inside_dead_zone or state.status in ("LOST", "SEARCHING")
        command = directional_command(filt_off["dx"], filt_off["dy"], half_w, half_h, active_target and not inside)
        if state.status == "LOST":
            command = "HOLD"
            inside = True
        north, east = virtual_ned(fx, fy, frame_cx, frame_cy, pixel_scale, inside or command == "HOLD")
        packet = {
            "command": command,
            "virtual_n": north,
            "virtual_e": east,
            "dx": filt_off["dx"],
            "dy": filt_off["dy"],
            "x_norm": filt_off["x_norm"],
            "y_norm": filt_off["y_norm"],
            "magnitude": filt_off["magnitude"],
            "centering_error": filt_off["centering_error"],
            "raw_offset_x": raw_off["dx"],
            "raw_offset_y": raw_off["dy"],
            "filtered_offset_x": filt_off["dx"],
            "filtered_offset_y": filt_off["dy"],
            "control_state": control_state(state.status, command),
            "dead_zone": "INSIDE" if inside else "OUTSIDE",
            "backend": self.backend.name,
        }
        self.backend.send(packet)
        return packet
