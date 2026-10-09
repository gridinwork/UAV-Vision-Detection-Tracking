"""Flight-controller backends.

VirtualFlightController is the only active implementation.
PX4Backend is reserved and disabled: it does not import MAVSDK and it does
not open a UDP, serial, Pixhawk, or QGroundControl connection.
"""

from __future__ import annotations

from typing import Protocol


class FlightControllerBackend(Protocol):
    name: str
    enabled: bool

    def connect(self) -> bool:
        ...

    def send(self, command: dict) -> None:
        ...

    def disconnect(self) -> None:
        ...


class VirtualFlightController:
    """Holds the latest virtual command for display and CSV export."""

    name = "VIRTUAL"
    enabled = True

    def __init__(self) -> None:
        self.connected = False
        self.last_command: dict = {}

    def connect(self) -> bool:
        self.connected = True
        return True

    def send(self, command: dict) -> None:
        self.last_command = dict(command)

    def disconnect(self) -> None:
        self.connected = False
        self.last_command = {}


class PX4Backend:
    """Disabled stub. Present so a future stack can be added deliberately."""

    name = "PX4"
    enabled = False

    def __init__(self) -> None:
        self.connected = False
        self.last_command: dict = {}

    def connect(self) -> bool:
        self.connected = False
        return False

    def send(self, command: dict) -> None:
        return None

    def disconnect(self) -> None:
        self.connected = False
