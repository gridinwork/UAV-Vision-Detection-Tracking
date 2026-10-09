"""Session log of virtual commands. Nothing here is sent to a vehicle."""

from __future__ import annotations

import csv
from pathlib import Path


COLUMNS = [
    "timestamp",
    "frame",
    "target_id",
    "target_x",
    "target_y",
    "x_offset",
    "y_offset",
    "filtered_x",
    "filtered_y",
    "target_size",
    "direction_command",
    "speed_command",
    "virtual_n",
    "virtual_e",
    "status",
]


class ControlCsvLog:
    def __init__(self, limit: int = 100_000) -> None:
        self.rows: list[dict] = []
        self.limit = limit

    def clear(self) -> None:
        self.rows.clear()

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) > self.limit:
            self.rows = self.rows[-self.limit :]

    def export(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key, "") for key in COLUMNS})
        return len(self.rows)
