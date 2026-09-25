"""CSV command-line interface for NeuroSync system identification."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .model import CalibrationRecord, fit


def load_csv(path: Path) -> list[CalibrationRecord]:
    records: list[CalibrationRecord] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                CalibrationRecord(
                    sensitivity=float(row["sensitivity"]),
                    accel_setting=float(row["accel_setting"]),
                    int_v=float(row["int_v"]),
                    int_v2=float(row["int_v2"]),
                    observed_angle_deg=float(row["observed_angle_deg"]),
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit NeuroSync FPS control constants.")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(fit(load_csv(args.csv)), indent=2))


if __name__ == "__main__":
    main()
