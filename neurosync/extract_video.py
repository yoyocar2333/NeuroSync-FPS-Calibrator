"""CLI: video -> NeuroSync motion integrals."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .vision import TrackingConfig, extract_video


def main() -> None:
    p = argparse.ArgumentParser(description="Extract NeuroSync motion features from a gesture video.")
    p.add_argument("video", type=Path)
    p.add_argument("--threshold-low", type=int, default=69)
    p.add_argument("--threshold-high", type=int, default=80)
    p.add_argument("--min-area", type=int, default=2)
    p.add_argument("--pixel-to-cm", type=float, default=22.6 / 2360.0)
    p.add_argument("--axis", choices=["x", "y", "xy"], default="x")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    args = p.parse_args()

    cfg = TrackingConfig(
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        min_area=args.min_area,
        pixel_to_cm=args.pixel_to_cm,
        axis=args.axis,
        roi=tuple(args.roi) if args.roi else None,
    )
    print(json.dumps(asdict(extract_video(args.video, cfg)), indent=2))


if __name__ == "__main__":
    main()
