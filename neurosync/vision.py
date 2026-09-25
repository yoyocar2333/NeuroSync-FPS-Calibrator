"""Computer-vision feature extraction for the NeuroSync reconstruction.

This module implements the pipeline documented in the supplied course slides:
gray-scale frames -> band-pass threshold -> connected-component centroid ->
pixel displacement -> physical velocity -> integral features used by the
system-identification model.

The original privacy-mask geometry and raw videos were not supplied, so masking
is configurable rather than hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    threshold_low: int = 69
    threshold_high: int = 80
    min_area: int = 2
    max_area: int | None = None
    pixel_to_cm: float = 22.6 / 2360.0
    axis: Literal["x", "y", "xy"] = "x"
    roi: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1


@dataclass(frozen=True)
class MotionFeatures:
    fps: float
    n_frames: int
    n_tracked: int
    tracking_rate: float
    int_v: float
    int_v2: float
    path_cm: float
    duration_s: float


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(frame, code)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def track_centroid(frame: np.ndarray, config: TrackingConfig = TrackingConfig()) -> tuple[float, float] | None:
    """Track the largest threshold-band component and return its centroid.

    The slides describe a grayscale/band-pass/connected-component pipeline and
    quote a threshold band of [69, 80]. This reconstruction follows that
    description exactly while keeping ROI and area filtering configurable.
    """
    gray = _gray(frame)
    x_offset = y_offset = 0
    if config.roi is not None:
        x0, y0, x1, y1 = config.roi
        if not (0 <= x0 < x1 <= gray.shape[1] and 0 <= y0 < y1 <= gray.shape[0]):
            raise ValueError("ROI must lie inside the frame")
        gray = gray[y0:y1, x0:x1]
        x_offset, y_offset = x0, y0

    mask = cv2.inRange(gray, config.threshold_low, config.threshold_high)
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates: list[tuple[int, int]] = []
    for label in range(1, n_labels):  # skip background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.min_area:
            continue
        if config.max_area is not None and area > config.max_area:
            continue
        candidates.append((area, label))

    if not candidates:
        return None

    _, label = max(candidates)
    cx, cy = centroids[label]
    return float(cx + x_offset), float(cy + y_offset)


def motion_features(
    centroids: Sequence[tuple[float, float] | None],
    fps: float,
    config: TrackingConfig = TrackingConfig(),
) -> MotionFeatures:
    """Convert tracked centroids to the two integral features used by NeuroSync.

    Missing detections break continuity: velocity is computed only when both
    adjacent frames have valid centroids, preventing interpolation from silently
    inventing motion that was not observed.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(centroids) < 2:
        raise ValueError("At least two frames are required")

    dt = 1.0 / fps
    velocities: list[float] = []
    path_cm = 0.0
    n_tracked = sum(c is not None for c in centroids)

    for prev, cur in zip(centroids, centroids[1:]):
        if prev is None or cur is None:
            continue
        dx = (cur[0] - prev[0]) * config.pixel_to_cm
        dy = (cur[1] - prev[1]) * config.pixel_to_cm
        if config.axis == "x":
            displacement = abs(dx)
        elif config.axis == "y":
            displacement = abs(dy)
        else:
            displacement = float(np.hypot(dx, dy))
        path_cm += displacement
        velocities.append(displacement / dt)

    v = np.asarray(velocities, dtype=float)
    int_v = float(np.sum(v) * dt)
    int_v2 = float(np.sum(v * v) * dt)
    n_frames = len(centroids)
    return MotionFeatures(
        fps=float(fps),
        n_frames=n_frames,
        n_tracked=n_tracked,
        tracking_rate=n_tracked / n_frames,
        int_v=int_v,
        int_v2=int_v2,
        path_cm=float(path_cm),
        duration_s=(n_frames - 1) * dt,
    )


def extract_video(path: Path, config: TrackingConfig = TrackingConfig()) -> MotionFeatures:
    """Extract NeuroSync motion features from a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise ValueError("Video reports an invalid FPS")

    centroids: list[tuple[float, float] | None] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            centroids.append(track_centroid(frame, config))
    finally:
        cap.release()

    if len(centroids) < 2:
        raise ValueError("Video contains fewer than two decodable frames")
    return motion_features(centroids, fps, config)
