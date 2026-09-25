"""System-identification model used by the NeuroSync calibrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CalibrationRecord:
    sensitivity: float
    accel_setting: float
    int_v: float
    int_v2: float
    observed_angle_deg: float


def predict_angle(
    k_sys: float,
    k_accel: float,
    sensitivity: np.ndarray,
    accel_setting: np.ndarray,
    int_v: np.ndarray,
    int_v2: np.ndarray,
) -> np.ndarray:
    """Predict camera rotation in degrees from the documented NeuroSync model."""
    return k_sys * sensitivity * int_v + k_accel * accel_setting * int_v2


def fit(records: Iterable[CalibrationRecord]) -> dict[str, float]:
    rows = list(records)
    if len(rows) < 2:
        raise ValueError("At least two calibration records are required.")

    s = np.asarray([r.sensitivity for r in rows], dtype=float)
    a = np.asarray([r.accel_setting for r in rows], dtype=float)
    int_v = np.asarray([r.int_v for r in rows], dtype=float)
    int_v2 = np.asarray([r.int_v2 for r in rows], dtype=float)
    y = np.asarray([r.observed_angle_deg for r in rows], dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        return predict_angle(params[0], params[1], s, a, int_v, int_v2) - y

    result = least_squares(residual, x0=np.array([1.0, 0.001], dtype=float))
    y_hat = predict_angle(result.x[0], result.x[1], s, a, int_v, int_v2)

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    mae = float(np.mean(np.abs(y - y_hat)))

    return {
        "k_sys": float(result.x[0]),
        "k_accel": float(result.x[1]),
        "r_squared": r2,
        "rmse_deg": rmse,
        "mae_deg": mae,
        "cost": float(result.cost),
        "n_records": float(len(rows)),
    }
