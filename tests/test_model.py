import numpy as np

from neurosync.model import CalibrationRecord, fit


def test_fit_recovers_known_parameters():
    k_sys = 1.2
    k_accel = 0.004

    rows = []
    for sensitivity, accel, int_v, int_v2 in [
        (1.0, 1.0, 20.0, 100.0),
        (1.0, 0.5, 30.0, 300.0),
        (0.5, 1.0, 45.0, 700.0),
        (0.5, 0.5, 55.0, 1200.0),
        (1.3, 0.7, 25.0, 500.0),
        (0.7, 1.4, 40.0, 900.0),
    ]:
        y = k_sys * sensitivity * int_v + k_accel * accel * int_v2
        rows.append(
            CalibrationRecord(
                sensitivity=sensitivity,
                accel_setting=accel,
                int_v=int_v,
                int_v2=int_v2,
                observed_angle_deg=y,
            )
        )

    result = fit(rows)
    assert np.isclose(result["k_sys"], k_sys, atol=1e-8)
    assert np.isclose(result["k_accel"], k_accel, atol=1e-8)
    assert result["r_squared"] > 0.999999
