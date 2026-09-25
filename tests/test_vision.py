import numpy as np

from neurosync.vision import TrackingConfig, motion_features, track_centroid


def frame_with_dot(x: int, y: int, value: int = 75) -> np.ndarray:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[y-1:y+2, x-1:x+2] = value
    return frame


def test_bandpass_connected_component_centroid():
    c = track_centroid(frame_with_dot(40, 30), TrackingConfig(min_area=2))
    assert c is not None
    assert np.allclose(c, (40.0, 30.0), atol=0.1)


def test_out_of_band_target_is_rejected():
    assert track_centroid(frame_with_dot(40, 30, value=120)) is None


def test_motion_integrals_are_physically_consistent():
    # 10 pixels per frame at 10 fps, 0.1 cm/pixel -> 10 cm/s.
    centroids = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    cfg = TrackingConfig(pixel_to_cm=0.1, axis="x")
    f = motion_features(centroids, fps=10.0, config=cfg)
    assert np.isclose(f.path_cm, 3.0)
    assert np.isclose(f.int_v, 3.0)
    assert np.isclose(f.int_v2, 30.0)
    assert f.tracking_rate == 1.0


def test_missing_detection_breaks_velocity_segment():
    centroids = [(0.0, 0.0), (10.0, 0.0), None, (30.0, 0.0), (40.0, 0.0)]
    cfg = TrackingConfig(pixel_to_cm=0.1, axis="x")
    f = motion_features(centroids, fps=10.0, config=cfg)
    assert np.isclose(f.path_cm, 2.0)
    assert np.isclose(f.int_v, 2.0)
    assert f.n_tracked == 4
