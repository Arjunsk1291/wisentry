"""Tests for respiration and spectrogram analytics."""

import numpy as np

from backend.csi_analytics import (CsiAnalytics, estimate_breathing,
                                   motion_spectrogram)


def _breathing_matrix(rate_hz, seconds=30.0, frame_rate=50.0, seed=0):
    times = np.arange(0, seconds, 1 / frame_rate)
    matrix = 35 + np.random.default_rng(seed).normal(0, 1, (len(times), 16))
    matrix[:, 6:10] += 1.5 * np.sin(2 * np.pi * rate_hz * times)[:, None]
    return matrix


def test_breathing_rate_recovered_within_half_bpm():
    for rate_hz in (0.15, 0.25, 0.4):
        result = estimate_breathing(_breathing_matrix(rate_hz), 50.0)
        assert abs(result["bpm"] - rate_hz * 60) < 0.5


def test_breathing_needs_enough_history():
    assert estimate_breathing(_breathing_matrix(0.3, seconds=5), 50.0) is None


def test_spectrogram_speed_index_rises_with_motion():
    rng = np.random.default_rng(1)
    still = 35 + rng.normal(0, 0.2, (250, 16))
    times = np.arange(250) / 50.0
    moving = still.copy()
    moving += 3 * np.sin(2 * np.pi * 4.0 * times)[:, None] * rng.normal(
        1, 0.3, 16)
    moving += 3 * np.sin(2 * np.pi * 6.0 * times)[:, None] * rng.normal(
        1, 0.3, 16)
    assert motion_spectrogram(moving, 50.0)["speed_index"] > 2.0


def test_analytics_compute_outputs():
    analytics = CsiAnalytics(50.0)
    for row in _breathing_matrix(0.3):
        analytics.ingest(1, row)
    output = analytics.compute()
    assert abs(output["breathing"]["bpm"] - 18.0) < 0.5
    assert output["spectrogram"] is not None
    assert len(output["heatmap"]["values"]) == 16
    assert 1 in output["link_activity"]


def test_breathing_uses_timestamps_under_jitter_and_drops():
    rng = np.random.default_rng(2)
    stamps = np.cumsum(rng.uniform(0.012, 0.045, 1200))  # ~35 Hz, uneven
    analytics = CsiAnalytics(50.0)
    for stamp in stamps:
        row = 35 + rng.normal(0, 1, 16)
        row[6:10] += 1.5 * np.sin(2 * np.pi * 0.3 * stamp)
        analytics.ingest(1, row, stamp)
    assert abs(analytics.compute()["breathing"]["bpm"] - 18.0) < 0.5
