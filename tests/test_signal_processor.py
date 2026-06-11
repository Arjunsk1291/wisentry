"""Unit tests for filtering and feature extraction."""

import numpy as np

from backend.csi_parser import CsiFrame
from backend.signal_processor import (
    SignalProcessor,
    average_into_bands,
    hampel_filter,
)

RNG = np.random.default_rng(42)


def _make_frame(device_id, sequence, amplitudes):
    return CsiFrame(
        device_id=device_id,
        sequence_number=sequence,
        esp32_timestamp_us=sequence * 20_000,
        rssi_dbm=-50,
        amplitudes=amplitudes.astype(np.float32),
        phases=np.zeros(len(amplitudes), dtype=np.float32),
    )


def test_hampel_removes_impulsive_spike():
    series = np.full(64, 20.0)
    series[10] = 200.0
    cleaned = hampel_filter(series, window_radius=5, sigma_threshold=3.0)
    assert cleaned[10] < 25.0
    assert np.allclose(cleaned[:5], 20.0)


def test_hampel_preserves_clean_signal():
    series = 20.0 + RNG.normal(0, 0.5, 64)
    cleaned = hampel_filter(series, window_radius=5, sigma_threshold=3.0)
    assert np.abs(cleaned - series).max() < 3.0


def test_band_averaging_shape_and_values():
    amplitudes = np.arange(64, dtype=np.float64)
    bands = average_into_bands(amplitudes, 16)
    assert bands.shape == (16,)
    assert bands[0] == np.mean([0, 1, 2, 3])


def test_window_emitted_after_window_frames():
    processor = SignalProcessor({"window_frames": 50, "feature_bands": 16})
    emitted = []
    for sequence in range(60):
        amplitudes = 20 + RNG.normal(0, 1, 64)
        window = processor.add_frame(_make_frame(1, sequence, amplitudes))
        if window is not None:
            emitted.append(window)
    assert len(emitted) == 11  # frames 50..60 inclusive of the first full one
    assert emitted[0].band_matrix.shape == (50, 16)


def test_motion_energy_separates_still_from_moving():
    processor = SignalProcessor({})
    still_windows, moving_windows = [], []
    for sequence in range(120):
        amplitudes = 20 + RNG.normal(0, 0.3, 64)
        window = processor.add_frame(_make_frame(1, sequence, amplitudes))
        if window is not None:
            still_windows.append(window.motion_energy)
    for sequence in range(120):
        amplitudes = 20 + RNG.normal(0, 5.0, 64)
        window = processor.add_frame(_make_frame(2, sequence, amplitudes))
        if window is not None:
            moving_windows.append(window.motion_energy)
    assert np.mean(moving_windows) > 5 * np.mean(still_windows)


def test_baseline_deviation_rises_on_amplitude_shift():
    processor = SignalProcessor({})
    last_window = None
    for sequence in range(150):  # baseline period: stable amplitudes
        window = processor.add_frame(
            _make_frame(1, sequence, 20 + RNG.normal(0, 0.5, 64))
        )
        if window is not None:
            last_window = window
    deviation_before_shift = last_window.baseline_deviation
    for sequence in range(150, 260):  # body shadow: shifted amplitudes
        window = processor.add_frame(
            _make_frame(1, sequence, 14 + RNG.normal(0, 0.5, 64))
        )
        if window is not None:
            last_window = window
    assert last_window.baseline_deviation > deviation_before_shift + 1.0
