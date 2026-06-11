"""CSI signal processing: filtering and feature extraction.

Maintains a rolling window of CSI frames per ESP32 receiver, cleans the
amplitude data (Hampel outlier rejection + Butterworth low-pass), and
produces fixed-size FeatureWindow objects that feed both the rule-based
detector and the ML models. The training pipeline (models/train_all.py)
uses this exact same code path, so training and runtime features can
never diverge.

Runs on: any OS with Python 3.9+. Dependencies: numpy, scipy.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

WINDOW_FRAMES_DEFAULT = 50
FEATURE_BANDS_DEFAULT = 16
HAMPEL_WINDOW_DEFAULT = 5
HAMPEL_SIGMA_DEFAULT = 3.0
LOWPASS_CUTOFF_HZ_DEFAULT = 10.0
FRAME_RATE_HZ_DEFAULT = 50.0
BASELINE_FRAMES = 100  # ~2 s at 50 Hz; room assumed empty at startup
BREATHING_HISTORY_FRAMES = 256  # ~5 s; long enough to resolve 0.2-0.5 Hz
BREATHING_BAND_LOW_HZ = 0.2
BREATHING_BAND_HIGH_HZ = 0.5
MEDIAN_ABS_DEVIATION_SCALE = 1.4826  # MAD -> sigma for Gaussian data
NUMERICAL_FLOOR = 1e-6  # avoids divide-by-zero on flat signals


@dataclass
class FeatureWindow:
    """One fixed-size analysis window for a single receiver.

    Attributes:
        device_id (int): Which ESP32 receiver produced the window.
        band_matrix (np.ndarray): float32[window_frames, feature_bands]
            z-scored band-averaged amplitudes (the ML model input).
        motion_energy (float): Mean temporal variance of band amplitudes —
            large when a body is moving.
        baseline_deviation (float): Mean absolute distance of the current
            band means from the empty-room baseline, in baseline-sigma units.
        breathing_energy (float): Spectral energy in the 0.2-0.5 Hz band of
            the mean amplitude over ~5 s — elevated when a still person
            breathes in the field.
        mean_amplitude (float): Window mean raw amplitude (device health).
        timestamp (float): Wall-clock time of the newest frame.
    """

    device_id: int
    band_matrix: np.ndarray
    motion_energy: float
    baseline_deviation: float
    breathing_energy: float
    mean_amplitude: float
    timestamp: float


def hampel_filter(amplitude_series, window_radius, sigma_threshold):
    """Replace impulsive outliers with the local median.

    Args:
        amplitude_series (np.ndarray): 1-D float array to clean.
        window_radius (int): Frames on each side of the evaluated sample.
        sigma_threshold (float): Outlier threshold in MAD-derived sigmas.

    Returns:
        np.ndarray: A cleaned copy of the input series.
    """
    cleaned_series = amplitude_series.copy()
    series_length = len(amplitude_series)
    for sample_index in range(series_length):
        window_start = max(0, sample_index - window_radius)
        window_end = min(series_length, sample_index + window_radius + 1)
        local_window = amplitude_series[window_start:window_end]
        local_median = np.median(local_window)
        local_sigma = MEDIAN_ABS_DEVIATION_SCALE * np.median(
            np.abs(local_window - local_median)
        )
        deviation = abs(amplitude_series[sample_index] - local_median)
        if deviation > sigma_threshold * max(local_sigma, NUMERICAL_FLOOR):
            cleaned_series[sample_index] = local_median
    return cleaned_series


def lowpass_filter_columns(amplitude_matrix, cutoff_hz, frame_rate_hz):
    """Apply a zero-phase Butterworth low-pass to each column (band).

    Args:
        amplitude_matrix (np.ndarray): float[frames, bands] time series.
        cutoff_hz (float): Low-pass cutoff frequency.
        frame_rate_hz (float): Sampling rate of the frame stream.

    Returns:
        np.ndarray: Filtered matrix of identical shape.
    """
    nyquist_hz = frame_rate_hz / 2.0
    normalized_cutoff = min(cutoff_hz / nyquist_hz, 0.99)
    filter_b, filter_a = butter(2, normalized_cutoff, btype="low")
    minimum_length_for_filtfilt = 3 * max(len(filter_a), len(filter_b))
    if amplitude_matrix.shape[0] <= minimum_length_for_filtfilt:
        return amplitude_matrix
    return filtfilt(filter_b, filter_a, amplitude_matrix, axis=0)


def average_into_bands(amplitude_array, band_count):
    """Average a per-subcarrier amplitude vector into coarse bands.

    Args:
        amplitude_array (np.ndarray): float[subcarriers] amplitudes.
        band_count (int): Number of output bands.

    Returns:
        np.ndarray: float32[band_count] band-mean amplitudes.
    """
    usable_length = (len(amplitude_array) // band_count) * band_count
    reshaped = amplitude_array[:usable_length].reshape(band_count, -1)
    return reshaped.mean(axis=1).astype(np.float32)


class SignalProcessor:
    """Per-device rolling-window feature extractor.

    Feed every parsed CsiFrame to add_frame(); once a device has a full
    window of frames you get a FeatureWindow back (then again every frame,
    sliding). The first BASELINE_FRAMES frames per device are also used to
    estimate the empty-room baseline.
    """

    def __init__(self, signal_config):
        """Configure the processor.

        Args:
            signal_config (dict): The `signal` section of config.yaml.
        """
        self.window_frames = int(
            signal_config.get("window_frames", WINDOW_FRAMES_DEFAULT)
        )
        self.feature_bands = int(
            signal_config.get("feature_bands", FEATURE_BANDS_DEFAULT)
        )
        self.hampel_window = int(
            signal_config.get("hampel_window", HAMPEL_WINDOW_DEFAULT)
        )
        self.hampel_sigma = float(
            signal_config.get("hampel_sigma", HAMPEL_SIGMA_DEFAULT)
        )
        self.lowpass_cutoff_hz = float(
            signal_config.get("lowpass_cutoff_hz", LOWPASS_CUTOFF_HZ_DEFAULT)
        )
        self.frame_rate_hz = float(
            signal_config.get("frame_rate_hz", FRAME_RATE_HZ_DEFAULT)
        )
        self._band_history_per_device = {}
        self._mean_history_per_device = {}
        self._baseline_samples_per_device = {}
        self._baseline_per_device = {}

    def _device_buffers(self, device_id):
        """Get (creating on first use) the rolling buffers for a device."""
        if device_id not in self._band_history_per_device:
            self._band_history_per_device[device_id] = deque(
                maxlen=self.window_frames
            )
            self._mean_history_per_device[device_id] = deque(
                maxlen=BREATHING_HISTORY_FRAMES
            )
            self._baseline_samples_per_device[device_id] = []
        return (
            self._band_history_per_device[device_id],
            self._mean_history_per_device[device_id],
        )

    def _update_baseline(self, device_id, band_vector):
        """Accumulate empty-room baseline statistics for a device."""
        baseline_samples = self._baseline_samples_per_device[device_id]
        if device_id in self._baseline_per_device:
            return
        baseline_samples.append(band_vector)
        if len(baseline_samples) >= BASELINE_FRAMES:
            stacked_samples = np.stack(baseline_samples)
            self._baseline_per_device[device_id] = (
                stacked_samples.mean(axis=0),
                np.maximum(stacked_samples.std(axis=0), NUMERICAL_FLOOR),
            )
            self._baseline_samples_per_device[device_id] = []

    def _compute_breathing_energy(self, mean_history):
        """Spectral energy of the mean amplitude in the breathing band."""
        if len(mean_history) < BREATHING_HISTORY_FRAMES:
            return 0.0
        series = np.asarray(mean_history, dtype=np.float64)
        series = series - series.mean()
        spectrum_power = np.abs(np.fft.rfft(series)) ** 2
        frequencies = np.fft.rfftfreq(len(series), d=1.0 / self.frame_rate_hz)
        breathing_mask = (frequencies >= BREATHING_BAND_LOW_HZ) & (
            frequencies <= BREATHING_BAND_HIGH_HZ
        )
        total_power = spectrum_power.sum() + NUMERICAL_FLOOR
        return float(spectrum_power[breathing_mask].sum() / total_power)

    def _build_feature_window(self, device_id, band_history, mean_history):
        """Assemble a FeatureWindow from full per-device buffers."""
        raw_band_matrix = np.stack(band_history).astype(np.float64)
        filtered_matrix = lowpass_filter_columns(
            raw_band_matrix, self.lowpass_cutoff_hz, self.frame_rate_hz
        )
        matrix_mean = filtered_matrix.mean()
        matrix_std = max(filtered_matrix.std(), NUMERICAL_FLOOR)
        zscored_matrix = ((filtered_matrix - matrix_mean) / matrix_std).astype(
            np.float32
        )
        motion_energy = float(
            np.var(np.diff(filtered_matrix, axis=0), axis=0).mean()
        )
        baseline_deviation = 0.0
        if device_id in self._baseline_per_device:
            baseline_mean, baseline_std = self._baseline_per_device[device_id]
            current_band_means = filtered_matrix.mean(axis=0)
            baseline_deviation = float(
                (np.abs(current_band_means - baseline_mean) / baseline_std).mean()
            )
        return FeatureWindow(
            device_id=device_id,
            band_matrix=zscored_matrix,
            motion_energy=motion_energy,
            baseline_deviation=baseline_deviation,
            breathing_energy=self._compute_breathing_energy(mean_history),
            mean_amplitude=float(raw_band_matrix.mean()),
            timestamp=time.time(),
        )

    def add_frame(self, csi_frame):
        """Ingest one CsiFrame; return a FeatureWindow when one is ready.

        Args:
            csi_frame (CsiFrame): A parsed frame from csi_parser.

        Returns:
            FeatureWindow | None: None until the device's window is full,
                afterwards a fresh sliding window every frame.
        """
        band_history, mean_history = self._device_buffers(csi_frame.device_id)
        cleaned_amplitudes = hampel_filter(
            csi_frame.amplitudes, self.hampel_window, self.hampel_sigma
        )
        band_vector = average_into_bands(cleaned_amplitudes, self.feature_bands)
        self._update_baseline(csi_frame.device_id, band_vector)
        band_history.append(band_vector)
        mean_history.append(float(band_vector.mean()))
        if len(band_history) < self.window_frames:
            return None
        return self._build_feature_window(
            csi_frame.device_id, band_history, mean_history
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    spike_series = np.full(64, 20.0)
    spike_series[30] = 120.0  # impulsive outlier
    cleaned = hampel_filter(spike_series, HAMPEL_WINDOW_DEFAULT, HAMPEL_SIGMA_DEFAULT)
    assert cleaned[30] < 30.0, "Hampel filter failed to remove a spike"

    from backend.csi_parser import CsiFrame  # local import for self-test only

    processor = SignalProcessor({})
    produced_windows = 0
    for frame_index in range(160):
        fake_frame = CsiFrame(
            device_id=1,
            sequence_number=frame_index,
            esp32_timestamp_us=frame_index * 20000,
            rssi_dbm=-50,
            amplitudes=(20 + rng.normal(0, 1, 64)).astype(np.float32),
            phases=np.zeros(64, dtype=np.float32),
        )
        window = processor.add_frame(fake_frame)
        if window is not None:
            produced_windows += 1
            assert window.band_matrix.shape == (50, 16)
    assert produced_windows == 160 - 50 + 1
    print("signal_processor self-test: PASS "
          f"({produced_windows} windows, hampel OK, baseline OK)")
