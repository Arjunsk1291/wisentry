"""CSI analytics: respiration rate, motion spectrogram, band heatmap.

Methods follow published magnitude-only CSI techniques that fit ESP32
receivers (single antenna, unreliable phase):

* Respiration rate: Hampel-cleaned band amplitudes (done upstream in
  SignalProcessor), band-pass 0.1-0.6 Hz, per-band breathing-to-noise ratio
  (BNR) to pick the bands that actually carry the chest motion, zero-padded
  FFT peak, BNR-weighted fusion across bands and receivers. See the review
  in "VitalCSI" (PMC12788229) for the family of methods (Liu et al.,
  WiResP, Ge & Ho).
* Motion spectrogram: CARM (Wang et al., MobiCom 2015) - PCA across the
  subcarrier-band streams, drop the first component, then STFT; the energy
  per frequency is a proxy for body-movement speed.

Runs on: any OS with Python 3.9+. Dependencies: numpy, scipy.
"""

from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt, stft

ANALYTICS_HISTORY_SECONDS = 30.0
BREATHING_MIN_SECONDS = 12.0
STILL_STEP_THRESHOLD = 0.16          # median |frame-to-frame| band-mean step
BREATHING_DECIMATION = 5            # 50 Hz -> 10 Hz before the FFT
BREATHING_BAND_HZ = (0.1, 0.6)      # 6-36 breaths per minute
BREATHING_FFT_POINTS = 4096
BREATHING_TOP_BANDS = 4
BREATHING_MIN_BNR = 0.25            # below this the estimate is withheld
SPECTROGRAM_SECONDS = 4.0
SPECTROGRAM_SEGMENT = 64
SPECTROGRAM_OVERLAP = 56
SPECTROGRAM_MAX_HZ = 10.0
HEATMAP_SECONDS = 5.0
# CARM speed-profile bands. With ~6 cm path change per cycle at 2.4 GHz,
# f Hz of CSI power oscillation ~ 0.06*f m/s of path-length change.
SPEED_PROFILE_BANDS_HZ = [(0.0, 1.0), (1.0, 2.5), (2.5, 5.0), (5.0, 10.0)]
HEATMAP_COLUMNS = 50


def trailing_still_frames(band_matrix, frame_rate_hz):
    """Frames at the end of the history with no gross body motion.

    Walking swamps the 0.1-0.6 Hz band, so respiration is estimated only
    over the trailing still segment. Stillness per second = median absolute
    frame-to-frame step of the band mean, against an adaptive threshold.

    Returns:
        int: Number of trailing frames judged still.
    """
    per_second = int(frame_rate_hz)
    steps = np.abs(np.diff(band_matrix.mean(axis=1)))
    seconds = len(steps) // per_second
    if seconds == 0:
        return 0
    medians = np.array([
        np.median(steps[len(steps) - (i + 1) * per_second:
                        len(steps) - i * per_second])
        for i in range(seconds)])
    threshold = max(STILL_STEP_THRESHOLD, 1.8 * np.percentile(medians, 20))
    still_seconds = 0
    for value in medians:
        if value > threshold:
            break
        still_seconds += 1
    return still_seconds * per_second


def estimate_breathing(band_matrix, frame_rate_hz):
    """Respiration rate from one receiver's band-amplitude history.

    Args:
        band_matrix (np.ndarray): float[frames, bands], oldest first.
        frame_rate_hz (float): Frame rate of the rows.

    Returns:
        dict | None: {"bpm", "bnr", "waveform", "waveform_rate_hz"} or None
            when there is not enough history.
    """
    band_matrix = band_matrix[-trailing_still_frames(band_matrix,
                                                     frame_rate_hz) or 1:]
    frames = band_matrix.shape[0]
    if frames < BREATHING_MIN_SECONDS * frame_rate_hz:
        return None
    usable = frames - frames % BREATHING_DECIMATION
    decimated = band_matrix[-usable:].reshape(
        -1, BREATHING_DECIMATION, band_matrix.shape[1]).mean(axis=1)
    rate = frame_rate_hz / BREATHING_DECIMATION
    # First difference whitens step changes (pose shifts, gain jumps) whose
    # 1/f spectrum would otherwise win at the band's low edge; a sinusoid's
    # frequency is unchanged by differencing.
    decimated = np.diff(decimated, axis=0)
    decimated = decimated - decimated.mean(axis=0)
    low, high = BREATHING_BAND_HZ
    b, a = butter(2, [low / (rate / 2), high / (rate / 2)], btype="band")
    filtered = filtfilt(b, a, decimated, axis=0)
    window = np.hanning(filtered.shape[0])[:, None]
    spectrum = np.abs(np.fft.rfft(filtered * window,
                                  n=BREATHING_FFT_POINTS, axis=0)) ** 2
    freqs = np.fft.rfftfreq(BREATHING_FFT_POINTS, d=1.0 / rate)
    in_band = (freqs >= low) & (freqs <= high)
    band_spectrum = spectrum[in_band]
    band_freqs = freqs[in_band]
    peak_rows = np.argmax(band_spectrum, axis=0)
    peak_freqs = band_freqs[peak_rows]
    # BNR: power within +-0.03 Hz of the peak over all in-band power.
    bnr = np.empty(band_spectrum.shape[1])
    for column, peak in enumerate(peak_freqs):
        near = np.abs(band_freqs - peak) <= 0.03
        bnr[column] = band_spectrum[near, column].sum() / max(
            band_spectrum[:, column].sum(), 1e-12)
    # A peak pinned to the band edge is leakage, not breathing.
    edge = (peak_freqs <= low + 0.02) | (peak_freqs >= high - 0.02)
    bnr[edge] = 0.0
    if not np.any(bnr > 0):
        return None
    top = np.argsort(bnr)[-BREATHING_TOP_BANDS:]
    top = top[bnr[top] > 0]
    weights = bnr[top]
    bpm = float(np.average(peak_freqs[top], weights=weights) * 60.0)
    # Band polarities differ, so show the strongest band rather than a sum.
    waveform = np.cumsum(filtered[:, top[-1]])
    waveform = waveform - waveform.mean()
    return {"bpm": bpm, "bnr": float(weights.max()),
            "waveform": waveform.astype(np.float32),
            "waveform_rate_hz": rate,
            "still_seconds": frames / frame_rate_hz}


def motion_spectrogram(band_matrix, frame_rate_hz):
    """CARM-style PCA + STFT spectrogram of body motion.

    Args:
        band_matrix (np.ndarray): float[frames, bands] (recent history).
        frame_rate_hz (float): Frame rate of the rows.

    Returns:
        dict | None: {"freqs", "times", "power" [freq, time], "speed_index"}
    """
    frames = int(SPECTROGRAM_SECONDS * frame_rate_hz)
    if band_matrix.shape[0] < frames:
        return None
    segment = band_matrix[-frames:]
    segment = segment - segment.mean(axis=0)
    _, _, components = np.linalg.svd(segment, full_matrices=False)
    # Drop the first principal component (CARM): keep components 2-4.
    projected = segment @ components[1:4].T
    freqs, times, coefficients = stft(
        projected.T, fs=frame_rate_hz, nperseg=SPECTROGRAM_SEGMENT,
        noverlap=SPECTROGRAM_OVERLAP, boundary=None)
    power = (np.abs(coefficients) ** 2).sum(axis=0)
    keep = (freqs > 0) & (freqs <= SPECTROGRAM_MAX_HZ)
    power = power[keep]
    freqs = freqs[keep]
    total = power.sum()
    speed_index = float((freqs[:, None] * power).sum() / total) if total > 0 \
        else 0.0
    band_edges = SPEED_PROFILE_BANDS_HZ
    band_energy = [float(power[(freqs > lo) & (freqs <= hi)].sum())
                   for lo, hi in band_edges]
    energy_total = sum(band_energy) or 1.0
    speed_profile = [round(e / energy_total, 3) for e in band_energy]
    return {"freqs": freqs.tolist(), "times": (times - times[-1]).tolist(),
            "speed_profile": speed_profile,
            "power": np.log10(power + 1e-6).tolist(),
            "speed_index": speed_index}


class CsiAnalytics:
    """Rolling per-receiver history plus the analyses above."""

    def __init__(self, frame_rate_hz=50.0):
        """Create empty histories.

        Args:
            frame_rate_hz (float): Nominal per-device frame rate.
        """
        self.frame_rate_hz = float(frame_rate_hz)
        self._history = {}
        self._timestamps = {}
        self._max_frames = int(ANALYTICS_HISTORY_SECONDS * self.frame_rate_hz)

    def ingest(self, device_id, band_vector, timestamp_seconds=None):
        """Append one Hampel-cleaned band vector for a device.

        Args:
            device_id (int): Receiver id.
            band_vector (np.ndarray): float[bands].
            timestamp_seconds (float | None): Sender timestamp. Real CSI
                arrives with jitter and drops; with timestamps the history
                is resampled onto a uniform grid before any spectral
                analysis, so rates are not skewed by an uneven frame rate.
        """
        history = self._history.setdefault(
            device_id, deque(maxlen=self._max_frames))
        stamps = self._timestamps.setdefault(
            device_id, deque(maxlen=self._max_frames))
        if timestamp_seconds is None:
            timestamp_seconds = (stamps[-1] + 1.0 / self.frame_rate_hz
                                 if stamps else 0.0)
        elif stamps and timestamp_seconds <= stamps[-1] - 60.0:
            # 32-bit microsecond counter wrapped (~71.6 min on ESP32).
            timestamp_seconds += 2 ** 32 / 1e6 * (
                1 + int((stamps[-1] - timestamp_seconds) // (2 ** 32 / 1e6)))
        if stamps and timestamp_seconds <= stamps[-1]:
            return  # duplicate or out-of-order frame
        history.append(np.asarray(band_vector, dtype=np.float32))
        stamps.append(float(timestamp_seconds))

    def _uniform_matrix(self, device_id):
        """History resampled to frame_rate_hz on the sender's clock."""
        matrix = np.stack(self._history[device_id])
        stamps = np.asarray(self._timestamps[device_id])
        if len(stamps) < 2:
            return matrix
        grid = np.arange(stamps[0], stamps[-1], 1.0 / self.frame_rate_hz)
        if len(grid) < 2:
            return matrix
        return np.stack([np.interp(grid, stamps, matrix[:, band])
                         for band in range(matrix.shape[1])], axis=1)

    def compute(self, baselines=None):
        """Run all analyses on the current histories.

        Args:
            baselines (dict | None): device_id -> (mean, std) empty-room
                band baselines, used for the attenuation heatmap.

        Returns:
            dict: breathing (fused), per-device breathing, spectrogram of
                the most active receiver, heatmap, link activity.
        """
        per_device_breathing = {}
        link_activity = {}
        link_stats = {}
        best_spectrogram, best_speed = None, -1.0
        heatmap = None
        for device_id in sorted(self._history):
            matrix = self._uniform_matrix(device_id)
            recent = matrix[-int(self.frame_rate_hz):]
            link_activity[device_id] = float(recent.mean(axis=1).std())
            stamps = np.asarray(self._timestamps[device_id])
            if len(stamps) > 10:
                gaps = np.diff(stamps[-int(5 * self.frame_rate_hz):])
                link_stats[device_id] = {
                    "rate_hz": round(float(1.0 / max(gaps.mean(), 1e-6)), 1),
                    "jitter_ms": round(float(gaps.std() * 1000), 1),
                    "loss_pct": round(float(max(0.0, 1 - (1 / self.frame_rate_hz) / max(gaps.mean(), 1e-6)) * 100), 1)}
            breathing = estimate_breathing(matrix, self.frame_rate_hz)
            if breathing is not None:
                per_device_breathing[device_id] = breathing
            spectrogram = motion_spectrogram(matrix, self.frame_rate_hz)
            if spectrogram is not None and \
                    link_activity[device_id] > best_speed:
                best_speed = link_activity[device_id]
                best_spectrogram = dict(spectrogram, device_id=device_id)
            if heatmap is None:
                heatmap = self._heatmap(device_id, matrix, baselines)
        fused = None
        valid = {d: b for d, b in per_device_breathing.items()
                 if b["bnr"] >= BREATHING_MIN_BNR}
        if valid:
            # Receivers can disagree (one link may barely see the chest);
            # a weighted mean of two different peaks is a number neither
            # link measured, so report the cleanest link (highest BNR), or
            # the median once three or more receivers agree on a peak.
            best = max(valid.items(), key=lambda item: item[1]["bnr"])
            bpms = np.array([b["bpm"] for b in valid.values()])
            bpm = float(np.median(bpms)) if len(valid) >= 3 \
                else best[1]["bpm"]
            fused = {"bpm": bpm, "bnr": best[1]["bnr"],
                     "waveform": best[1]["waveform"][
                         int(2 * best[1]["waveform_rate_hz"]):].tolist(),
                     "waveform_rate_hz": best[1]["waveform_rate_hz"],
                     "devices": len(valid),
                     "device_id": best[0],
                     "window_seconds": best[1]["still_seconds"]}
        return {"breathing": fused,
                "breathing_per_device": {
                    d: round(b["bpm"], 1)
                    for d, b in per_device_breathing.items()},
                "spectrogram": best_spectrogram,
                "heatmap": heatmap,
                "link_activity": link_activity,
                "link_stats": link_stats}

    def _heatmap(self, device_id, matrix, baselines):
        """Attenuation per band over the last few seconds (one receiver)."""
        frames = int(HEATMAP_SECONDS * self.frame_rate_hz)
        recent = matrix[-frames:]
        if baselines and device_id in baselines:
            baseline_mean = np.maximum(baselines[device_id][0], 1e-6)
            values = 1.0 - recent / baseline_mean
        else:
            values = recent / max(float(recent.mean()), 1e-6) - 1.0
        columns = min(HEATMAP_COLUMNS, values.shape[0])
        usable = values.shape[0] - values.shape[0] % columns
        binned = values[-usable:].reshape(columns, -1, values.shape[1]).mean(
            axis=1)
        return {"device_id": device_id, "values": binned.T.tolist(),
                "seconds": HEATMAP_SECONDS}
