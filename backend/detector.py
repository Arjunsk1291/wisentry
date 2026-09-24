"""Presence + pose detection state machine and shared system state.

Two responsibilities:

1. SystemState — a thread-safe snapshot store. The pipeline thread writes
   into it; the dashboard and main loop read consistent snapshots out of
   it. This is the single hand-off point between backend and UI.

2. PresencePoseDetector — turns FeatureWindows (and, when available, ML
   inference results) into debounced OCCUPIED/EMPTY and pose decisions
   with hysteresis, so the reported state never flickers frame-to-frame.
   When no trained models exist it falls back to rule-based thresholds on
   motion energy and baseline deviation.

Runs on: any OS with Python 3.9+. Dependencies: numpy, backend.skeleton,
backend.ml_engine.
"""

import threading
import time
from collections import deque

import numpy as np

from backend.ml_engine import POSE_CLASS_LABELS
from backend.skeleton import SkeletonEstimator

EVENT_KIND_ENTERED = "entered"
EVENT_KIND_LEFT = "left"
EVENT_KIND_POSE_CHANGE = "pose_change"
WAVEFORM_HISTORY_POINTS = 256  # ~5 s of per-device mean amplitude at 50 Hz
RULE_PRESENCE_PROBABILITY_ON = 0.9   # confidence reported by rule hits
RULE_PRESENCE_PROBABILITY_OFF = 0.1
RULE_WALKING_MOTION_FACTOR = 4.0     # motion >= factor*threshold => walking
RULE_BREATHING_PRESENCE_BONUS = 0.35  # breathing evidence lowers the bar
POSITION_SMOOTHING_ALPHA = 0.2       # EMA for the room-map position dot
POSE_PROBABILITY_EMA_ALPHA = 0.3     # temporal smoothing of pose votes
# Physics gate: walking needs body motion. When every receiver reports
# near-zero motion energy, a "walking" vote is a classifier error (seen in
# end-to-end runs: a still person after walking in was held as walking).
WALKING_MIN_MOTION_ENERGY = 0.12


class SystemState:
    """Thread-safe shared state between pipeline, detector, and dashboard."""

    def __init__(self, event_log_max_entries=30):
        """Create empty state.

        Args:
            event_log_max_entries (int): Cap on the dashboard event log.
        """
        self._lock = threading.Lock()
        self.simulation_mode = False
        self.event_listeners = []  # callables(kind, message), e.g. CSV logger
        self._presence = False
        self._presence_confidence = 0.0
        self._pose_label = None
        self._pose_confidence = 0.0
        self._keypoints = None
        self._position_estimate = None
        self._events = deque(maxlen=event_log_max_entries)
        self._device_stats = {}
        self._device_heard = {}
        self._waveforms = {}
        self._analytics = {}
        self._frames_processed = 0
        self._windows_processed = 0

    def record_frame(self, csi_frame, sender_ip):
        """Update per-device statistics and waveform history for one frame.

        Args:
            csi_frame (CsiFrame): The parsed frame.
            sender_ip (str): Source IP of the UDP datagram.
        """
        with self._lock:
            self._frames_processed += 1
            stats = self._device_stats.setdefault(
                csi_frame.device_id,
                {"ip": sender_ip, "rssi": 0, "packets": 0, "last_seen": 0.0},
            )
            stats["ip"] = sender_ip
            stats["rssi"] = csi_frame.rssi_dbm
            stats["packets"] += 1
            stats["last_seen"] = csi_frame.receive_time
            waveform = self._waveforms.setdefault(
                csi_frame.device_id, deque(maxlen=WAVEFORM_HISTORY_POINTS)
            )
            waveform.append(
                (csi_frame.receive_time, float(csi_frame.amplitudes.mean()))
            )

    def update_detection(
        self,
        presence,
        presence_confidence,
        pose_label,
        pose_confidence,
        keypoints,
        position_estimate,
    ):
        """Store the detector's latest decision (called from pipeline)."""
        with self._lock:
            self._windows_processed += 1
            self._presence = presence
            self._presence_confidence = presence_confidence
            self._pose_label = pose_label
            self._pose_confidence = pose_confidence
            self._keypoints = keypoints
            self._position_estimate = position_estimate

    def update_analytics(self, analytics):
        """Store the latest CsiAnalytics.compute() result."""
        with self._lock:
            self._analytics = analytics

    def add_event(self, event_kind, message):
        """Append a timestamped event and notify registered listeners."""
        with self._lock:
            self._events.append((time.time(), event_kind, message))
            listeners = list(self.event_listeners)
        for listener_callable in listeners:
            listener_callable(event_kind, message)

    def note_heard(self, device_id, sender_ip, receive_time):
        """Record that a device's datagram reached the laptop (liveness).

        Called from the UDP receive thread, ahead of processing, so the
        dashboard can tell "device offline" apart from "pipeline behind".
        """
        with self._lock:
            self._device_heard[device_id] = (receive_time, sender_ip)

    def snapshot(self):
        """Return a consistent copy of everything the dashboard needs.

        Returns:
            dict: presence, confidences, pose, keypoints, events, device
                stats, waveforms, frame counters, simulation flag.
        """
        with self._lock:
            return {
                "simulation_mode": self.simulation_mode,
                "presence": self._presence,
                "presence_confidence": self._presence_confidence,
                "pose_label": self._pose_label,
                "pose_confidence": self._pose_confidence,
                "keypoints": (
                    None if self._keypoints is None else self._keypoints.copy()
                ),
                "position_estimate": self._position_estimate,
                "events": list(self._events),
                "device_stats": {
                    device_id: dict(
                        stats,
                        last_heard=self._device_heard.get(
                            device_id, (stats["last_seen"], None))[0],
                    )
                    for device_id, stats in self._device_stats.items()
                },
                "waveforms": {
                    device_id: list(points)
                    for device_id, points in self._waveforms.items()
                },
                "frames_processed": self._frames_processed,
                "analytics": self._analytics,
                "windows_processed": self._windows_processed,
            }


def apply_walking_motion_gate(pose_probabilities, motion_energy):
    """Suppress the walking class when no receiver sees body motion.

    Args:
        pose_probabilities (np.ndarray): float[4] in POSE_CLASS_LABELS order.
        motion_energy (float): Highest recent motion energy across devices.

    Returns:
        np.ndarray: Renormalised probabilities (unchanged if motion present).
    """
    probabilities = np.asarray(pose_probabilities, dtype=np.float64).copy()
    if motion_energy >= WALKING_MIN_MOTION_ENERGY:
        return probabilities
    walking_index = POSE_CLASS_LABELS.index("walking")
    probabilities[walking_index] = 0.0
    total = probabilities.sum()
    if total <= 1e-9:
        return np.asarray(pose_probabilities, dtype=np.float64)
    return probabilities / total


class PresencePoseDetector:
    """Debounced presence + pose state machine with rule-based fallback."""

    def __init__(self, detection_config, room_config, ml_engine, system_state):
        """Wire the detector to its config, models, and shared state.

        Args:
            detection_config (dict): `detection` section of config.yaml.
            room_config (dict): `room` section (for position estimation).
            ml_engine (MlEngine): May report models_available == False.
            system_state (SystemState): Where decisions are published.
        """
        self.rule_motion_threshold = float(
            detection_config.get("rule_motion_threshold", 1.2)
        )
        self.presence_on_count = int(detection_config.get("presence_on_count", 3))
        self.presence_off_count = int(
            detection_config.get("presence_off_count", 10)
        )
        self.pose_confidence_threshold = float(
            detection_config.get("pose_confidence_threshold", 0.6)
        )
        self.pose_change_count = int(detection_config.get("pose_change_count", 3))
        self._device_positions = room_config.get("device_positions", {})
        self._ml_engine = ml_engine
        self._state = system_state
        self._skeleton_estimator = SkeletonEstimator()
        self._presence = False
        self._consecutive_on_votes = 0
        self._consecutive_off_votes = 0
        self._current_pose = None
        self._pose_candidate = None
        self._pose_candidate_votes = 0
        self._smoothed_position = None
        self._last_deviation_per_device = {}
        self._last_pose_probabilities_per_device = {}
        self._last_motion_per_device = {}
        self._last_presence_per_device = {}
        self._pose_probability_ema = None

    def _rule_based_scores(self, feature_window):
        """Score presence/pose from thresholds when no ML is available.

        Args:
            feature_window (FeatureWindow): Current window.

        Returns:
            tuple[float, np.ndarray]: (presence probability, pose probs).
        """
        motion = feature_window.motion_energy
        deviation = feature_window.baseline_deviation
        breathing = feature_window.breathing_energy
        presence_evidence = (
            motion > self.rule_motion_threshold
            or deviation > 1.0
            or (deviation > 0.5 and breathing > RULE_BREATHING_PRESENCE_BONUS)
        )
        presence_probability = (
            RULE_PRESENCE_PROBABILITY_ON
            if presence_evidence
            else RULE_PRESENCE_PROBABILITY_OFF
        )
        pose_probabilities = np.full(len(POSE_CLASS_LABELS), 0.05)
        walking_threshold = RULE_WALKING_MOTION_FACTOR * self.rule_motion_threshold
        if motion >= walking_threshold:
            pose_probabilities[POSE_CLASS_LABELS.index("walking")] = 0.85
        elif deviation > 2.0:
            pose_probabilities[POSE_CLASS_LABELS.index("standing")] = 0.85
        elif deviation > 1.0:
            pose_probabilities[POSE_CLASS_LABELS.index("sitting")] = 0.85
        else:
            pose_probabilities[POSE_CLASS_LABELS.index("lying")] = 0.85
        return presence_probability, pose_probabilities / pose_probabilities.sum()

    def _update_presence(self, presence_probability):
        """Apply hysteresis to the per-window presence vote."""
        if presence_probability >= 0.5:
            self._consecutive_on_votes += 1
            self._consecutive_off_votes = 0
        else:
            self._consecutive_off_votes += 1
            self._consecutive_on_votes = 0
        if not self._presence and self._consecutive_on_votes >= self.presence_on_count:
            self._presence = True
            self._state.add_event(EVENT_KIND_ENTERED, "Person entered the room")
        elif self._presence and self._consecutive_off_votes >= self.presence_off_count:
            self._presence = False
            self._current_pose = None
            self._pose_candidate = None
            self._pose_candidate_votes = 0
            self._last_pose_probabilities_per_device.clear()
            self._pose_probability_ema = None
            self._state.add_event(EVENT_KIND_LEFT, "Room is now empty")

    def _update_pose(self, pose_probabilities):
        """Debounce pose changes; only confident, repeated votes win."""
        best_pose_index = int(np.argmax(pose_probabilities))
        best_confidence = float(pose_probabilities[best_pose_index])
        best_label = POSE_CLASS_LABELS[best_pose_index]
        if best_confidence < self.pose_confidence_threshold:
            return best_confidence
        if best_label == self._current_pose:
            self._pose_candidate = None
            self._pose_candidate_votes = 0
            return best_confidence
        if best_label == self._pose_candidate:
            self._pose_candidate_votes += 1
        else:
            self._pose_candidate = best_label
            self._pose_candidate_votes = 1
        if self._pose_candidate_votes >= self.pose_change_count:
            previous_pose = self._current_pose
            self._current_pose = best_label
            self._pose_candidate = None
            self._pose_candidate_votes = 0
            if previous_pose is not None:
                self._state.add_event(
                    EVENT_KIND_POSE_CHANGE,
                    f"Pose changed: {previous_pose} → {best_label}",
                )
            else:
                self._state.add_event(
                    EVENT_KIND_POSE_CHANGE, f"Pose detected: {best_label}"
                )
        return best_confidence

    def _estimate_position(self):
        """Estimate the person's room position from per-device deviation.

        Weighted centroid of receiver positions, weighted by how disturbed
        each receiver's signal is, smoothed with an EMA. Crude by design —
        honest single-dot localization, not tracking.

        Returns:
            tuple[float, float] | None: (x, y) meters, or None when empty.
        """
        if not self._presence or not self._last_deviation_per_device:
            self._smoothed_position = None
            return None
        weight_sum = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        for device_id, deviation in self._last_deviation_per_device.items():
            device_info = self._device_positions.get(device_id)
            if device_info is None:
                continue
            weight = max(deviation, 0.05)
            weight_sum += weight
            weighted_x += weight * float(device_info["x"])
            weighted_y += weight * float(device_info["y"])
        if weight_sum == 0.0:
            return self._smoothed_position
        raw_position = (weighted_x / weight_sum, weighted_y / weight_sum)
        if self._smoothed_position is None:
            self._smoothed_position = raw_position
        else:
            alpha = POSITION_SMOOTHING_ALPHA
            self._smoothed_position = (
                (1 - alpha) * self._smoothed_position[0] + alpha * raw_position[0],
                (1 - alpha) * self._smoothed_position[1] + alpha * raw_position[1],
            )
        return self._smoothed_position

    def process_window(self, feature_window):
        """Consume one FeatureWindow and publish the updated decision.

        Args:
            feature_window (FeatureWindow): Output of SignalProcessor.
        """
        self._last_deviation_per_device[feature_window.device_id] = max(
            feature_window.baseline_deviation, feature_window.motion_energy
        )
        inference_result = self._ml_engine.infer(feature_window)
        if inference_result is not None:
            presence_probability = inference_result.presence_probability
            pose_probabilities = inference_result.pose_probabilities
            skeleton_offsets = inference_result.skeleton_offsets
        else:
            presence_probability, pose_probabilities = self._rule_based_scores(
                feature_window
            )
            skeleton_offsets = None
        # Fuse presence across receivers (median of each device's latest
        # probability) so one receiver with a biased model cannot hold the
        # room "occupied" forever by breaking the off-vote streak.
        self._last_presence_per_device[feature_window.device_id] = float(
            presence_probability)
        fused_presence_probability = float(np.median(
            list(self._last_presence_per_device.values())))
        self._update_presence(fused_presence_probability)
        # Pose votes from different receivers can disagree window-to-window;
        # average the latest probabilities across devices so one flaky
        # receiver cannot flip the reported pose.
        self._last_pose_probabilities_per_device[feature_window.device_id] = (
            pose_probabilities
        )
        fused_pose_probabilities = np.mean(
            list(self._last_pose_probabilities_per_device.values()), axis=0
        )
        self._last_motion_per_device[feature_window.device_id] = (
            feature_window.motion_energy
        )
        fused_pose_probabilities = apply_walking_motion_gate(
            fused_pose_probabilities,
            max(self._last_motion_per_device.values()),
        )
        # Temporal EMA on top of device fusion: breathing-induced classifier
        # wobble is faster than real pose changes, so smooth it out.
        if self._pose_probability_ema is None:
            self._pose_probability_ema = fused_pose_probabilities
        else:
            alpha = POSE_PROBABILITY_EMA_ALPHA
            self._pose_probability_ema = (
                (1 - alpha) * self._pose_probability_ema
                + alpha * fused_pose_probabilities
            )
        pose_confidence = 0.0
        keypoints = None
        if self._presence:
            pose_confidence = self._update_pose(self._pose_probability_ema)
            if self._current_pose is not None:
                keypoints = self._skeleton_estimator.estimate(
                    self._current_pose,
                    feature_window.motion_energy,
                    skeleton_offsets,
                )
        self._state.update_detection(
            presence=self._presence,
            presence_confidence=(
                fused_presence_probability if self._presence
                else 1 - fused_presence_probability
            ),
            pose_label=self._current_pose,
            pose_confidence=pose_confidence,
            keypoints=keypoints,
            position_estimate=self._estimate_position(),
        )


if __name__ == "__main__":
    from backend.signal_processor import FeatureWindow

    class _NoMlEngine:
        models_available = False

        def infer(self, _):
            return None

    state = SystemState()
    detector = PresencePoseDetector(
        detection_config={"presence_on_count": 3, "presence_off_count": 5},
        room_config={"device_positions": {1: {"x": 4.7, "y": 0.5, "label": "RX-1"}}},
        ml_engine=_NoMlEngine(),
        system_state=state,
    )

    def make_window(motion, deviation):
        return FeatureWindow(
            device_id=1,
            band_matrix=np.zeros((50, 16), dtype=np.float32),
            motion_energy=motion,
            baseline_deviation=deviation,
            breathing_energy=0.0,
            mean_amplitude=20.0,
            timestamp=time.time(),
        )

    for _ in range(10):
        detector.process_window(make_window(motion=0.1, deviation=0.1))
    assert state.snapshot()["presence"] is False
    for _ in range(5):
        detector.process_window(make_window(motion=8.0, deviation=2.5))
    assert state.snapshot()["presence"] is True
    assert state.snapshot()["pose_label"] == "walking"
    for _ in range(10):
        detector.process_window(make_window(motion=0.1, deviation=0.1))
    assert state.snapshot()["presence"] is False
    events = [kind for _, kind, _ in state.snapshot()["events"]]
    assert events[0] == EVENT_KIND_ENTERED and events[-1] == EVENT_KIND_LEFT
    print("detector self-test: PASS (hysteresis, pose, events)")
