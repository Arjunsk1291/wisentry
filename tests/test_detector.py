"""Unit tests for the presence/pose state machine."""

import time

import numpy as np

from backend.detector import (
    EVENT_KIND_ENTERED,
    EVENT_KIND_LEFT,
    PresencePoseDetector,
    SystemState,
)
from backend.signal_processor import FeatureWindow


class _NoMlEngine:
    """Stands in for MlEngine when no models are trained."""

    models_available = False

    def infer(self, _feature_window):
        return None


def _make_detector(state):
    return PresencePoseDetector(
        detection_config={
            "rule_motion_threshold": 1.2,
            "presence_on_count": 3,
            "presence_off_count": 5,
            "pose_confidence_threshold": 0.6,
            "pose_change_count": 3,
        },
        room_config={
            "device_positions": {1: {"x": 4.0, "y": 1.0, "label": "RX-1"}}
        },
        ml_engine=_NoMlEngine(),
        system_state=state,
    )


def _window(motion, deviation, breathing=0.0, device_id=1):
    return FeatureWindow(
        device_id=device_id,
        band_matrix=np.zeros((50, 16), dtype=np.float32),
        motion_energy=motion,
        baseline_deviation=deviation,
        breathing_energy=breathing,
        mean_amplitude=20.0,
        timestamp=time.time(),
    )


def test_presence_requires_consecutive_votes():
    state = SystemState()
    detector = _make_detector(state)
    detector.process_window(_window(motion=9.0, deviation=3.0))
    detector.process_window(_window(motion=9.0, deviation=3.0))
    assert state.snapshot()["presence"] is False  # only 2 of 3 votes
    detector.process_window(_window(motion=9.0, deviation=3.0))
    assert state.snapshot()["presence"] is True


def test_single_quiet_window_does_not_clear_presence():
    state = SystemState()
    detector = _make_detector(state)
    for _ in range(3):
        detector.process_window(_window(motion=9.0, deviation=3.0))
    detector.process_window(_window(motion=0.0, deviation=0.0))
    assert state.snapshot()["presence"] is True  # hysteresis holds


def test_full_enter_exit_cycle_emits_events():
    state = SystemState()
    detector = _make_detector(state)
    for _ in range(4):
        detector.process_window(_window(motion=9.0, deviation=3.0))
    for _ in range(6):
        detector.process_window(_window(motion=0.0, deviation=0.0))
    kinds = [kind for _, kind, _ in state.snapshot()["events"]]
    assert EVENT_KIND_ENTERED in kinds
    assert EVENT_KIND_LEFT in kinds


def test_pose_walking_when_motion_high():
    state = SystemState()
    detector = _make_detector(state)
    for _ in range(8):
        detector.process_window(_window(motion=9.0, deviation=3.0))
    assert state.snapshot()["pose_label"] == "walking"


def test_breathing_evidence_supports_presence():
    state = SystemState()
    detector = _make_detector(state)
    for _ in range(4):
        detector.process_window(
            _window(motion=0.2, deviation=0.7, breathing=0.5)
        )
    assert state.snapshot()["presence"] is True


def test_keypoints_published_when_pose_known():
    state = SystemState()
    detector = _make_detector(state)
    for _ in range(8):
        detector.process_window(_window(motion=9.0, deviation=3.0))
    keypoints = state.snapshot()["keypoints"]
    assert keypoints is not None and keypoints.shape == (17, 2)


def test_event_listeners_receive_events():
    state = SystemState()
    captured = []
    state.event_listeners.append(lambda kind, msg: captured.append(kind))
    detector = _make_detector(state)
    for _ in range(4):
        detector.process_window(_window(motion=9.0, deviation=3.0))
    assert EVENT_KIND_ENTERED in captured


def test_walking_gate_suppresses_walking_without_motion():
    import numpy as np
    from backend.detector import apply_walking_motion_gate
    gated = apply_walking_motion_gate(np.array([0.2, 0.1, 0.0, 0.7]), 0.03)
    assert gated[3] == 0.0
    assert abs(gated.sum() - 1.0) < 1e-9
    assert int(np.argmax(gated)) == 0


def test_walking_gate_keeps_walking_with_motion():
    import numpy as np
    from backend.detector import apply_walking_motion_gate
    probabilities = np.array([0.2, 0.1, 0.0, 0.7])
    assert np.allclose(apply_walking_motion_gate(probabilities, 0.5),
                       probabilities)
