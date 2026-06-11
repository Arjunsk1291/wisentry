"""Unit tests for the synthetic CSI generator."""

import numpy as np

from backend.csi_parser import parse_csi_datagram
from simulation.simulator import (
    SCENARIO_LOOP_SECONDS,
    SUBCARRIER_COUNT,
    CsiSimulator,
    make_device_baseline,
    scenario_state_at,
    synthesize_csi_vector,
)


def test_scenario_script_timeline():
    assert scenario_state_at(2.0) == (False, None)
    assert scenario_state_at(6.0) == (True, "walking")
    assert scenario_state_at(12.0) == (True, "standing")
    assert scenario_state_at(17.0) == (True, "sitting")
    assert scenario_state_at(22.0) == (True, "lying")
    assert scenario_state_at(26.0) == (True, "walking")
    assert scenario_state_at(29.0) == (False, None)


def test_scenario_loops():
    for offset in (0.0, SCENARIO_LOOP_SECONDS, 3 * SCENARIO_LOOP_SECONDS):
        assert scenario_state_at(12.0 + offset) == (True, "standing")


def test_datagrams_parse_and_look_physical():
    simulator = CsiSimulator(
        {"frame_rate_hz": 50.0, "simulated_devices": 3},
        {"udp_listen_port": 5566},
        random_seed=1,
    )
    for device_id in (1, 2, 3):
        datagram = simulator._build_datagram_for_device(device_id, 12.0)
        frame = parse_csi_datagram(datagram)
        assert frame.device_id == device_id
        assert len(frame.amplitudes) == SUBCARRIER_COUNT
        assert 5.0 < frame.amplitudes.mean() < 60.0


def test_occupied_room_disturbs_amplitudes():
    rng = np.random.default_rng(3)
    baseline = make_device_baseline(1)
    empty_vector = synthesize_csi_vector(baseline, False, None, 1.0, 32.0, rng)
    occupied_vector = synthesize_csi_vector(
        baseline, True, "standing", 1.0, 32.0, rng
    )
    empty_amplitudes = np.abs(empty_vector)
    occupied_amplitudes = np.abs(occupied_vector)
    center_band = slice(24, 40)  # where the body-shadow dip is centred
    assert (
        occupied_amplitudes[center_band].mean()
        < empty_amplitudes[center_band].mean() - 2.0
    )


def test_walking_has_more_temporal_variance_than_standing():
    rng = np.random.default_rng(4)
    baseline = make_device_baseline(1)

    def temporal_std(pose_label):
        means = [
            np.abs(
                synthesize_csi_vector(
                    baseline, True, pose_label, t * 0.02, 32.0, rng
                )
            ).mean()
            for t in range(100)
        ]
        return np.std(np.diff(means))

    assert temporal_std("walking") > 2.0 * temporal_std("standing")
