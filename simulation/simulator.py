"""Synthetic CSI generator — full-pipeline test without hardware.

Emits protocol-v1 UDP datagrams to the real UDP server on localhost, so
everything downstream (server, parser, signal processing, detection) is
exercised exactly as it would be with hardware. The generated CSI carries
the physics the ML models learn from (ENGINEERING_SPEC.md §9.1):

  * static per-device multipath baseline (Rician-like) + sensor noise
  * body shadowing: a pose-dependent attenuation dip across subcarriers
  * breathing modulation (~0.3 Hz) when the person is still
  * large erratic fluctuations and a moving dip while walking
  * ~1% dropped packets and occasional out-of-order delivery

The same physics functions are imported by models/train_all.py to build
labeled synthetic training data, so training and runtime distributions
match by construction.

Runs on: any OS with Python 3.9+. Dependencies: numpy, backend.csi_parser.
"""

import socket
import threading
import time

import numpy as np

from backend.csi_parser import build_csi_datagram

SUBCARRIER_COUNT = 64
BASELINE_MEAN_AMPLITUDE = 35.0     # comfortably inside int8 after dips
BASELINE_RIPPLE_AMPLITUDE = 8.0    # static multipath ripple across subcarriers
SENSOR_NOISE_SIGMA = 1.0
BREATHING_FREQUENCY_HZ = 0.3
PACKET_DROP_PROBABILITY = 0.01
OUT_OF_ORDER_PROBABILITY = 0.005
SIMULATED_RSSI_DBM = -48

# Pose signatures: (dip depth multiplier, dip width in subcarriers,
# extra motion noise sigma, breathing modulation depth).
POSE_SIGNATURES = {
    "standing": {"depth": 0.45, "width": 6.0, "motion": 0.4, "breathing": 2.0},
    "sitting": {"depth": 0.30, "width": 10.0, "motion": 0.3, "breathing": 2.4},
    "lying": {"depth": 0.18, "width": 18.0, "motion": 0.2, "breathing": 3.0},
    "walking": {"depth": 0.55, "width": 7.0, "motion": 4.5, "breathing": 0.0},
}

# The 30-second scripted scenario from ENGINEERING_SPEC.md §9:
# (start_second, end_second, occupied, pose or None)
SCENARIO_SCRIPT = [
    (0.0, 5.0, False, None),
    (5.0, 10.0, True, "walking"),    # person walks in
    (10.0, 15.0, True, "standing"),
    (15.0, 20.0, True, "sitting"),
    (20.0, 25.0, True, "lying"),
    (25.0, 30.0, True, "walking"),   # person walks out (still moving)
]
SCENARIO_LOOP_SECONDS = 30.0
# The person "leaves" at second 28 of each loop (mid walking-out segment),
# giving the detector empty air before the loop restarts.
SCENARIO_EXIT_SECOND = 28.0


def scenario_state_at(scenario_time_seconds):
    """Return the scripted ground truth at a point in the scenario loop.

    Args:
        scenario_time_seconds (float): Seconds since scenario start
            (wrapped into the 30 s loop).

    Returns:
        tuple[bool, str | None]: (occupied, pose label or None).
    """
    loop_time = scenario_time_seconds % SCENARIO_LOOP_SECONDS
    if loop_time >= SCENARIO_EXIT_SECOND:
        return False, None
    for segment_start, segment_end, occupied, pose_label in SCENARIO_SCRIPT:
        if segment_start <= loop_time < segment_end:
            return occupied, pose_label
    return False, None


def synthesize_csi_vector(
    device_baseline,
    occupied,
    pose_label,
    elapsed_seconds,
    dip_center_subcarrier,
    random_generator,
):
    """Generate one complex CSI vector for one device at one instant.

    Args:
        device_baseline (np.ndarray): complex[64] static multipath baseline.
        occupied (bool): Whether a person is present.
        pose_label (str | None): Current pose when occupied.
        elapsed_seconds (float): Continuous time for breathing phase.
        dip_center_subcarrier (float): Center of the body-shadow dip.
        random_generator (np.random.Generator): Source of randomness.

    Returns:
        np.ndarray: complex[64] CSI values ready for int8 quantisation.
    """
    csi_vector = device_baseline.copy()
    noise = random_generator.normal(0, SENSOR_NOISE_SIGMA, SUBCARRIER_COUNT)
    if occupied and pose_label is not None:
        signature = POSE_SIGNATURES[pose_label]
        subcarrier_indices = np.arange(SUBCARRIER_COUNT)
        dip_profile = signature["depth"] * np.exp(
            -((subcarrier_indices - dip_center_subcarrier) ** 2)
            / (2.0 * signature["width"] ** 2)
        )
        breathing_modulation = signature["breathing"] * np.sin(
            2.0 * np.pi * BREATHING_FREQUENCY_HZ * elapsed_seconds
        )
        motion_noise = random_generator.normal(
            0, signature["motion"], SUBCARRIER_COUNT
        )
        csi_vector = csi_vector * (1.0 - dip_profile)
        csi_vector = csi_vector + breathing_modulation * dip_profile * 10.0
        noise = noise + motion_noise
    return csi_vector + noise


def make_device_baseline(device_id):
    """Build the static multipath baseline for a simulated device.

    Args:
        device_id (int): Seeds the per-device ripple pattern.

    Returns:
        np.ndarray: complex[64] baseline (mostly real, mild phase slope).
    """
    baseline_generator = np.random.default_rng(1000 + device_id)
    subcarrier_indices = np.arange(SUBCARRIER_COUNT)
    ripple = BASELINE_RIPPLE_AMPLITUDE * np.sin(
        2.0 * np.pi * subcarrier_indices / SUBCARRIER_COUNT
        * baseline_generator.uniform(1.0, 3.0)
        + baseline_generator.uniform(0, 2 * np.pi)
    )
    amplitudes = BASELINE_MEAN_AMPLITUDE + ripple
    phase_slope = baseline_generator.uniform(-0.05, 0.05)
    phases = phase_slope * subcarrier_indices
    return amplitudes * np.exp(1j * phases)


class CsiSimulator(threading.Thread):
    """Thread that streams scripted synthetic CSI datagrams over UDP."""

    def __init__(self, simulation_config, network_config, random_seed=None):
        """Configure the simulator.

        Args:
            simulation_config (dict): `simulation` section of config.yaml.
            network_config (dict): `network` section (for the target port).
            random_seed (int | None): Fix for reproducible tests.
        """
        super().__init__(name="CsiSimulator", daemon=True)
        self.frame_rate_hz = float(simulation_config.get("frame_rate_hz", 50.0))
        self.device_count = int(simulation_config.get("simulated_devices", 2))
        self.target_address = (
            "127.0.0.1",
            int(network_config.get("udp_listen_port", 5566)),
        )
        self._random_generator = np.random.default_rng(random_seed)
        self._stop_event = threading.Event()
        self._sequence_numbers = {device: 0 for device in
                                  range(1, self.device_count + 1)}
        self._baselines = {device: make_device_baseline(device)
                           for device in range(1, self.device_count + 1)}
        self._held_back_datagram = None
        self.sent_datagram_count = 0

    def _build_datagram_for_device(self, device_id, elapsed_seconds):
        """Synthesize and serialize one frame for one device."""
        occupied, pose_label = scenario_state_at(elapsed_seconds)
        dip_center = SUBCARRIER_COUNT / 2.0
        if pose_label == "walking":
            walk_phase = elapsed_seconds * 0.8 + device_id
            dip_center = (SUBCARRIER_COUNT / 2.0) * (1.0 + 0.6 *
                                                     np.sin(walk_phase))
        csi_vector = synthesize_csi_vector(
            self._baselines[device_id], occupied, pose_label,
            elapsed_seconds, dip_center, self._random_generator,
        )
        self._sequence_numbers[device_id] += 1
        return build_csi_datagram(
            device_id=device_id,
            sequence_number=self._sequence_numbers[device_id],
            esp32_timestamp_us=int(elapsed_seconds * 1_000_000),
            rssi_dbm=SIMULATED_RSSI_DBM,
            csi_complex_values=csi_vector,
        )

    def _send_with_impairments(self, sender_socket, datagram):
        """Send a datagram, simulating drops and out-of-order delivery."""
        roll = self._random_generator.random()
        if roll < PACKET_DROP_PROBABILITY:
            return  # dropped in the air
        if roll < PACKET_DROP_PROBABILITY + OUT_OF_ORDER_PROBABILITY:
            if self._held_back_datagram is None:
                self._held_back_datagram = datagram
                return
        sender_socket.sendto(datagram, self.target_address)
        self.sent_datagram_count += 1
        if self._held_back_datagram is not None:
            sender_socket.sendto(self._held_back_datagram, self.target_address)
            self.sent_datagram_count += 1
            self._held_back_datagram = None

    def run(self):
        """Stream frames for all devices at the configured rate."""
        sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        frame_interval = 1.0 / self.frame_rate_hz
        start_time = time.monotonic()
        next_frame_time = start_time
        while not self._stop_event.is_set():
            elapsed_seconds = time.monotonic() - start_time
            for device_id in range(1, self.device_count + 1):
                datagram = self._build_datagram_for_device(
                    device_id, elapsed_seconds
                )
                self._send_with_impairments(sender_socket, datagram)
            next_frame_time += frame_interval
            sleep_duration = next_frame_time - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        sender_socket.close()

    def stop(self):
        """Request the streaming loop to exit."""
        self._stop_event.set()


if __name__ == "__main__":
    from backend.csi_parser import parse_csi_datagram

    assert scenario_state_at(2.0) == (False, None)
    assert scenario_state_at(7.0) == (True, "walking")
    assert scenario_state_at(17.0) == (True, "sitting")
    assert scenario_state_at(29.0) == (False, None)
    assert scenario_state_at(32.0) == (False, None)  # loop wraps

    test_simulator = CsiSimulator(
        {"frame_rate_hz": 50.0, "simulated_devices": 2},
        {"udp_listen_port": 5566},
        random_seed=7,
    )
    test_datagram = test_simulator._build_datagram_for_device(1, 12.0)
    parsed = parse_csi_datagram(test_datagram)
    assert parsed.device_id == 1
    assert len(parsed.amplitudes) == SUBCARRIER_COUNT
    assert 5.0 < parsed.amplitudes.mean() < 60.0
    print("simulator self-test: PASS (scenario script + datagram round-trip)")
