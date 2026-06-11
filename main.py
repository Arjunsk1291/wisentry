"""WiSentry entry point.

Usage:
    python main.py                       # live mode (hardware required)
    python main.py --simulate            # synthetic CSI + dashboard
    python main.py --simulate --headless # pipeline only, no browser UI
    python main.py --simulate --headless --duration 60
    python main.py --collect --label standing --duration 120

Starts the UDP server, the processing pipeline (signal processing →
ML/rule detection → state), optionally the simulator, and (unless
--headless) the Dash dashboard at http://localhost:8050.

Runs on: Windows 10/11, Ubuntu 20.04+. Dependencies: see requirements.txt.
"""

import argparse
import queue
import sys
import threading
import time

from backend.config_loader import ConfigError, load_config
from backend.data_logger import DataLogger
from backend.detector import PresencePoseDetector, SystemState
from backend.ml_engine import MlEngine
from backend.signal_processor import SignalProcessor
from backend.udp_server import UdpCsiServer

PIPELINE_QUEUE_POLL_SECONDS = 0.5
INFERENCE_STRIDE_FRAMES = 10  # run detection every Nth window per device
SHUTDOWN_JOIN_TIMEOUT_SECONDS = 3.0
COLLECT_LABEL_CHOICES = ["empty", "standing", "sitting", "lying", "walking"]


def parse_arguments():
    """Define and parse the command-line interface.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="WiSentry — WiFi CSI presence & pose detection"
    )
    parser.add_argument("--simulate", action="store_true",
                        help="run with synthetic CSI (no hardware needed)")
    parser.add_argument("--headless", action="store_true",
                        help="run without the web dashboard")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop automatically after N seconds")
    parser.add_argument("--config", default=None,
                        help="path to an alternate config.yaml")
    parser.add_argument("--collect", action="store_true",
                        help="record labeled feature windows for training")
    parser.add_argument("--label", choices=COLLECT_LABEL_CHOICES,
                        default=None,
                        help="ground-truth label for --collect mode")
    return parser.parse_args()


class Pipeline(threading.Thread):
    """Consumes parsed frames, produces detections, updates SystemState."""

    def __init__(self, config, udp_server, system_state, data_logger,
                 collect_label=None):
        """Wire the pipeline together.

        Args:
            config (dict): Full parsed config.yaml.
            udp_server (UdpCsiServer): Source of parsed frames.
            system_state (SystemState): Shared state for the dashboard.
            data_logger (DataLogger): Event/raw/dataset logging.
            collect_label (str | None): When set, record labeled windows
                instead of running detection.
        """
        super().__init__(name="Pipeline", daemon=True)
        self._udp_server = udp_server
        self._state = system_state
        self._logger = data_logger
        self._collect_label = collect_label
        self._signal_processor = SignalProcessor(config["signal"])
        self._ml_engine = MlEngine(config["ml"])
        self._detector = PresencePoseDetector(
            config["detection"], config["room"], self._ml_engine, system_state
        )
        self._stop_event = threading.Event()
        self._window_counter_per_device = {}
        self._collected_band_matrices = []
        self._collected_auxiliary = []
        self.frames_processed = 0
        self.windows_detected = 0

    def _detection_due(self, device_id):
        """Stride inference so CPU stays low at high frame rates."""
        counter = self._window_counter_per_device.get(device_id, 0) + 1
        self._window_counter_per_device[device_id] = counter
        return counter % INFERENCE_STRIDE_FRAMES == 0

    def _handle_window(self, feature_window):
        """Route a FeatureWindow to detection or dataset collection."""
        if self._collect_label is not None:
            self._collected_band_matrices.append(feature_window.band_matrix)
            self._collected_auxiliary.append([
                feature_window.motion_energy,
                feature_window.baseline_deviation,
                feature_window.breathing_energy,
            ])
            return
        self._detector.process_window(feature_window)
        self.windows_detected += 1

    def run(self):
        """Main pipeline loop: queue → processor → detector → state."""
        while not self._stop_event.is_set():
            try:
                csi_frame, sender_ip = self._udp_server.frame_queue.get(
                    timeout=PIPELINE_QUEUE_POLL_SECONDS
                )
            except queue.Empty:
                continue
            self.frames_processed += 1
            self._state.record_frame(csi_frame, sender_ip)
            self._logger.log_raw_frame(csi_frame)
            feature_window = self._signal_processor.add_frame(csi_frame)
            if feature_window is None:
                continue
            if self._detection_due(csi_frame.device_id):
                self._handle_window(feature_window)

    def finalize_collection(self):
        """Persist collected labeled windows (collect mode only).

        Returns:
            int: Number of windows saved.
        """
        if self._collect_label is None or not self._collected_band_matrices:
            return 0
        saved_path = self._logger.save_labeled_windows(
            self._collect_label,
            self._collected_band_matrices,
            self._collected_auxiliary,
        )
        print(f"collect: saved {len(self._collected_band_matrices)} windows "
              f"labeled '{self._collect_label}' to {saved_path}")
        return len(self._collected_band_matrices)

    def stop(self):
        """Request pipeline shutdown."""
        self._stop_event.set()


def start_simulator_if_requested(arguments, config, system_state):
    """Start the synthetic CSI stream when --simulate is given.

    Args:
        arguments (argparse.Namespace): Parsed CLI arguments.
        config (dict): Full configuration.
        system_state (SystemState): Marked with simulation_mode=True.

    Returns:
        CsiSimulator | None: The running simulator thread, if any.
    """
    if not arguments.simulate:
        return None
    from simulation.simulator import CsiSimulator

    system_state.simulation_mode = True
    simulator = CsiSimulator(config["simulation"], config["network"])
    simulator.start()
    print("simulator: streaming synthetic CSI "
          f"({simulator.device_count} devices @ {simulator.frame_rate_hz} Hz)")
    return simulator


def run_headless_wait(arguments, system_state):
    """Block until --duration elapses (or forever) in headless mode."""
    started = time.monotonic()
    try:
        while True:
            time.sleep(0.5)
            if arguments.duration is not None:
                if time.monotonic() - started >= arguments.duration:
                    return
    except KeyboardInterrupt:
        print("\nmain: interrupted by user.")


def print_summary(pipeline, udp_server, system_state):
    """Print an end-of-run report used by the validation gates."""
    snapshot = system_state.snapshot()
    print("-" * 50)
    print(f"frames received : {udp_server.received_datagram_count}")
    print(f"frames processed: {pipeline.frames_processed}")
    print(f"windows detected: {pipeline.windows_detected}")
    print(f"parse errors    : {udp_server.parse_error_count}")
    print(f"devices seen    : {sorted(snapshot['device_stats'])}")
    print("event history   :")
    for event_time, kind, message in snapshot["events"]:
        print(f"  {time.strftime('%H:%M:%S', time.localtime(event_time))} "
              f"[{kind}] {message}")


def main():
    """Program entry point.

    Returns:
        int: Process exit code (0 = clean run).
    """
    arguments = parse_arguments()
    if arguments.collect and arguments.label is None:
        print("error: --collect requires --label "
              f"(one of {COLLECT_LABEL_CHOICES})")
        return 2
    try:
        config = load_config(arguments.config)
    except ConfigError as config_error:
        print(f"error: {config_error}")
        return 1

    system_state = SystemState(
        event_log_max_entries=config["dashboard"]["event_log_max_entries"]
    )
    data_logger = DataLogger(config["logging"])
    system_state.event_listeners.append(data_logger.log_event)
    udp_server = UdpCsiServer(config["network"])
    pipeline = Pipeline(
        config, udp_server, system_state, data_logger,
        collect_label=arguments.label if arguments.collect else None,
    )
    udp_server.start()
    pipeline.start()
    simulator = start_simulator_if_requested(arguments, config, system_state)

    if arguments.headless or arguments.collect:
        run_headless_wait(arguments, system_state)
    else:
        from dashboard.app import run_dashboard
        run_dashboard(config, system_state, duration=arguments.duration)

    if simulator is not None:
        simulator.stop()
    pipeline.stop()
    udp_server.stop()
    pipeline.join(timeout=SHUTDOWN_JOIN_TIMEOUT_SECONDS)
    udp_server.join(timeout=SHUTDOWN_JOIN_TIMEOUT_SECONDS)
    pipeline.finalize_collection()
    data_logger.close()
    print_summary(pipeline, udp_server, system_state)
    if pipeline.frames_processed == 0:
        print("main: FAILED — no frames were processed.")
        return 1
    print("main: clean shutdown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
