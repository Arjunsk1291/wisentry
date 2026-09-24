"""Logging of detector events (CSV) and raw CSI captures (binary .npz).

Also implements the labeled dataset recorder used by collection mode
(`python main.py --collect --label standing`), which Phase 6 uses to
gather real training data.

Runs on: any OS with Python 3.9+. Dependencies: numpy.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
RAW_CAPTURE_FLUSH_FRAMES = 2000  # frames buffered before an .npz flush
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


class DataLogger:
    """Writes event CSVs, optional raw CSI captures, and labeled datasets."""

    def __init__(self, logging_config):
        """Prepare the log directory and open files lazily.

        Args:
            logging_config (dict): `logging` section of config.yaml
                (keys: log_dir, log_raw_csi, log_events).
        """
        self.log_directory = PROJECT_ROOT_DIRECTORY / logging_config.get(
            "log_dir", "logs"
        )
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.events_enabled = bool(logging_config.get("log_events", True))
        self.raw_enabled = bool(logging_config.get("log_raw_csi", False))
        session_stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        self._events_path = self.log_directory / f"events_{session_stamp}.csv"
        self._events_file = None
        self._events_writer = None
        self._raw_buffer = []
        self._raw_flush_index = 0
        self._session_stamp = session_stamp

    def _ensure_events_writer(self):
        """Open the events CSV on first use and write its header."""
        if self._events_writer is not None:
            return
        self._events_file = open(
            self._events_path, "w", newline="", encoding="utf-8"
        )
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow(["unix_time", "iso_time", "kind", "message"])

    def log_event(self, event_kind, message):
        """Append one detector event to the session CSV.

        Args:
            event_kind (str): "entered", "left", or "pose_change".
            message (str): Human-readable event description.
        """
        if not self.events_enabled:
            return
        self._ensure_events_writer()
        now = time.time()
        self._events_writer.writerow(
            [f"{now:.3f}", datetime.fromtimestamp(now).isoformat(), event_kind,
             message]
        )
        self._events_file.flush()

    def log_raw_frame(self, csi_frame):
        """Buffer one raw frame; flush to .npz periodically.

        Args:
            csi_frame (CsiFrame): The parsed frame to capture.
        """
        if not self.raw_enabled:
            return
        self._raw_buffer.append(
            (
                csi_frame.receive_time,
                csi_frame.device_id,
                csi_frame.sequence_number,
                csi_frame.rssi_dbm,
                csi_frame.amplitudes,
            )
        )
        if len(self._raw_buffer) >= RAW_CAPTURE_FLUSH_FRAMES:
            self.flush_raw_capture()

    def flush_raw_capture(self):
        """Write buffered raw frames to a compressed .npz file."""
        if not self._raw_buffer:
            return
        capture_path = self.log_directory / (
            f"csi_raw_{self._session_stamp}_{self._raw_flush_index:04d}.npz"
        )
        np.savez_compressed(
            capture_path,
            receive_times=np.array([row[0] for row in self._raw_buffer]),
            device_ids=np.array([row[1] for row in self._raw_buffer]),
            sequence_numbers=np.array([row[2] for row in self._raw_buffer]),
            rssi_values=np.array([row[3] for row in self._raw_buffer]),
            amplitudes=np.stack([row[4] for row in self._raw_buffer]),
        )
        self._raw_buffer = []
        self._raw_flush_index += 1

    def save_labeled_windows(self, label, band_matrices, auxiliary_features):
        """Persist labeled feature windows for Phase 6 real-data training.

        Args:
            label (str): Class label ("empty", "standing", "sitting",
                "lying", "walking").
            band_matrices (list[np.ndarray]): z-scored [W, B] windows.
            auxiliary_features (list[list[float]]): per-window auxiliary
                vectors from signal_processor.build_auxiliary_vector (21).

        Returns:
            Path: The dataset file that was written.
        """
        dataset_directory = self.log_directory / "dataset"
        dataset_directory.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_directory / (
            f"{label}_{self._session_stamp}.npz"
        )
        np.savez_compressed(
            dataset_path,
            label=label,
            band_matrices=np.stack(band_matrices),
            auxiliary_features=np.array(auxiliary_features, dtype=np.float32),
        )
        return dataset_path

    def close(self):
        """Flush and close all open files."""
        self.flush_raw_capture()
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None
            self._events_writer = None


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        logger = DataLogger({"log_dir": temp_dir, "log_events": True})
        logger.log_event("entered", "self-test entry")
        logger.close()
        # The log dir resolves relative to project root, so look there.
        written = list((PROJECT_ROOT_DIRECTORY / temp_dir).glob("events_*.csv"))
        assert written, "events CSV was not written"
        contents = written[0].read_text(encoding="utf-8")
        assert "self-test entry" in contents
    print("data_logger self-test: PASS (event CSV written and readable)")
