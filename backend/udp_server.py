"""UDP server that receives CSI datagrams from ESP32 receivers.

Runs as a daemon thread: binds the configured port, parses each datagram
with backend.csi_parser, and pushes (CsiFrame, sender_ip) tuples onto a
bounded queue for the processing pipeline. Malformed datagrams are
counted, never fatal.

Runs on: any OS with Python 3.9+. Dependencies: backend.csi_parser.
"""

import queue
import socket
import threading

from backend.csi_parser import CsiParseError, parse_csi_datagram

RECEIVE_BUFFER_BYTES = 2048      # max expected datagram is 12 + 2*256 bytes
SOCKET_TIMEOUT_SECONDS = 0.5     # lets the thread notice stop requests
FRAME_QUEUE_MAX_SIZE = 1024      # backpressure cap (~3 s at 6 RX x 50 Hz); drops oldest
PARSE_ERROR_REPORT_INTERVAL = 100  # log every Nth malformed datagram


class UdpCsiServer(threading.Thread):
    """Background thread feeding parsed CSI frames into a queue."""

    def __init__(self, network_config, on_heard=None):
        """Create (but do not yet start) the server thread.

        Args:
            network_config (dict): `network` section of config.yaml
                (keys: udp_listen_host, udp_listen_port).
            on_heard (callable | None): Called as on_heard(device_id, ip,
                receive_time) for every valid datagram, on the receive
                thread, so device liveness does not depend on how far
                behind the processing pipeline is.
        """
        self._on_heard = on_heard
        super().__init__(name="UdpCsiServer", daemon=True)
        self.listen_host = network_config.get("udp_listen_host", "0.0.0.0")
        self.listen_port = int(network_config.get("udp_listen_port", 5566))
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)
        self.parse_error_count = 0
        self.received_datagram_count = 0
        self.dropped_frame_count = 0
        self._stop_event = threading.Event()
        self._socket = None

    def run(self):
        """Bind the socket and pump datagrams until stop() is called."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(SOCKET_TIMEOUT_SECONDS)
        try:
            self._socket.bind((self.listen_host, self.listen_port))
        except OSError as bind_error:
            print(
                f"udp_server: cannot bind {self.listen_host}:{self.listen_port}"
                f" — {bind_error}. Is another WiSentry instance running?"
            )
            return
        while not self._stop_event.is_set():
            self._receive_one_datagram()
        self._socket.close()

    def _receive_one_datagram(self):
        """Receive, parse, and enqueue a single datagram (or time out)."""
        try:
            datagram_bytes, sender_address = self._socket.recvfrom(
                RECEIVE_BUFFER_BYTES
            )
        except socket.timeout:
            return
        except OSError:
            return  # socket closed during shutdown
        self.received_datagram_count += 1
        try:
            parsed_frame = parse_csi_datagram(datagram_bytes)
        except CsiParseError as parse_error:
            self.parse_error_count += 1
            if self.parse_error_count % PARSE_ERROR_REPORT_INTERVAL == 1:
                print(f"udp_server: malformed datagram from "
                      f"{sender_address[0]}: {parse_error}")
            return
        if self._on_heard is not None:
            self._on_heard(parsed_frame.device_id, sender_address[0],
                           parsed_frame.receive_time)
        try:
            self.frame_queue.put_nowait((parsed_frame, sender_address[0]))
        except queue.Full:
            self.dropped_frame_count += 1
            try:  # drop the oldest frame to keep latency bounded
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((parsed_frame, sender_address[0]))
            except (queue.Empty, queue.Full):
                pass

    def stop(self):
        """Request shutdown; the thread exits within one socket timeout."""
        self._stop_event.set()


if __name__ == "__main__":
    import time

    import numpy as np

    from backend.csi_parser import build_csi_datagram

    server = UdpCsiServer({"udp_listen_host": "127.0.0.1",
                           "udp_listen_port": 55660})
    server.start()
    time.sleep(0.2)
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    test_payload = build_csi_datagram(
        device_id=1, sequence_number=1, esp32_timestamp_us=1000,
        rssi_dbm=-40, csi_complex_values=np.ones(64, dtype=complex) * (10 + 5j),
    )
    sender_socket.sendto(test_payload, ("127.0.0.1", 55660))
    sender_socket.sendto(b"garbage", ("127.0.0.1", 55660))
    time.sleep(0.3)
    received_frame, sender_ip = server.frame_queue.get(timeout=1.0)
    assert received_frame.device_id == 1 and sender_ip == "127.0.0.1"
    assert server.parse_error_count == 1
    server.stop()
    server.join(timeout=2.0)
    print("udp_server self-test: PASS (receive, parse, malformed counted)")
