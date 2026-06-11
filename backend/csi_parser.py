"""CSI wire-protocol parser and builder (protocol version 1).

Implements the binary UDP datagram format defined in ENGINEERING_SPEC.md §5.2. The
ESP32 receiver firmware, the simulator, and this parser must all agree on
this layout — a shared test vector at the bottom of this file pins it down.

Datagram layout (little-endian):
    offset 0   uint8        protocol_version (always 1)
    offset 1   uint8        device_id
    offset 2   uint32       sequence_number
    offset 6   uint32       esp32_timestamp_us (truncated to 32 bits)
    offset 10  int8         rssi_dbm
    offset 11  uint8        subcarrier_count
    offset 12  int8[2*N]    interleaved (imaginary, real) CSI pairs

Runs on: any OS with Python 3.9+. Dependencies: numpy.
"""

import struct
import time
from dataclasses import dataclass, field

import numpy as np

PROTOCOL_VERSION = 1
HEADER_STRUCT = struct.Struct("<BBIIbB")
HEADER_SIZE_BYTES = HEADER_STRUCT.size  # 12
DEFAULT_SUBCARRIER_COUNT = 64
BYTES_PER_SUBCARRIER = 2  # one int8 imaginary + one int8 real


class CsiParseError(Exception):
    """Raised when a datagram cannot be parsed as a protocol-v1 CSI frame."""


@dataclass
class CsiFrame:
    """One parsed CSI measurement from a single ESP32 receiver.

    Attributes:
        device_id (int): Receiver identity (0-255) from firmware config.
        sequence_number (int): Monotonic counter set by the firmware.
        esp32_timestamp_us (int): Firmware micros() truncated to 32 bits.
        rssi_dbm (int): Received signal strength of the sensed packet.
        amplitudes (np.ndarray): float32[subcarrier_count] per-subcarrier
            amplitude, sqrt(real^2 + imag^2).
        phases (np.ndarray): float32[subcarrier_count] per-subcarrier phase
            in radians, atan2(imag, real).
        receive_time (float): Laptop wall-clock time.time() at parse.
    """

    device_id: int
    sequence_number: int
    esp32_timestamp_us: int
    rssi_dbm: int
    amplitudes: np.ndarray
    phases: np.ndarray
    receive_time: float = field(default_factory=time.time)


def parse_csi_datagram(datagram_bytes):
    """Parse one UDP datagram into a CsiFrame.

    Args:
        datagram_bytes (bytes): Raw datagram payload as received.

    Returns:
        CsiFrame: The decoded frame.

    Raises:
        CsiParseError: On short datagrams, wrong protocol version, or a
            payload length that does not match the declared subcarrier count.
    """
    if len(datagram_bytes) < HEADER_SIZE_BYTES:
        raise CsiParseError(
            f"Datagram too short: {len(datagram_bytes)} bytes "
            f"(need at least {HEADER_SIZE_BYTES})."
        )
    (
        protocol_version,
        device_id,
        sequence_number,
        esp32_timestamp_us,
        rssi_dbm,
        subcarrier_count,
    ) = HEADER_STRUCT.unpack_from(datagram_bytes, 0)

    if protocol_version != PROTOCOL_VERSION:
        raise CsiParseError(
            f"Unsupported protocol version {protocol_version} "
            f"(this parser implements version {PROTOCOL_VERSION})."
        )
    expected_payload_bytes = subcarrier_count * BYTES_PER_SUBCARRIER
    actual_payload_bytes = len(datagram_bytes) - HEADER_SIZE_BYTES
    if actual_payload_bytes != expected_payload_bytes:
        raise CsiParseError(
            f"Payload length mismatch: header declares {subcarrier_count} "
            f"subcarriers ({expected_payload_bytes} bytes) but datagram "
            f"carries {actual_payload_bytes} bytes."
        )

    interleaved_pairs = np.frombuffer(
        datagram_bytes, dtype=np.int8, offset=HEADER_SIZE_BYTES
    ).astype(np.float32)
    imaginary_components = interleaved_pairs[0::2]
    real_components = interleaved_pairs[1::2]
    amplitude_array = np.sqrt(
        real_components**2 + imaginary_components**2
    ).astype(np.float32)
    phase_array = np.arctan2(imaginary_components, real_components).astype(
        np.float32
    )
    return CsiFrame(
        device_id=device_id,
        sequence_number=sequence_number,
        esp32_timestamp_us=esp32_timestamp_us,
        rssi_dbm=rssi_dbm,
        amplitudes=amplitude_array,
        phases=phase_array,
    )


def build_csi_datagram(
    device_id,
    sequence_number,
    esp32_timestamp_us,
    rssi_dbm,
    csi_complex_values,
):
    """Build a protocol-v1 datagram (used by the simulator and tests).

    Args:
        device_id (int): Receiver identity, 0-255.
        sequence_number (int): Monotonic packet counter.
        esp32_timestamp_us (int): Microsecond timestamp (will be truncated
            to 32 bits, matching firmware behaviour).
        rssi_dbm (int): RSSI in dBm, -128..127.
        csi_complex_values (np.ndarray): complex array of per-subcarrier
            CSI; real and imaginary parts are clipped to int8 range.

    Returns:
        bytes: A datagram that parse_csi_datagram() will decode losslessly
            (up to int8 quantisation of the complex values).
    """
    subcarrier_count = len(csi_complex_values)
    header_bytes = HEADER_STRUCT.pack(
        PROTOCOL_VERSION,
        device_id,
        sequence_number & 0xFFFFFFFF,
        esp32_timestamp_us & 0xFFFFFFFF,
        int(np.clip(rssi_dbm, -128, 127)),
        subcarrier_count,
    )
    interleaved_int8 = np.empty(subcarrier_count * 2, dtype=np.int8)
    interleaved_int8[0::2] = np.clip(
        np.round(np.imag(csi_complex_values)), -128, 127
    ).astype(np.int8)
    interleaved_int8[1::2] = np.clip(
        np.round(np.real(csi_complex_values)), -128, 127
    ).astype(np.int8)
    return header_bytes + interleaved_int8.tobytes()


def _self_test():
    """Round-trip a known frame and verify a pinned byte-level test vector."""
    test_complex_csi = np.array([3 + 4j, -5 + 12j, 7 + 0j], dtype=complex)
    test_datagram = build_csi_datagram(
        device_id=7,
        sequence_number=42,
        esp32_timestamp_us=123456,
        rssi_dbm=-55,
        csi_complex_values=test_complex_csi,
    )
    pinned_header_hex = "01072a00000040e20100c903"
    assert test_datagram[:HEADER_SIZE_BYTES].hex() == pinned_header_hex, (
        "Wire format drifted from the pinned protocol-v1 test vector! "
        f"got {test_datagram[:HEADER_SIZE_BYTES].hex()}"
    )
    parsed_frame = parse_csi_datagram(test_datagram)
    assert parsed_frame.device_id == 7
    assert parsed_frame.sequence_number == 42
    assert parsed_frame.rssi_dbm == -55
    expected_amplitudes = np.array([5.0, 13.0, 7.0], dtype=np.float32)
    assert np.allclose(parsed_frame.amplitudes, expected_amplitudes)
    print("csi_parser self-test: PASS (round-trip + pinned test vector)")


if __name__ == "__main__":
    _self_test()
