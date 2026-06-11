"""Unit tests for the protocol-v1 CSI parser/builder."""

import numpy as np
import pytest

from backend.csi_parser import (
    HEADER_SIZE_BYTES,
    CsiParseError,
    build_csi_datagram,
    parse_csi_datagram,
)


def _example_datagram(subcarrier_count=64):
    complex_values = np.full(subcarrier_count, 10 + 5j, dtype=complex)
    return build_csi_datagram(
        device_id=3,
        sequence_number=99,
        esp32_timestamp_us=5_000_000,
        rssi_dbm=-60,
        csi_complex_values=complex_values,
    )


def test_round_trip_preserves_header_fields():
    frame = parse_csi_datagram(_example_datagram())
    assert frame.device_id == 3
    assert frame.sequence_number == 99
    assert frame.esp32_timestamp_us == 5_000_000
    assert frame.rssi_dbm == -60
    assert len(frame.amplitudes) == 64


def test_amplitude_and_phase_math():
    datagram = build_csi_datagram(
        device_id=1, sequence_number=1, esp32_timestamp_us=0, rssi_dbm=-50,
        csi_complex_values=np.array([3 + 4j], dtype=complex),
    )
    frame = parse_csi_datagram(datagram)
    assert frame.amplitudes[0] == pytest.approx(5.0)
    assert frame.phases[0] == pytest.approx(np.arctan2(4, 3))


def test_pinned_wire_format_test_vector():
    """The shared firmware/simulator/parser byte layout must never drift."""
    datagram = build_csi_datagram(
        device_id=7, sequence_number=42, esp32_timestamp_us=123456,
        rssi_dbm=-55,
        csi_complex_values=np.array([3 + 4j, -5 + 12j, 7 + 0j], dtype=complex),
    )
    assert datagram[:HEADER_SIZE_BYTES].hex() == "01072a00000040e20100c903"
    assert datagram[HEADER_SIZE_BYTES:].hex() == "04030cfb0007"


def test_short_datagram_rejected():
    with pytest.raises(CsiParseError):
        parse_csi_datagram(b"\x01\x02\x03")


def test_wrong_protocol_version_rejected():
    corrupted = b"\x02" + _example_datagram()[1:]
    with pytest.raises(CsiParseError):
        parse_csi_datagram(corrupted)


def test_payload_length_mismatch_rejected():
    truncated = _example_datagram()[:-4]
    with pytest.raises(CsiParseError):
        parse_csi_datagram(truncated)
