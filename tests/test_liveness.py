"""Device liveness must not depend on processing backlog."""
import time

from backend.detector import SystemState
from dashboard.app import _online_device_count, render_device_table


def test_heard_devices_count_online_even_when_pipeline_lags():
    state = SystemState(event_log_max_entries=10)
    now = time.time()
    stale = now - 10.0  # pipeline is 10 s behind
    state._device_stats[1] = {"ip": "10.0.0.2", "rssi": -50,
                              "packets": 5, "last_seen": stale}
    state.note_heard(1, "10.0.0.2", now)
    snap = state.snapshot()
    snap["render_time"] = now
    assert _online_device_count(snap) == 1
    assert "ONLINE" in str(render_device_table(snap))
    assert "OFFLINE" not in str(render_device_table(snap))


def test_silent_device_goes_offline():
    state = SystemState(event_log_max_entries=10)
    now = time.time()
    state._device_stats[2] = {"ip": "10.0.0.3", "rssi": -50,
                              "packets": 5, "last_seen": now - 5.0}
    snap = state.snapshot()
    snap["render_time"] = now
    assert _online_device_count(snap) == 0
    assert "OFFLINE" in str(render_device_table(snap))
