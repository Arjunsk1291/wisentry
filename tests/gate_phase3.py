"""Phase 3 validation gate (run manually, not via pytest).

Start `python main.py --simulate --duration 45` in another terminal (or
let this script start it itself with --spawn), then this script:
  1. fetches the page and layout (all panel ids must exist),
  2. polls the Dash update endpoint like a browser would,
  3. confirms every panel returns content and that the status panel
     transitions EMPTY → OCCUPIED as the simulated person walks in.

Exits 0 on success.
"""

import json
import subprocess
import sys
import time
import urllib.request

DASHBOARD_URL = "http://127.0.0.1:8050"
PANEL_IDS = [
    "panel-status-bar", "panel-waveform", "panel-pose-figure",
    "panel-room-map", "panel-event-log", "panel-coverage",
    "panel-device-table", "panel-vitals", "panel-spectrogram",
    "panel-heatmap",
]
CALLBACK_PAYLOAD = {
    "output": (
        "..panel-status-bar.children...panel-waveform.figure..."
        "panel-pose-figure.children...panel-room-map.figure..."
        "panel-event-log.children...panel-coverage.children..."
        "panel-device-table.children...panel-vitals.children..."
        "panel-spectrogram.figure...panel-heatmap.figure.."
    ),
    "outputs": [
        {"id": "panel-status-bar", "property": "children"},
        {"id": "panel-waveform", "property": "figure"},
        {"id": "panel-pose-figure", "property": "children"},
        {"id": "panel-room-map", "property": "figure"},
        {"id": "panel-event-log", "property": "children"},
        {"id": "panel-coverage", "property": "children"},
        {"id": "panel-device-table", "property": "children"},
        {"id": "panel-vitals", "property": "children"},
        {"id": "panel-spectrogram", "property": "figure"},
        {"id": "panel-heatmap", "property": "figure"},
    ],
    "inputs": [{"id": "refresh-tick", "property": "n_intervals", "value": 1}],
    "changedPropIds": ["refresh-tick.n_intervals"],
}


def fetch(path):
    """GET a dashboard path, returning the body as text."""
    with urllib.request.urlopen(DASHBOARD_URL + path, timeout=5) as response:
        return response.read().decode("utf-8")


def poll_panels():
    """POST the interval callback exactly as the browser does.

    Returns:
        dict: panel-id -> rendered JSON content.
    """
    request = urllib.request.Request(
        DASHBOARD_URL + "/_dash-update-component",
        data=json.dumps(CALLBACK_PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())["response"]


def wait_for_dashboard(timeout_seconds=30):
    """Block until the dashboard answers HTTP, or fail."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            fetch("/")
            return
        except OSError:
            time.sleep(1)
    print("GATE FAIL: dashboard never came up")
    sys.exit(1)


def main():
    """Run the gate sequence; exit 0 on pass."""
    spawned_process = None
    if "--spawn" in sys.argv:
        spawned_process = subprocess.Popen(
            [sys.executable, "main.py", "--simulate", "--duration", "45"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    try:
        wait_for_dashboard()
        layout_json = fetch("/_dash-layout")
        for panel_id in PANEL_IDS:
            assert panel_id in layout_json, f"layout missing {panel_id}"
        assert "SIMULATION MODE" in layout_json
        print("gate: layout contains all panels + simulation banner")

        first_poll = poll_panels()
        assert set(first_poll) == set(PANEL_IDS), "callback panel set wrong"
        first_status = json.dumps(first_poll["panel-status-bar"])
        print(f"gate: first poll OK (status contains "
              f"{'OCCUPIED' if 'OCCUPIED' in first_status else 'EMPTY'})")

        saw_occupied, saw_empty = False, False
        for _ in range(30):  # ~30 s: covers empty + occupied scenario phases
            status_text = json.dumps(poll_panels()["panel-status-bar"])
            saw_occupied |= "OCCUPIED" in status_text
            saw_empty |= "EMPTY" in status_text
            if saw_occupied and saw_empty:
                break
            time.sleep(1)
        assert saw_occupied and saw_empty, (
            f"panels not updating: occupied={saw_occupied} empty={saw_empty}"
        )
        print("gate: status panel transitioned EMPTY <-> OCCUPIED — "
              "panels are live")
        print("GATE PASS")
        return 0
    finally:
        if spawned_process is not None:
            spawned_process.wait(timeout=90)


if __name__ == "__main__":
    sys.exit(main())
