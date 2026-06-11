"""WiSentry web dashboard — 7 live panels served by Dash.

Panels (ENGINEERING_SPEC.md §8): status bar, CSI waveform, pose stick figure,
room map, event log, coverage advisor, device table. Reads consistent
snapshots from backend.detector.SystemState on a 200 ms interval; never
touches the pipeline threads directly.

Runs on: any OS with Python 3.9+. Open http://localhost:8050 in any
browser. Dependencies: dash, plotly, numpy.
"""

import base64
import time
from datetime import datetime

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from backend.skeleton import POSE_TEMPLATES, SKELETON_BONES

BACKGROUND_COLOR = "#111418"
PANEL_COLOR = "#1b2026"
TEXT_COLOR = "#e8e8e8"
ACCENT_CYAN = "#22d3ee"
OCCUPIED_GREEN = "#16a34a"
EMPTY_RED = "#7f1d1d"
WARNING_YELLOW = "#eab308"
EVENT_COLORS = {"entered": "#4ade80", "left": "#f87171",
                "pose_change": "#facc15"}
SVG_WIDTH = 200
SVG_HEIGHT = 240
WAVEFORM_LINE_COLORS = ["#22d3ee", "#a78bfa", "#fb923c", "#4ade80",
                        "#f472b6", "#facc15"]
PANEL_STYLE = {
    "backgroundColor": PANEL_COLOR,
    "borderRadius": "8px",
    "padding": "12px",
    "margin": "6px",
    "flex": "1",
    "minWidth": "300px",
}
COVERAGE_ADVICE = {
    1: ("presence only (~15 m²)", "Add 1 more ESP32 to unlock pose detection."),
    2: ("pose active (~25 m²)", "Add 1 more ESP32 to unlock skeleton mode."),
    3: ("full system active (~35 m²)", "Coverage scales with your room layout."),
}


def _panel_title(title_text):
    """Uniform panel heading element."""
    return html.H3(title_text, style={"marginTop": "0", "fontSize": "14px",
                                      "color": ACCENT_CYAN,
                                      "letterSpacing": "1px"})


def build_layout(simulation_mode, update_interval_ms):
    """Construct the static page skeleton; panels fill in via callback.

    Args:
        simulation_mode (bool): Shows the red simulation banner when True.
        update_interval_ms (int): Refresh period for dcc.Interval.

    Returns:
        dash.html.Div: The root layout element.
    """
    banner = None
    if simulation_mode:
        banner = html.Div(
            "⚠ SIMULATION MODE — no hardware connected",
            style={"backgroundColor": "#7f1d1d", "color": "white",
                   "textAlign": "center", "padding": "6px",
                   "fontWeight": "bold"},
        )
    return html.Div(
        style={"backgroundColor": BACKGROUND_COLOR, "color": TEXT_COLOR,
               "fontFamily": "Segoe UI, sans-serif", "minHeight": "100vh",
               "padding": "8px"},
        children=[
            dcc.Interval(id="refresh-tick", interval=update_interval_ms),
            banner,
            html.H2("WiSentry — WiFi CSI Presence & Pose",
                    style={"margin": "8px 6px"}),
            html.Div(id="panel-status-bar"),
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("CSI WAVEFORM"),
                    dcc.Graph(id="panel-waveform",
                              config={"displayModeBar": False}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("POSE FIGURE (skeleton: experimental)"),
                    html.Div(id="panel-pose-figure",
                             style={"textAlign": "center"}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("ROOM MAP"),
                    dcc.Graph(id="panel-room-map",
                              config={"displayModeBar": False}),
                ]),
            ]),
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("EVENT LOG"),
                    html.Div(id="panel-event-log",
                             style={"maxHeight": "240px", "overflowY": "auto",
                                    "fontSize": "13px"}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("COVERAGE ADVISOR"),
                    html.Div(id="panel-coverage"),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("DEVICE TABLE"),
                    html.Div(id="panel-device-table"),
                ]),
            ]),
        ],
    )


def render_status_bar(snapshot):
    """PANEL 1 — full-width occupancy card, pose + confidence, device count."""
    occupied = snapshot["presence"]
    status_text = "OCCUPIED" if occupied else "EMPTY"
    status_color = OCCUPIED_GREEN if occupied else EMPTY_RED
    pose_text = snapshot["pose_label"] or "—"
    confidence = snapshot["pose_confidence"] if occupied else 0.0
    online_count = _online_device_count(snapshot)
    confidence_bar = html.Div(style={
        "backgroundColor": "#374151", "borderRadius": "4px",
        "height": "10px", "width": "160px", "display": "inline-block",
        "marginLeft": "8px",
    }, children=html.Div(style={
        "backgroundColor": WARNING_YELLOW, "height": "10px",
        "borderRadius": "4px", "width": f"{int(confidence * 100)}%",
    }))
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "24px",
               "backgroundColor": status_color, "borderRadius": "8px",
               "padding": "14px", "margin": "6px"},
        children=[
            html.Span(status_text,
                      style={"fontSize": "26px", "fontWeight": "bold"}),
            html.Span([f"Pose: {pose_text} "
                       f"({confidence:.0%} confidence)", confidence_bar]),
            html.Span(f"ESP32 online: {online_count}",
                      style={"marginLeft": "auto", "fontSize": "16px"}),
        ],
    )


def render_waveform(snapshot, history_seconds):
    """PANEL 2 — scrolling mean-amplitude line per device, last 5 s."""
    figure = go.Figure()
    now = time.time()
    for plot_index, (device_id, points) in enumerate(
            sorted(snapshot["waveforms"].items())):
        recent = [(t - now, amp) for t, amp in points
                  if now - t <= history_seconds]
        figure.add_trace(go.Scatter(
            x=[t for t, _ in recent], y=[amp for _, amp in recent],
            mode="lines", name=f"RX-{device_id}",
            line={"color": WAVEFORM_LINE_COLORS[
                plot_index % len(WAVEFORM_LINE_COLORS)], "width": 1.5},
        ))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL_COLOR,
        plot_bgcolor=PANEL_COLOR, height=240,
        margin={"l": 40, "r": 10, "t": 10, "b": 30},
        xaxis_title="seconds ago", yaxis_title="mean amplitude",
        xaxis_range=[-history_seconds, 0], showlegend=True,
        legend={"orientation": "h", "y": 1.15},
    )
    return figure


def _keypoints_to_svg(keypoints, pose_label):
    """Render keypoints as an SVG stick figure (data-URI for html.Img)."""
    points = [(float(x) * SVG_WIDTH, float(y) * SVG_HEIGHT)
              for x, y in keypoints]
    bone_lines = "".join(
        f'<line x1="{points[a][0]:.1f}" y1="{points[a][1]:.1f}" '
        f'x2="{points[b][0]:.1f}" y2="{points[b][1]:.1f}" '
        f'stroke="{ACCENT_CYAN}" stroke-width="3"/>'
        for a, b in SKELETON_BONES
    )
    joint_dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{WARNING_YELLOW}"/>'
        for x, y in points
    )
    head_x, head_y = points[0]
    svg_text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" '
        f'height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
        f'<rect width="100%" height="100%" fill="{PANEL_COLOR}"/>'
        f'<circle cx="{head_x:.1f}" cy="{head_y:.1f}" r="10" '
        f'fill="none" stroke="{ACCENT_CYAN}" stroke-width="3"/>'
        f"{bone_lines}{joint_dots}"
        f'<text x="8" y="{SVG_HEIGHT - 8}" fill="{TEXT_COLOR}" '
        f'font-size="13">{pose_label}</text></svg>'
    )
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render_pose_figure(snapshot):
    """PANEL 3 — stick figure for the current pose, keypoint dots overlaid."""
    pose_label = snapshot["pose_label"]
    if not snapshot["presence"] or pose_label is None:
        return html.Div("no person detected",
                        style={"padding": "60px 0", "color": "#6b7280"})
    keypoints = snapshot["keypoints"]
    if keypoints is None:
        keypoints = POSE_TEMPLATES[pose_label]
    return html.Img(src=_keypoints_to_svg(keypoints, pose_label))


def render_room_map(snapshot, room_config):
    """PANEL 4 — top-down room with device dots and estimated person dot."""
    width = float(room_config["width_meters"])
    depth = float(room_config["depth_meters"])
    figure = go.Figure()
    positions = room_config.get("device_positions", {})
    figure.add_trace(go.Scatter(
        x=[p["x"] for p in positions.values()],
        y=[p["y"] for p in positions.values()],
        mode="markers+text", name="ESP32",
        text=[p["label"] for p in positions.values()],
        textposition="top center", textfont={"color": TEXT_COLOR},
        marker={"color": ACCENT_CYAN, "size": 14, "symbol": "square"},
    ))
    person = snapshot["position_estimate"]
    if person is not None:
        figure.add_trace(go.Scatter(
            x=[person[0]], y=[person[1]], mode="markers", name="person (est.)",
            marker={"color": "#fb923c", "size": 20},
        ))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL_COLOR,
        plot_bgcolor="#10141a", height=240,
        margin={"l": 30, "r": 10, "t": 10, "b": 30},
        xaxis={"range": [-0.3, width + 0.3], "title": "meters"},
        yaxis={"range": [depth + 0.3, -0.3], "scaleanchor": "x"},
        showlegend=False,
    )
    return figure


def render_event_log(snapshot):
    """PANEL 5 — last 30 events, newest first, color-coded."""
    rows = []
    for event_time, kind, message in reversed(snapshot["events"]):
        stamp = datetime.fromtimestamp(event_time).strftime("%H:%M:%S")
        rows.append(html.Div(
            f"{stamp}  {message}",
            style={"color": EVENT_COLORS.get(kind, TEXT_COLOR),
                   "padding": "2px 0"},
        ))
    return rows or html.Div("no events yet", style={"color": "#6b7280"})


def _online_device_count(snapshot, timeout_seconds=2.0):
    """Devices that sent a packet within the timeout."""
    now = time.time()
    return sum(1 for stats in snapshot["device_stats"].values()
               if now - stats["last_seen"] <= timeout_seconds)


def render_coverage(snapshot):
    """PANEL 6 — scaling advice keyed on live online-device count."""
    online_count = _online_device_count(snapshot)
    if online_count == 0:
        headline, advice = ("no receivers online",
                            "Check power, WiFi credentials, and firewall.")
    else:
        headline, advice = COVERAGE_ADVICE[min(online_count, 3)]
    return html.Div([
        html.Div(f"{online_count} receiver(s) online → {headline}",
                 style={"fontSize": "16px", "fontWeight": "bold",
                        "marginBottom": "8px"}),
        html.Div(advice, style={"color": "#9ca3af"}),
        html.Div(f"frames processed: {snapshot['frames_processed']:,}",
                 style={"marginTop": "12px", "fontSize": "12px",
                        "color": "#6b7280"}),
    ])


def render_device_table(snapshot, timeout_seconds=2.0):
    """PANEL 7 — Device ID | IP | RSSI | Packets/s | Status."""
    now = time.time()
    header = html.Tr([html.Th(col, style={"textAlign": "left",
                                          "padding": "4px 10px"})
                      for col in ["Device", "IP", "RSSI", "Packets", "Status"]])
    body_rows = []
    for device_id, stats in sorted(snapshot["device_stats"].items()):
        online = now - stats["last_seen"] <= timeout_seconds
        status_cell = html.Td("● ONLINE" if online else "● OFFLINE",
                              style={"color": "#4ade80" if online
                                     else "#f87171", "padding": "4px 10px"})
        body_rows.append(html.Tr([
            html.Td(f"RX-{device_id}", style={"padding": "4px 10px"}),
            html.Td(stats["ip"], style={"padding": "4px 10px"}),
            html.Td(f"{stats['rssi']} dBm", style={"padding": "4px 10px"}),
            html.Td(f"{stats['packets']:,}", style={"padding": "4px 10px"}),
            status_cell,
        ]))
    if not body_rows:
        return html.Div("no devices seen yet", style={"color": "#6b7280"})
    return html.Table([header] + body_rows, style={"width": "100%",
                                                   "fontSize": "13px"})


def run_dashboard(config, system_state, duration=None):
    """Create and serve the dashboard (blocking unless duration given).

    Args:
        config (dict): Full parsed config.yaml.
        system_state (SystemState): Live state written by the pipeline.
        duration (float | None): When set, serve in a background thread
            for this many seconds, then return (used by validation gates).
    """
    dashboard_config = config["dashboard"]
    app = Dash(__name__, title="WiSentry")
    app.layout = build_layout(
        system_state.simulation_mode,
        int(dashboard_config["update_interval_ms"]),
    )

    @app.callback(
        Output("panel-status-bar", "children"),
        Output("panel-waveform", "figure"),
        Output("panel-pose-figure", "children"),
        Output("panel-room-map", "figure"),
        Output("panel-event-log", "children"),
        Output("panel-coverage", "children"),
        Output("panel-device-table", "children"),
        Input("refresh-tick", "n_intervals"),
    )
    def refresh_all_panels(_tick_count):
        snapshot = system_state.snapshot()
        return (
            render_status_bar(snapshot),
            render_waveform(snapshot,
                            dashboard_config["waveform_history_seconds"]),
            render_pose_figure(snapshot),
            render_room_map(snapshot, config["room"]),
            render_event_log(snapshot),
            render_coverage(snapshot),
            render_device_table(snapshot),
        )

    host = dashboard_config["host"]
    port = int(dashboard_config["port"])
    print(f"dashboard: serving at http://{host}:{port}")
    if duration is None:
        app.run(host=host, port=port, debug=False)
        return
    import threading

    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False),
        daemon=True,
    )
    server_thread.start()
    time.sleep(duration)


if __name__ == "__main__":
    from backend.detector import SystemState

    test_state = SystemState()
    test_state.simulation_mode = True
    layout = build_layout(True, 200)
    snapshot = test_state.snapshot()
    render_status_bar(snapshot)
    render_event_log(snapshot)
    render_coverage(snapshot)
    render_device_table(snapshot)
    render_pose_figure(snapshot)
    print("dashboard self-test: PASS (layout + all renderers on empty state)")
