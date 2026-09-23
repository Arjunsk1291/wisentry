"""WiSentry web dashboard — 7 live panels served by Dash.

Panels (ENGINEERING_SPEC.md §8): status bar, CSI waveform, pose stick figure,
room map, event log, coverage advisor, device table. Reads consistent
snapshots from backend.detector.SystemState on a 200 ms interval; never
touches the pipeline threads directly.

Runs on: any OS with Python 3.9+. Open http://localhost:8050 in any
browser. Dependencies: dash, plotly, numpy.
"""

import base64
import math
import time
from datetime import datetime

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from backend.skeleton import POSE_TEMPLATES, SKELETON_BONES

BACKGROUND_COLOR = "#05070d"
PANEL_COLOR = "rgba(13, 20, 33, 0.82)"
PLOT_COLOR = "#0a101b"
TEXT_COLOR = "#dbe7f5"
MUTED_TEXT = "#6b7a90"
ACCENT_CYAN = "#22d3ee"
ACCENT_VIOLET = "#a78bfa"
OCCUPIED_GREEN = "#10b981"
EMPTY_RED = "#ef4444"
FONT_STACK = "'JetBrains Mono', 'Fira Code', 'DejaVu Sans Mono', monospace"
GRID_COLOR = "rgba(34, 211, 238, 0.08)"
WARNING_YELLOW = "#eab308"
EVENT_COLORS = {"entered": "#4ade80", "left": "#f87171",
                "pose_change": "#facc15"}
SVG_WIDTH = 200
SVG_HEIGHT = 240
WAVEFORM_LINE_COLORS = ["#22d3ee", "#a78bfa", "#fb923c", "#4ade80",
                        "#f472b6", "#facc15"]
PANEL_STYLE = {
    "background": "linear-gradient(160deg, rgba(20, 30, 48, 0.9), "
                  "rgba(8, 12, 22, 0.9))",
    "border": "1px solid rgba(34, 211, 238, 0.22)",
    "boxShadow": "0 0 18px rgba(34, 211, 238, 0.08), "
                 "inset 0 0 24px rgba(34, 211, 238, 0.03)",
    "borderRadius": "12px",
    "padding": "14px",
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
    return html.H3([html.Span("▍", style={"color": ACCENT_VIOLET}),
                    title_text],
                   style={"marginTop": "0", "fontSize": "12px",
                          "color": ACCENT_CYAN, "letterSpacing": "3px",
                          "fontWeight": "600"})


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
            style={"background": "linear-gradient(90deg, rgba(239,68,68,0.0),"
                                 " rgba(239,68,68,0.35), rgba(239,68,68,0.0))",
                   "color": "#fecaca", "textAlign": "center",
                   "padding": "6px", "letterSpacing": "2px",
                   "fontSize": "12px", "fontWeight": "bold"},
        )
    return html.Div(
        style={"backgroundColor": BACKGROUND_COLOR,
               "backgroundImage":
                   "radial-gradient(circle at 15% 0%, rgba(34,211,238,0.10), "
                   "transparent 40%), radial-gradient(circle at 90% 10%, "
                   "rgba(167,139,250,0.10), transparent 45%), "
                   "linear-gradient(rgba(34,211,238,0.035) 1px, "
                   "transparent 1px), linear-gradient(90deg, "
                   "rgba(34,211,238,0.035) 1px, transparent 1px)",
               "backgroundSize": "auto, auto, 32px 32px, 32px 32px",
               "color": TEXT_COLOR, "fontFamily": FONT_STACK,
               "minHeight": "100vh", "padding": "10px"},
        children=[
            dcc.Interval(id="refresh-tick", interval=update_interval_ms),
            banner,
            html.Div(style={"display": "flex", "alignItems": "baseline",
                            "gap": "14px", "margin": "8px 6px 4px"},
                     children=[
                html.H2("WISENTRY", style={
                    "margin": "0", "letterSpacing": "6px",
                    "color": ACCENT_CYAN,
                    "textShadow": "0 0 12px rgba(34,211,238,0.6)"}),
                html.Span("WiFi CSI presence & pose sensing",
                          style={"color": MUTED_TEXT, "fontSize": "13px",
                                 "letterSpacing": "1px"}),
            ]),
            html.Div(id="panel-status-bar"),
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("CSI WAVEFORM"),
                    dcc.Graph(id="panel-waveform",
                              config={"displayModeBar": False}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("POSE · 3D SKELETON (experimental)"),
                    html.Div(id="panel-pose-figure",
                             style={"textAlign": "center"}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("ROOM · 3D LINK MAP"),
                    dcc.Graph(id="panel-room-map",
                              config={"displayModeBar": False}),
                ]),
            ]),
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("VITALS · RESPIRATION"),
                    html.Div(id="panel-vitals"),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("MOTION SPECTROGRAM (PCA + STFT)"),
                    dcc.Graph(id="panel-spectrogram",
                              config={"displayModeBar": False}),
                ]),
                html.Div(style=PANEL_STYLE, children=[
                    _panel_title("SUBCARRIER ATTENUATION"),
                    dcc.Graph(id="panel-heatmap",
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
        "backgroundColor": "rgba(255,255,255,0.08)", "borderRadius": "6px",
        "height": "8px", "width": "180px", "display": "inline-block",
        "marginLeft": "10px", "verticalAlign": "middle",
    }, children=html.Div(style={
        "background": f"linear-gradient(90deg, {ACCENT_CYAN}, "
                      f"{ACCENT_VIOLET})",
        "boxShadow": "0 0 10px rgba(34,211,238,0.7)", "height": "8px",
        "borderRadius": "6px", "width": f"{int(confidence * 100)}%",
    }))
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "28px",
               "background": "linear-gradient(90deg, rgba(10,16,28,0.95), "
                             "rgba(14,22,38,0.85))",
               "border": f"1px solid {status_color}",
               "boxShadow": f"0 0 22px {status_color}55",
               "borderRadius": "12px", "padding": "14px 18px",
               "margin": "6px"},
        children=[
            html.Span([html.Span("●", style={
                           "color": status_color, "marginRight": "10px",
                           "textShadow": f"0 0 10px {status_color}"}),
                       status_text],
                      style={"fontSize": "26px", "fontWeight": "bold",
                             "letterSpacing": "4px"}),
            html.Span([f"Pose: {pose_text} "
                       f"({confidence:.0%} confidence)", confidence_bar]),
            html.Span(f"ESP32 ONLINE · {online_count}",
                      style={"marginLeft": "auto", "fontSize": "14px",
                             "letterSpacing": "2px", "color": ACCENT_CYAN}),
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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_COLOR, font={"family": FONT_STACK, "size": 11}, height=240,
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


def _grid_lines(x0, x1, y0, y1, step):
    """One polyline (None-separated) for a floor grid - a single trace."""
    xs, ys = [], []
    count_x = int(round((x1 - x0) / step))
    count_y = int(round((y1 - y0) / step))
    for i in range(count_x + 1):
        g = x0 + i * step
        xs += [g, g, None]
        ys += [y0, y1, None]
    for i in range(count_y + 1):
        g = y0 + i * step
        xs += [x0, x1, None]
        ys += [g, g, None]
    return xs, ys


# COCO-17 index groups used by the 3D lift.
_LEFT_ARM = (7, 9)
_RIGHT_ARM = (8, 10)
_LEFT_LEG = (13, 15)
_RIGHT_LEG = (14, 16)


def lift_keypoints_3d(keypoints, pose_label):
    """Lift normalised 2D keypoints into a plausible 3D body for display.

    The model outputs 2D image-plane keypoints only; depth here is a
    pose-conditioned display heuristic (not an estimate), used so the
    figure reads as a body in a room rather than a flat drawing.

    Args:
        keypoints (sequence): 17 (x, y) pairs in [0, 1], y pointing down.
        pose_label (str): standing | sitting | lying | walking.

    Returns:
        tuple[list, list, list]: x (lateral), y (depth), z (height) in
        metres-like units for a ~1.75 m figure.
    """
    height_scale = 1.75
    lateral, depth, up = [], [], []
    for index, (x_norm, y_norm) in enumerate(keypoints):
        x_norm, y_norm = float(x_norm), float(y_norm)
        if pose_label == "lying":
            # Template is a top-down view: x runs along the body and y
            # spreads left/right, so map y to depth and lay it on a bed.
            lateral.append((x_norm - 0.5) * height_scale)
            depth.append((y_norm - 0.75) * height_scale)
            up.append(0.45 + (0.06 if index <= 4 else 0.0))
            continue
        lx = (x_norm - 0.5) * height_scale
        lz = (1.0 - y_norm) * height_scale
        ly = 0.0
        if pose_label == "sitting" and index in (13, 14, 15, 16):
            hip_x = float(keypoints[11 if index % 2 else 12][0])
            lx = (hip_x - 0.5) * height_scale
            ly = 0.45 if index in (13, 14) else 0.5
        if pose_label == "walking":
            if index in _LEFT_LEG or index in _RIGHT_ARM:
                ly = 0.18 if index in (13, 15) else -0.14
            elif index in _RIGHT_LEG or index in _LEFT_ARM:
                ly = -0.18 if index in (14, 16) else 0.14
        if index <= 4:
            ly += 0.03
        lateral.append(lx)
        depth.append(ly)
        up.append(lz)
    return lateral, depth, up


def render_pose_3d(keypoints, pose_label, confidence=None):
    """Build the 3D skeleton figure: glowing bones, joints, head, floor."""
    xs, ys, zs = lift_keypoints_3d(keypoints, pose_label)
    figure = go.Figure()
    # Floor grid and a soft shadow ring under the body.
    gx, gy = _grid_lines(-1.0, 1.0, -1.0, 1.0, 0.2)
    figure.add_trace(go.Scatter3d(
        x=gx, y=gy, z=[0 if v is not None else None for v in gx],
        mode="lines", line={"color": "#0f3b47", "width": 1},
        hoverinfo="skip", showlegend=False))
    center_x = (xs[11] + xs[12]) / 2
    center_y = (ys[11] + ys[12]) / 2
    ring = [i * 2 * math.pi / 40 for i in range(41)]
    figure.add_trace(go.Scatter3d(
        x=[center_x + 0.35 * math.cos(a) for a in ring],
        y=[center_y + 0.35 * math.sin(a) for a in ring],
        z=[0.0] * len(ring), mode="lines",
        line={"color": "#6d5bd0", "width": 4},
        hoverinfo="skip", showlegend=False))
    # Bones: wide translucent glow pass, then a bright core pass.
    for width, color in ((18, "#0e4f5c"),
                         (7, ACCENT_CYAN)):
        bx, by, bz = [], [], []
        for a, b in SKELETON_BONES:
            bx += [xs[a], xs[b], None]
            by += [ys[a], ys[b], None]
            bz += [zs[a], zs[b], None]
        figure.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode="lines",
            line={"color": color, "width": width},
            hoverinfo="skip", showlegend=False))
    body_joints = list(range(5, 17))
    figure.add_trace(go.Scatter3d(
        x=[xs[i] for i in body_joints], y=[ys[i] for i in body_joints],
        z=[zs[i] for i in body_joints], mode="markers",
        marker={"size": 5, "color": ACCENT_VIOLET,
                "line": {"color": "#ffffff", "width": 1}},
        hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter3d(
        x=[xs[0]], y=[ys[0]], z=[zs[0] + 0.05], mode="markers",
        marker={"size": 16, "color": "#0b6b7a",
                "line": {"color": ACCENT_CYAN, "width": 3}},
        hoverinfo="skip", showlegend=False))
    axis = {"visible": False, "showbackground": False}
    title = pose_label.upper()
    if confidence is not None:
        title += f"  ·  {confidence:.0%}"
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        height=260, margin={"l": 0, "r": 0, "t": 24, "b": 0},
        font={"family": FONT_STACK, "size": 11},
        title={"text": title, "x": 0.5, "y": 0.97,
               "font": {"color": TEXT_COLOR, "size": 12}},
        uirevision="pose",
        scene={"xaxis": {**axis, "range": [-1.1, 1.1]},
               "yaxis": {**axis, "range": [-1.1, 1.1]},
               "zaxis": {**axis, "range": [0, 1.95]},
               "aspectmode": "manual",
               "aspectratio": {"x": 1, "y": 1, "z": 0.78},
               "camera": {"eye": {"x": 0.55, "y": -1.05, "z": 0.35},
                          "center": {"x": 0, "y": 0, "z": -0.05}}},
    )
    return figure


def render_pose_figure(snapshot):
    """PANEL 3 — 3D skeleton for the current pose (keypoints lifted)."""
    pose_label = snapshot["pose_label"]
    if not snapshot["presence"] or pose_label is None:
        return html.Div("no person detected",
                        style={"padding": "100px 0", "color": MUTED_TEXT,
                               "letterSpacing": "2px"})
    keypoints = snapshot["keypoints"]
    if keypoints is None:
        keypoints = POSE_TEMPLATES[pose_label]
    return dcc.Graph(
        figure=render_pose_3d(keypoints, pose_label,
                              snapshot.get("pose_confidence")),
        config={"displayModeBar": False},
        style={"height": "260px"})


ROOM_HEIGHT_METERS = 2.6
DEVICE_HEIGHT_METERS = 1.0


def _room_wireframe(width, depth, height):
    """Line segments for the room box (floor, ceiling edges, corners)."""
    corners = [(0, 0), (width, 0), (width, depth), (0, depth), (0, 0)]
    xs, ys, zs = [], [], []
    for z in (0.0, height):
        xs += [c[0] for c in corners] + [None]
        ys += [c[1] for c in corners] + [None]
        zs += [z] * len(corners) + [None]
    for cx, cy in corners[:4]:
        xs += [cx, cx, None]
        ys += [cy, cy, None]
        zs += [0.0, height, None]
    return xs, ys, zs


def render_room_map(snapshot, room_config):
    """PANEL 4 — 3D room: walls, receivers, live TX→RX links, person."""
    width = float(room_config["width_meters"])
    depth = float(room_config["depth_meters"])
    figure = go.Figure()
    gx, gy = _grid_lines(0.0, width, 0.0, depth, 0.5)
    figure.add_trace(go.Scatter3d(
        x=gx, y=gy, z=[0 if v is not None else None for v in gx],
        mode="lines", line={"color": "#0f3b47", "width": 1},
        hoverinfo="skip", showlegend=False))
    wx, wy, wz = _room_wireframe(width, depth, ROOM_HEIGHT_METERS)
    figure.add_trace(go.Scatter3d(
        x=wx, y=wy, z=wz, mode="lines",
        line={"color": "#1f6f86", "width": 3}, hoverinfo="skip",
        showlegend=False))
    positions = room_config.get("device_positions", {})
    activity = (snapshot.get("analytics") or {}).get("link_activity", {})
    transmitter = None
    for key, position in positions.items():
        if str(position.get("label", "")).upper().startswith("TX"):
            transmitter = position
    if transmitter is not None:
        for key, position in positions.items():
            if position is transmitter:
                continue
            level = float(activity.get(int(key), 0.0))
            glow = min(1.0, level / 1.5)
            color = (f"rgb({int(34 + 200 * glow)}, "
                     f"{int(211 - 120 * glow)}, {int(238 - 60 * glow)})")
            figure.add_trace(go.Scatter3d(
                x=[transmitter["x"], position["x"]],
                y=[transmitter["y"], position["y"]],
                z=[DEVICE_HEIGHT_METERS] * 2, mode="lines",
                line={"color": color, "width": 3 + 9 * glow},
                hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter3d(
        x=[p["x"] for p in positions.values()],
        y=[p["y"] for p in positions.values()],
        z=[DEVICE_HEIGHT_METERS] * len(positions),
        mode="markers+text", text=[p["label"] for p in positions.values()],
        textposition="top center", textfont={"color": TEXT_COLOR, "size": 10},
        marker={"color": ACCENT_CYAN, "size": 6, "symbol": "square"},
        hoverinfo="skip", showlegend=False))
    person = snapshot["position_estimate"]
    pose_label = snapshot["pose_label"]
    if person is not None and snapshot["presence"] and pose_label:
        keypoints = snapshot["keypoints"]
        if keypoints is None:
            keypoints = POSE_TEMPLATES[pose_label]
        xs, ys, zs = lift_keypoints_3d(keypoints, pose_label)
        px, py = float(person[0]), float(person[1])
        bx, by, bz = [], [], []
        for a, b in SKELETON_BONES:
            bx += [px + xs[a], px + xs[b], None]
            by += [py + ys[a], py + ys[b], None]
            bz += [zs[a], zs[b], None]
        figure.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode="lines",
            line={"color": "#fb923c", "width": 6}, hoverinfo="skip",
            showlegend=False))
        figure.add_trace(go.Scatter3d(
            x=[px + xs[0]], y=[py + ys[0]], z=[zs[0] + 0.05],
            mode="markers", marker={"size": 8, "color": "#fb923c"},
            hoverinfo="skip", showlegend=False))
        figure.add_trace(go.Scatter3d(
            x=[px], y=[py], z=[0.0], mode="markers+text",
            text=[f"({px:.1f}, {py:.1f}) m"], textposition="bottom center",
            textfont={"color": "#fdba74", "size": 10},
            marker={"size": 5, "color": "#fb923c", "symbol": "circle-open"},
            hoverinfo="skip", showlegend=False))
    axis = {"visible": False, "showbackground": False}
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=260,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        font={"family": FONT_STACK, "size": 11},
        uirevision="room",
        scene={"xaxis": {**axis, "range": [-0.2, width + 0.2]},
               "yaxis": {**axis, "range": [-0.2, depth + 0.2]},
               "zaxis": {**axis, "range": [0, ROOM_HEIGHT_METERS + 0.1]},
               "aspectmode": "data",
               "camera": {"eye": {"x": -1.05, "y": -1.5, "z": 1.1},
                          "center": {"x": 0, "y": 0, "z": -0.15}}},
    )
    return figure


def _empty_figure(message, height=200):
    """Placeholder figure with a centred message."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False,
                          font={"color": MUTED_TEXT, "size": 12})
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False}, yaxis={"visible": False})
    return figure


def render_vitals(snapshot):
    """PANEL — respiration rate (BNR-weighted FFT) with waveform."""
    analytics = snapshot.get("analytics") or {}
    breathing = analytics.get("breathing")
    still = snapshot["presence"] and snapshot["pose_label"] in (
        "standing", "sitting", "lying")
    if not still or breathing is None:
        reason = ("waiting for a still person" if snapshot["presence"]
                  else "no person detected")
        return html.Div([
            html.Div("-- br/min", style={"fontSize": "34px",
                                         "color": MUTED_TEXT}),
            html.Div(f"respiration: {reason}",
                     style={"color": MUTED_TEXT, "fontSize": "12px"}),
        ], style={"padding": "40px 0", "textAlign": "center"})
    waveform = breathing["waveform"][-200:]
    rate = breathing["waveform_rate_hz"]
    times = [(i - len(waveform)) / rate for i in range(len(waveform))]
    figure = go.Figure(go.Scatter(
        x=times, y=waveform, mode="lines",
        line={"color": "#34d399", "width": 2}, hoverinfo="skip"))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_COLOR, height=150,
        margin={"l": 30, "r": 10, "t": 5, "b": 25},
        font={"family": FONT_STACK, "size": 10},
        xaxis={"title": "s", "gridcolor": GRID_COLOR},
        yaxis={"showticklabels": False, "gridcolor": GRID_COLOR},
        showlegend=False)
    per_device = analytics.get("breathing_per_device", {})
    detail = "  ".join(f"RX-{d}: {v:.1f}" for d, v in sorted(per_device.items()))
    return html.Div([
        html.Div([
            html.Span(f"{breathing['bpm']:.1f}", style={
                "fontSize": "40px", "fontWeight": "bold", "color": "#34d399",
                "textShadow": "0 0 12px rgba(52,211,153,0.6)"}),
            html.Span(" br/min", style={"color": MUTED_TEXT}),
            html.Span(f"   BNR {breathing['bnr']:.2f} · "
                      f"{breathing['devices']} RX · "
                      f"{breathing.get('window_seconds', 0):.0f} s window"
                      + (" · settling" if breathing.get(
                          "window_seconds", 99) < 20 else ""),
                      style={"color": MUTED_TEXT, "fontSize": "12px",
                             "marginLeft": "12px"}),
        ]),
        dcc.Graph(figure=figure, config={"displayModeBar": False}),
        html.Div(detail, style={"color": MUTED_TEXT, "fontSize": "11px"}),
    ])


def render_spectrogram(snapshot):
    """PANEL — CARM-style PCA + STFT motion spectrogram."""
    spectrogram = (snapshot.get("analytics") or {}).get("spectrogram")
    if spectrogram is None:
        return _empty_figure("collecting 4 s of CSI…", 230)
    figure = go.Figure(go.Heatmap(
        z=spectrogram["power"], x=spectrogram["times"],
        y=spectrogram["freqs"], colorscale="Plasma", showscale=False,
        hoverinfo="skip"))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_COLOR, height=230,
        margin={"l": 40, "r": 10, "t": 24, "b": 30},
        font={"family": FONT_STACK, "size": 10},
        title={"text": f"RX-{spectrogram['device_id']} · motion index "
                       f"{spectrogram['speed_index']:.2f} Hz",
               "x": 0.02, "font": {"size": 11, "color": TEXT_COLOR}},
        xaxis={"title": "seconds ago"}, yaxis={"title": "Hz"})
    return figure


def render_heatmap(snapshot):
    """PANEL — per-band body-shadow attenuation over the last seconds."""
    heatmap = (snapshot.get("analytics") or {}).get("heatmap")
    if heatmap is None:
        return _empty_figure("collecting CSI…", 230)
    values = heatmap["values"]
    columns = len(values[0]) if values else 0
    seconds = heatmap["seconds"]
    figure = go.Figure(go.Heatmap(
        z=values, x=[-seconds + seconds * (i + 1) / columns
                     for i in range(columns)],
        y=list(range(len(values))), colorscale="Viridis", zmin=-0.1,
        zmax=0.6, showscale=False, hoverinfo="skip"))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_COLOR, height=230,
        margin={"l": 40, "r": 10, "t": 24, "b": 30},
        font={"family": FONT_STACK, "size": 10},
        title={"text": f"RX-{heatmap['device_id']} · attenuation vs "
                       "empty room", "x": 0.02,
               "font": {"size": 11, "color": TEXT_COLOR}},
        xaxis={"title": "seconds ago"}, yaxis={"title": "subcarrier band"})
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
    return rows or html.Div("no events yet", style={"color": MUTED_TEXT})


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
        html.Div(advice, style={"color": MUTED_TEXT}),
        html.Div(f"frames processed: {snapshot['frames_processed']:,}",
                 style={"marginTop": "12px", "fontSize": "12px",
                        "color": MUTED_TEXT}),
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
        return html.Div("no devices seen yet", style={"color": MUTED_TEXT})
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
        Output("panel-vitals", "children"),
        Output("panel-spectrogram", "figure"),
        Output("panel-heatmap", "figure"),
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
            render_vitals(snapshot),
            render_spectrogram(snapshot),
            render_heatmap(snapshot),
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
