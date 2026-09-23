"""Volumetric body rendering for the dashboard (display only).

Turns the 17 lifted keypoints into a solid, shaded figure: tapered limb
capsules, torso and pelvis volumes, and a head sphere, merged into one
Mesh3d so the browser draws a single WebGL object. Depth comes from the
display lift in dashboard.app, not from a 3D estimate.
"""

import numpy as np
import plotly.graph_objects as go

# (joint a, joint b, radius at a, radius at b) in metres.
LIMB_SEGMENTS = [
    (5, 7, 0.055, 0.045), (7, 9, 0.045, 0.035),      # left arm
    (6, 8, 0.055, 0.045), (8, 10, 0.045, 0.035),     # right arm
    (11, 13, 0.08, 0.06), (13, 15, 0.06, 0.045),     # left leg
    (12, 14, 0.08, 0.06), (14, 16, 0.06, 0.045),     # right leg
]
RING_SIDES = 14
SPHERE_STEPS = 12


def _orthonormal_frame(direction):
    """Two unit vectors perpendicular to direction."""
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(direction @ helper) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    first = np.cross(direction, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(direction, first)


def _tapered_tube(start, end, radius_start, radius_end):
    """Vertices and faces of an open tapered tube from start to end."""
    first, second = _orthonormal_frame(end - start)
    angles = np.linspace(0, 2 * np.pi, RING_SIDES, endpoint=False)
    ring = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    vertices = np.vstack([start + radius_start * ring, end + radius_end * ring])
    faces = []
    for i in range(RING_SIDES):
        j = (i + 1) % RING_SIDES
        faces.append((i, j, RING_SIDES + i))
        faces.append((j, RING_SIDES + j, RING_SIDES + i))
    return vertices, np.array(faces)


def _ellipsoid(center, radii):
    """Vertices and faces of a UV ellipsoid."""
    theta = np.linspace(0, np.pi, SPHERE_STEPS)
    phi = np.linspace(0, 2 * np.pi, SPHERE_STEPS * 2, endpoint=False)
    vertices = []
    for t in theta:
        for p in phi:
            vertices.append(center + radii * np.array(
                [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]))
    columns = len(phi)
    faces = []
    for row in range(len(theta) - 1):
        for col in range(columns):
            a = row * columns + col
            b = row * columns + (col + 1) % columns
            faces.append((a, b, a + columns))
            faces.append((b, b + columns, a + columns))
    return np.array(vertices), np.array(faces)


def body_mesh(xs, ys, zs, offset=(0.0, 0.0), scale=1.0):
    """Merged vertices/faces/intensity for one body.

    Args:
        xs, ys, zs (list[float]): Lifted keypoints (17 each).
        offset (tuple): (x, y) placement in the scene.
        scale (float): Uniform scale (radii scale too).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: vertices, faces,
            per-vertex intensity (height, for the gradient shading).
    """
    points = np.column_stack([xs, ys, zs]).astype(float) * scale
    points[:, 0] += offset[0]
    points[:, 1] += offset[1]
    parts = []
    for a, b, radius_a, radius_b in LIMB_SEGMENTS:
        parts.append(_tapered_tube(points[a], points[b], radius_a * scale,
                                   radius_b * scale))
    for joint in (5, 6, 7, 8, 11, 12, 13, 14, 9, 10, 15, 16):
        radius = (0.06 if joint in (11, 12, 13, 14) else 0.05) * scale
        parts.append(_ellipsoid(points[joint], np.array([radius] * 3)))
    shoulder_mid = (points[5] + points[6]) / 2
    hip_mid = (points[11] + points[12]) / 2
    torso_axis = shoulder_mid - hip_mid
    torso_len = max(np.linalg.norm(torso_axis), 1e-6)
    first, second = _orthonormal_frame(torso_axis)
    shoulder_width = np.linalg.norm(points[5] - points[6]) / 2
    # Torso: an ellipsoid spanning hips to shoulders, rotated into place.
    vertices, faces = _ellipsoid(np.zeros(3), np.array(
        [max(shoulder_width, 0.12 * scale) * 1.05, 0.11 * scale,
         torso_len * 0.58]))
    across = points[6] - points[5]
    across = across / max(np.linalg.norm(across), 1e-9)
    up = torso_axis / torso_len
    depth_axis = np.cross(up, across)
    rotation = np.column_stack([across, depth_axis, up])
    parts.append((vertices @ rotation.T + (shoulder_mid + hip_mid) / 2, faces))
    neck_top = shoulder_mid + up * 0.08 * scale
    parts.append(_tapered_tube(shoulder_mid, neck_top, 0.045 * scale,
                               0.04 * scale))
    head_center = points[0] + np.array([0, 0, 0.04 * scale])
    parts.append(_ellipsoid(head_center, np.array(
        [0.095, 0.105, 0.12]) * scale))
    all_vertices, all_faces, offset_count = [], [], 0
    for vertices, faces in parts:
        all_vertices.append(vertices)
        all_faces.append(faces + offset_count)
        offset_count += len(vertices)
    vertices = np.vstack(all_vertices)
    return vertices, np.vstack(all_faces), vertices[:, 2]


def body_mesh_trace(xs, ys, zs, colorscale, offset=(0.0, 0.0), scale=1.0,
                    opacity=0.92):
    """Plotly Mesh3d trace with soft specular lighting."""
    vertices, faces, intensity = body_mesh(xs, ys, zs, offset, scale)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=intensity, colorscale=colorscale, showscale=False,
        opacity=opacity, flatshading=False, hoverinfo="skip",
        lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.9,
                  "roughness": 0.35, "fresnel": 0.6},
        lightposition={"x": 200, "y": -300, "z": 800})


def ellipsoid_trace(center, radii, rotation, color, opacity):
    """Transparent ellipsoid (e.g. a Fresnel zone) as a Mesh3d."""
    vertices, faces = _ellipsoid(np.zeros(3), np.asarray(radii, float))
    vertices = vertices @ np.asarray(rotation).T + np.asarray(center)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color,
        opacity=opacity, hoverinfo="skip", flatshading=False,
        lighting={"ambient": 0.9, "diffuse": 0.2, "specular": 0.1})
