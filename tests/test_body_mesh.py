"""Volumetric body mesh tests."""

import numpy as np

from backend.skeleton import POSE_TEMPLATES
from dashboard.app import lift_keypoints_3d
from dashboard.body_mesh import body_mesh


def test_body_mesh_is_valid_and_placed():
    xs, ys, zs = lift_keypoints_3d(POSE_TEMPLATES["standing"], "standing")
    vertices, faces, _ = body_mesh(xs, ys, zs, offset=(2.0, 1.0))
    assert faces.max() < len(vertices)
    assert abs(vertices[:, 0].mean() - 2.0) < 0.3
    assert abs(vertices[:, 1].mean() - 1.0) < 0.3
    assert 1.6 < vertices[:, 2].max() < 1.95
    assert np.isfinite(vertices).all()
