"""Dashboard renderer tests (no server)."""

from backend.skeleton import POSE_TEMPLATES
from dashboard.app import lift_keypoints_3d, render_pose_3d


def test_lift_keypoints_3d_shapes_and_heights():
    for pose_label, keypoints in POSE_TEMPLATES.items():
        xs, ys, zs = lift_keypoints_3d(keypoints, pose_label)
        assert len(xs) == len(ys) == len(zs) == 17
        assert min(zs) >= 0.0
    _, _, standing_z = lift_keypoints_3d(POSE_TEMPLATES["standing"],
                                         "standing")
    _, _, lying_z = lift_keypoints_3d(POSE_TEMPLATES["lying"], "lying")
    assert max(standing_z) > 1.5
    assert max(lying_z) < 0.6


def test_render_pose_3d_builds_figure():
    figure = render_pose_3d(POSE_TEMPLATES["sitting"], "sitting", 0.8)
    assert len(figure.data) > 0
    assert "SITTING" in figure.layout.title.text
