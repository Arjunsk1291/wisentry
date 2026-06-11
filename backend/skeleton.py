"""17-point skeleton estimation (EXPERIMENTAL tier).

Produces COCO-ordered keypoints for the dashboard's pose figure. Per
ENGINEERING_SPEC.md §2, skeleton estimation from commodity ESP32 CSI is research
grade: this module anchors the skeleton on calibrated pose templates and
lets the (optional) skeleton regression model refine joint offsets. All
outputs are normalized to a 0..1 box (x right, y down) and must always be
presented as experimental in the UI and docs.

Runs on: any OS with Python 3.9+. Dependencies: numpy.
"""

import numpy as np

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KEYPOINT_COUNT = len(COCO_KEYPOINT_NAMES)  # 17

# Skeleton connectivity for drawing (pairs of keypoint indices).
SKELETON_BONES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),       # arms + shoulder line
    (5, 11), (6, 12), (11, 12),                     # torso
    (11, 13), (13, 15), (12, 14), (14, 16),         # legs
    (0, 5), (0, 6),                                 # neck approximation
]

MODEL_REFINEMENT_SCALE = 0.08  # max joint offset the ML model may add
MOTION_JITTER_SCALE = 0.01     # subtle CSI-driven liveliness on the figure

# Canonical normalized templates, COCO order, (x, y) with y growing downward.
_STANDING_TEMPLATE = np.array([
    [0.50, 0.06], [0.48, 0.05], [0.52, 0.05], [0.46, 0.06], [0.54, 0.06],
    [0.42, 0.20], [0.58, 0.20], [0.38, 0.36], [0.62, 0.36],
    [0.36, 0.50], [0.64, 0.50], [0.45, 0.52], [0.55, 0.52],
    [0.44, 0.72], [0.56, 0.72], [0.44, 0.94], [0.56, 0.94],
])
_SITTING_TEMPLATE = np.array([
    [0.50, 0.18], [0.48, 0.17], [0.52, 0.17], [0.46, 0.18], [0.54, 0.18],
    [0.42, 0.32], [0.58, 0.32], [0.38, 0.46], [0.62, 0.46],
    [0.40, 0.58], [0.60, 0.58], [0.45, 0.60], [0.55, 0.60],
    [0.62, 0.62], [0.66, 0.64], [0.62, 0.88], [0.66, 0.88],
])
_LYING_TEMPLATE = np.array([
    [0.06, 0.74], [0.07, 0.72], [0.07, 0.76], [0.08, 0.71], [0.08, 0.77],
    [0.20, 0.70], [0.20, 0.80], [0.32, 0.66], [0.32, 0.84],
    [0.42, 0.64], [0.42, 0.86], [0.52, 0.72], [0.52, 0.78],
    [0.72, 0.72], [0.72, 0.78], [0.92, 0.72], [0.92, 0.78],
])
_WALKING_TEMPLATE = np.array([
    [0.50, 0.06], [0.48, 0.05], [0.52, 0.05], [0.46, 0.06], [0.54, 0.06],
    [0.42, 0.20], [0.58, 0.20], [0.34, 0.32], [0.66, 0.32],
    [0.30, 0.44], [0.70, 0.44], [0.45, 0.52], [0.55, 0.52],
    [0.34, 0.70], [0.64, 0.72], [0.28, 0.92], [0.70, 0.94],
])
POSE_TEMPLATES = {
    "standing": _STANDING_TEMPLATE,
    "sitting": _SITTING_TEMPLATE,
    "lying": _LYING_TEMPLATE,
    "walking": _WALKING_TEMPLATE,
}


class SkeletonEstimator:
    """Template-anchored skeleton estimator with optional ML refinement."""

    def __init__(self):
        """Initialise with a fixed RNG so jitter is reproducible in tests."""
        self._random_generator = np.random.default_rng(2026)

    def estimate(self, pose_label, motion_energy, model_offsets=None):
        """Estimate 17 normalized keypoints for the current pose.

        Args:
            pose_label (str): One of POSE_TEMPLATES' keys ("standing",
                "sitting", "lying", "walking").
            motion_energy (float): Current motion energy; scales the subtle
                CSI-driven jitter so the figure looks alive when moving.
            model_offsets (np.ndarray | None): Optional float[17, 2] joint
                offsets from the skeleton regression model, clipped to
                MODEL_REFINEMENT_SCALE.

        Returns:
            np.ndarray: float32[17, 2] keypoints in a normalized 0..1 box.
        """
        if pose_label not in POSE_TEMPLATES:
            raise ValueError(
                f"Unknown pose label '{pose_label}'; expected one of "
                f"{sorted(POSE_TEMPLATES)}"
            )
        keypoints = POSE_TEMPLATES[pose_label].copy()
        jitter_amount = MOTION_JITTER_SCALE * min(motion_energy, 5.0)
        keypoints += self._random_generator.normal(
            0.0, jitter_amount, size=keypoints.shape
        )
        if model_offsets is not None:
            clipped_offsets = np.clip(
                model_offsets, -MODEL_REFINEMENT_SCALE, MODEL_REFINEMENT_SCALE
            )
            keypoints += clipped_offsets
        return np.clip(keypoints, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    estimator = SkeletonEstimator()
    for pose_name in POSE_TEMPLATES:
        estimated = estimator.estimate(pose_name, motion_energy=1.0)
        assert estimated.shape == (KEYPOINT_COUNT, 2)
        assert estimated.min() >= 0.0 and estimated.max() <= 1.0
    try:
        estimator.estimate("flying", 0.0)
        raise AssertionError("Unknown pose should have raised ValueError")
    except ValueError:
        pass
    print("skeleton self-test: PASS (4 templates, bounds, unknown-pose guard)")
